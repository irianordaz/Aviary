"""External subsystem builders for multi-fuel engine simulation in Aviary.

This module provides OpenMDAO components and Aviary SubsystemBuilder-based classes
that enable simulating aircraft operations with different engine decks and fuel
types across different mission phases. It supports the common scenario where an
aircraft uses different fuel types (e.g., Jet-A for climb, SAF blend for cruise,
LNG for descent), each with its own performance characteristics and fuel density.

The module is implemented as a pure SubsystemBuilder-based extension: the main
class ``MultiEngineTableBuilder`` inherits from
``aviary.subsystems.subsystem_builder.SubsystemBuilder`` and relies only on the
framework hooks exposed by that base class (``build_post_mission`` in particular).
Per-phase fuel burn is computed post-mission from the mass timeseries of each
trajectory phase.

Classes
-------
MultiEngineFuelBurnComp
    OpenMDAO ExplicitComponent that computes per-engine total fuel burn (mass
    and volume) from phase start/end gross-mass inputs, summing fuel burn across
    phases that share the same (csv, fuel_density) pair.

EngineTableBuilder
    Thin wrapper around Aviary's EngineDeck that resolves a CSV path and sets
    the per-engine Aircraft.Fuel.DENSITY option. It is used as the actual
    propulsion EngineModel registered with the AviaryProblem.

MultiEngineTableBuilder
    SubsystemBuilder-based engine builder that holds a mapping of mission phases
    to engine decks (with optional per-phase fuel densities). It is registered
    as an external subsystem; its ``build_post_mission`` adds the
    ``MultiEngineFuelBurnComp`` to the post-mission group, and its
    ``wire_trajectory`` helper connects phase mass timeseries to that component
    after ``AviaryProblem.build_model()``.

Variable names
--------------
TOTAL_MULTI_FUEL_MASS, TOTAL_MULTI_FUEL_VOLUME
    Local output variable names used by ``MultiEngineFuelBurnComp``. These are
    defined here rather than in Aviary's Mission namespace so the module works
    with an unmodified Aviary install.

Usage
-----
    from copy import deepcopy
    from multi_fuel.table_builder import MultiEngineTableBuilder

    engine = MultiEngineTableBuilder(
        phase_engine_map={
            'climb':   ('models/engines/turbofan_28k.csv', 6.7),
            'cruise':  ('models/engines/turbofan_22k.csv', 6.4),
            'descent': ('models/engines/turbofan_22k.csv', 6.4),
        },
    )

    prob = AviaryProblem()
    prob.load_inputs(inputs, phase_info)
    prob.load_external_subsystems([engine])
    prob.check_and_preprocess_inputs()
    prob.build_model()
    engine.wire_trajectory(prob.model)
    prob.setup()
    prob.run_aviary_problem()
"""

import numpy as np
import openmdao.api as om

from aviary.subsystems.propulsion.engine_deck import EngineDeck
from aviary.subsystems.propulsion.propulsion_builder import CorePropulsionBuilder
from aviary.subsystems.propulsion.propulsion_mission import PropulsionMission
from aviary.subsystems.subsystem_builder import SubsystemBuilder
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import get_path
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.variable_info.variables import Aircraft

# Output variable names produced by MultiEngineFuelBurnComp. Defined locally so
# the module does not require Aviary to register new entries in Mission.
TOTAL_MULTI_FUEL_MASS = 'mission:total_multi_fuel_mass'
TOTAL_MULTI_FUEL_VOLUME = 'mission:total_multi_fuel_volume'

# Aviary default fuel density used when no override is provided.
_DEFAULT_FUEL_DENSITY_LBM_GAL = _MetaData[Aircraft.Fuel.DENSITY]['default_value']


class MultiEngineFuelBurnComp(om.ExplicitComponent):
    """Compute per-engine total fuel burn from phase start/end gross-mass inputs.

    Sums ``mass_start - mass_end`` for each phase in ``phase_engine_map`` and
    aggregates by unique ``(csv, fuel_density)`` pair. Outputs both mass (lbm)
    and volume (galUS) totals per unique engine configuration.
    """

    def initialize(self):
        self.options.declare('phase_engine_map', recordable=False)

    def setup(self):
        phase_engine_map = self.options['phase_engine_map']
        for phase in phase_engine_map:
            self.add_input(f'mass_start_{phase}', val=0.0, units='lbm')
            self.add_input(f'mass_end_{phase}', val=0.0, units='lbm')

        unique_entries = list(dict.fromkeys(phase_engine_map.values()))
        self._unique_entries = unique_entries
        num_entries = len(unique_entries)
        self.add_output(TOTAL_MULTI_FUEL_MASS, val=0.0, shape=num_entries, units='lbm')
        self.add_output(
            TOTAL_MULTI_FUEL_VOLUME, val=0.0, shape=num_entries, units='galUS'
        )

    def setup_partials(self):
        phase_engine_map = self.options['phase_engine_map']
        unique_entries = self._unique_entries

        for phase, entry in phase_engine_map.items():
            idx = unique_entries.index(entry)
            density = entry[1]

            self.declare_partials(
                TOTAL_MULTI_FUEL_MASS,
                f'mass_start_{phase}',
                rows=[idx],
                cols=[0],
                val=1.0,
            )
            self.declare_partials(
                TOTAL_MULTI_FUEL_MASS,
                f'mass_end_{phase}',
                rows=[idx],
                cols=[0],
                val=-1.0,
            )
            self.declare_partials(
                TOTAL_MULTI_FUEL_VOLUME,
                f'mass_start_{phase}',
                rows=[idx],
                cols=[0],
                val=1.0 / density,
            )
            self.declare_partials(
                TOTAL_MULTI_FUEL_VOLUME,
                f'mass_end_{phase}',
                rows=[idx],
                cols=[0],
                val=-1.0 / density,
            )

    def compute(self, inputs, outputs):
        phase_engine_map = self.options['phase_engine_map']
        unique_entries = self._unique_entries
        num_entries = len(unique_entries)
        mass_totals = np.zeros(num_entries)

        for phase, entry in phase_engine_map.items():
            phase_fuel_mass = (
                inputs[f'mass_start_{phase}'].flat[0]
                - inputs[f'mass_end_{phase}'].flat[0]
            )
            mass_totals[unique_entries.index(entry)] += phase_fuel_mass

        densities = np.array([entry[1] for entry in unique_entries])
        outputs[TOTAL_MULTI_FUEL_MASS] = mass_totals
        outputs[TOTAL_MULTI_FUEL_VOLUME] = mass_totals / densities


class EngineTableBuilder(EngineDeck):
    """EngineDeck wrapper that reads engine performance data from a CSV file.

    Resolves a CSV path via ``aviary.utils.functions.get_path`` and stores it in
    an AviaryValues options container under ``Aircraft.Engine.DATA_FILE`` before
    delegating construction to ``EngineDeck``. An optional per-engine
    ``Aircraft.Fuel.DENSITY`` override may be set on the caller-provided options.

    Parameters
    ----------
    name : str
        Label for this engine model. Default ``'engine_table'``.
    csv_path : str
        Path to the engine performance CSV data file, either relative to the
        Aviary models directory or absolute. Default
        ``'models/engines/turbofan_22k.csv'``.
    options : AviaryValues, optional
        Additional engine configuration options. If None, an empty AviaryValues
        container is created.
    """

    def __init__(
        self,
        name: str = 'engine_table',
        csv_path: str = 'models/engines/turbofan_22k.csv',
        options: AviaryValues = None,
    ):
        if options is None:
            options = AviaryValues()
        options.set_val(Aircraft.Engine.DATA_FILE, get_path(csv_path))
        super().__init__(name=name, options=options)


class MultiPhasePropulsionBuilder(CorePropulsionBuilder):
    """Phase-aware drop-in replacement for ``CorePropulsionBuilder``.

    Aviary's ``CorePropulsionBuilder`` always builds its ``PropulsionMission``
    from a single, fixed list of ``engine_models`` and uses that same engine in
    every phase. This subclass overrides ``build_mission`` to instead pick the
    engine model registered for the current phase (via ``phase_engines``) and
    build a ``PropulsionMission`` around that engine. The phase identity is
    read from ``subsystem_options['phase_name']`` (populated upstream by
    ``MultiEngineTableBuilder.configure_phase_info``).

    Falling back to ``default_engine`` for unknown phases keeps callers from
    accidentally breaking when a phase is not in ``phase_engines`` (e.g., a
    reserve phase added later).

    Parameters
    ----------
    name : str
        Builder name. Must remain ``'propulsion'`` so it slots into
        ``AviaryGroup.subsystems[0]`` correctly.
    phase_engines : dict, optional
        Mapping ``{phase_name: EngineModel}`` — the engine to use in each
        phase's ``PropulsionMission``.
    default_engine : EngineModel
        Engine used when ``subsystem_options['phase_name']`` is missing or
        unrecognized; also used by ``build_pre_mission`` (which is called once
        and is not phase-aware).
    """

    def __init__(
        self,
        name: str = 'propulsion',
        meta_data=None,
        phase_engines: dict = None,
        default_engine=None,
    ):
        super().__init__(
            name=name, meta_data=meta_data, engine_models=[default_engine]
        )
        self._phase_engines = phase_engines or {}
        self._default_engine = default_engine

    def build_mission(self, num_nodes, aviary_inputs, user_options, subsystem_options):
        phase_name = (subsystem_options or {}).get('phase_name')
        engine = self._phase_engines.get(phase_name, self._default_engine)
        return PropulsionMission(
            num_nodes=num_nodes,
            aviary_options=aviary_inputs,
            engine_models=[engine],
            user_options=user_options,
            engine_options={},
        )


class MultiEngineTableBuilder(SubsystemBuilder):
    """SubsystemBuilder that wires multi-fuel engine decks per mission phase.

    Holds a mapping of mission phases to engine deck CSV files and optional
    per-phase fuel densities, and provides three integration hooks:

    - ``configure_phase_info``: tags each phase's ``subsystem_options`` with
      its ``phase_name`` under the propulsion subsystem's key, so the
      per-phase engine selector knows which engine to use.
    - ``install_propulsion``: swaps Aviary's default ``CorePropulsionBuilder``
      with a ``MultiPhasePropulsionBuilder`` that selects the per-phase engine
      when each phase's ODE is built.
    - ``build_post_mission``: contributes a component that aggregates fuel
      burn (mass and volume) per unique ``(csv, density)`` pair from the
      phase mass timeseries, after the trajectory has been wired.

    Parameters
    ----------
    name : str
        Subsystem name used in OpenMDAO paths and when identifying the
        post-mission component within the ``post_mission`` group. Default
        ``'multi_engine_table'``.
    phase_engine_map : dict, optional
        Mapping of phase name to engine configuration. Each value can be either
        a string CSV path (uses default fuel density) or a ``(csv_path,
        density)`` tuple. Paths may be Aviary-relative or absolute. Default
        None (empty mapping).
    """

    def __init__(
        self,
        name: str = 'multi_engine_table',
        phase_engine_map: dict = None,
    ):
        super().__init__(name=name)

        raw_phase_engine_map = phase_engine_map or {}
        self.phase_engine_map = {}
        for phase, phase_entry in raw_phase_engine_map.items():
            if isinstance(phase_entry, str):
                csv_path = phase_entry
                fuel_density = _DEFAULT_FUEL_DENSITY_LBM_GAL
            else:
                csv_path, fuel_density = phase_entry
            self.phase_engine_map[phase] = (csv_path, fuel_density)

        self._phase_engines = {}
        for phase, (csv_path, fuel_density) in self.phase_engine_map.items():
            engine_options = AviaryValues()
            engine_options.set_val(
                Aircraft.Fuel.DENSITY, fuel_density, 'lbm/galUS'
            )
            self._phase_engines[phase] = EngineTableBuilder(
                name=f'{name}_{phase}', csv_path=csv_path, options=engine_options
            )

    def configure_phase_info(
        self, phase_info: dict, propulsion_name: str = 'propulsion'
    ) -> dict:
        """Tag each phase's subsystem_options with its name under ``propulsion``.

        Aviary's mission ODE passes ``phase_info[phase]['subsystem_options']
        [propulsion_name]`` to the propulsion builder's ``build_mission`` as
        ``subsystem_options``. ``MultiPhasePropulsionBuilder`` reads
        ``subsystem_options['phase_name']`` to pick the engine for that phase,
        so we set that key here.

        Parameters
        ----------
        phase_info : dict
            The phase_info dictionary to configure. ``pre_mission`` and
            ``post_mission`` keys are skipped.
        propulsion_name : str
            Name of the propulsion subsystem in Aviary (default
            ``'propulsion'``). Must match the ``name`` of the
            ``MultiPhasePropulsionBuilder`` installed in
            ``install_propulsion``.

        Returns
        -------
        dict
            The same ``phase_info`` dict, with ``phase_name`` set at
            ``phase_info[phase]['subsystem_options'][propulsion_name]['phase_name']``.
        """
        skip = {'pre_mission', 'post_mission'}
        for phase_name, phase_opts in phase_info.items():
            if phase_name in skip:
                continue
            phase_opts.setdefault('subsystem_options', {}).setdefault(
                propulsion_name, {}
            )['phase_name'] = phase_name
        return phase_info

    def install_propulsion(self, aviary_group, propulsion_name: str = 'propulsion'):
        """Swap Aviary's ``CorePropulsionBuilder`` for a phase-aware one.

        Must be called after ``check_and_preprocess_inputs()`` (which adds
        ``CorePropulsionBuilder`` to ``aviary_group.subsystems``) and before
        ``build_model()`` (which materializes phase ODEs from those
        subsystems). The swap preserves the existing builder name so that
        ``configure_phase_info`` continues to address the right
        ``subsystem_options`` key.

        Parameters
        ----------
        aviary_group : AviaryGroup
            Typically ``prob.model``. Its ``subsystems`` list is mutated in
            place; the first builder named ``propulsion_name`` is replaced.
        propulsion_name : str
            Name of the propulsion subsystem to replace.
        """
        if not self._phase_engines:
            raise ValueError(f'{self.name}: phase_engine_map is empty.')
        default_engine = next(iter(self._phase_engines.values()))
        for subsystem_index, subsystem in enumerate(aviary_group.subsystems):
            if getattr(subsystem, 'name', None) == propulsion_name:
                aviary_group.subsystems[subsystem_index] = MultiPhasePropulsionBuilder(
                    name=propulsion_name,
                    phase_engines=self._phase_engines,
                    default_engine=default_engine,
                )
                return
        raise RuntimeError(
            f'{self.name}: could not find a subsystem named '
            f'{propulsion_name!r} to replace; call after '
            'check_and_preprocess_inputs().'
        )

    def build_post_mission(
        self,
        aviary_inputs=None,
        mission_info=None,
        subsystem_options=None,
        phase_mission_bus_lengths=None,
    ):
        """Return the ``MultiEngineFuelBurnComp`` for post-mission fuel accounting.

        The returned component is added by ``AviaryGroup`` to its
        ``post_mission`` group under this builder's ``name``. After
        ``build_model()``, inputs must be connected via ``wire_trajectory``.
        """
        if not self.phase_engine_map:
            return None
        return MultiEngineFuelBurnComp(phase_engine_map=self.phase_engine_map)

    def wire_trajectory(self, aviary_group):
        """Connect phase mass timeseries to the post-mission fuel burn component.

        Must be called after ``AviaryProblem.build_model()`` (so the trajectory
        and post-mission group exist) and before ``setup()``. Only phases that
        appear in both ``phase_engine_map`` and
        ``aviary_group.regular_phases`` are connected.

        Input paths are resolved at the AviaryGroup level since the
        ``post_mission`` group promotes its inputs with ``*``.
        """
        regular_phases = set(aviary_group.regular_phases)
        for phase in self.phase_engine_map:
            if phase not in regular_phases:
                continue
            aviary_group.connect(
                f'traj.{phase}.timeseries.mass',
                f'{self.name}.mass_start_{phase}',
                src_indices=[0],
            )
            aviary_group.connect(
                f'traj.{phase}.timeseries.mass',
                f'{self.name}.mass_end_{phase}',
                src_indices=[-1],
            )
