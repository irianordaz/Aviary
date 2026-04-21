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
TOTAL_FUEL_MULTI, TOTAL_FUEL_VOLUME_MULTI
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
    prob.load_external_subsystems([engine.get_default_engine(), engine])
    prob.check_and_preprocess_inputs()
    prob.build_model()
    engine.wire_trajectory(prob.model)
    prob.setup()
    prob.run_aviary_problem()
"""

import numpy as np
import openmdao.api as om

from aviary.subsystems.propulsion.engine_deck import EngineDeck
from aviary.subsystems.subsystem_builder import SubsystemBuilder
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import get_path
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.variable_info.variables import Aircraft

# Output variable names produced by MultiEngineFuelBurnComp. Defined locally so
# the module does not require Aviary to register new entries in Mission.
TOTAL_FUEL_MULTI = 'mission:total_fuel_multi'
TOTAL_FUEL_VOLUME_MULTI = 'mission:total_fuel_volume_multi'

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
        n = len(unique_entries)
        self.add_output(TOTAL_FUEL_MULTI, val=0.0, shape=n, units='lbm')
        self.add_output(TOTAL_FUEL_VOLUME_MULTI, val=0.0, shape=n, units='galUS')

    def setup_partials(self):
        phase_engine_map = self.options['phase_engine_map']
        unique_entries = self._unique_entries

        for phase, entry in phase_engine_map.items():
            idx = unique_entries.index(entry)
            density = entry[1]

            self.declare_partials(
                TOTAL_FUEL_MULTI,
                f'mass_start_{phase}',
                rows=[idx],
                cols=[0],
                val=1.0,
            )
            self.declare_partials(
                TOTAL_FUEL_MULTI,
                f'mass_end_{phase}',
                rows=[idx],
                cols=[0],
                val=-1.0,
            )
            self.declare_partials(
                TOTAL_FUEL_VOLUME_MULTI,
                f'mass_start_{phase}',
                rows=[idx],
                cols=[0],
                val=1.0 / density,
            )
            self.declare_partials(
                TOTAL_FUEL_VOLUME_MULTI,
                f'mass_end_{phase}',
                rows=[idx],
                cols=[0],
                val=-1.0 / density,
            )

    def compute(self, inputs, outputs):
        phase_engine_map = self.options['phase_engine_map']
        unique_entries = self._unique_entries
        n = len(unique_entries)
        mass_totals = np.zeros(n)

        for phase, entry in phase_engine_map.items():
            fuel = inputs[f'mass_start_{phase}'].flat[0] - inputs[f'mass_end_{phase}'].flat[0]
            mass_totals[unique_entries.index(entry)] += fuel

        densities = np.array([entry[1] for entry in unique_entries])
        outputs[TOTAL_FUEL_MULTI] = mass_totals
        outputs[TOTAL_FUEL_VOLUME_MULTI] = mass_totals / densities


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


class MultiEngineTableBuilder(SubsystemBuilder):
    """SubsystemBuilder that adds post-mission multi-fuel burn accounting.

    Holds a mapping of mission phases to engine deck CSV files and optional
    per-phase fuel densities. Registered with ``AviaryProblem`` as an external
    subsystem, it contributes only a post-mission component that aggregates fuel
    burn per unique (csv, density) pair using the mass timeseries of each phase.
    A reference engine (see ``get_default_engine``) is exposed for the user to
    register separately as Aviary's EngineModel for actual propulsion.

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

        raw_map = phase_engine_map or {}
        self.phase_engine_map = {}
        for phase, val in raw_map.items():
            if isinstance(val, str):
                self.phase_engine_map[phase] = (val, _DEFAULT_FUEL_DENSITY_LBM_GAL)
            else:
                csv, density = val
                self.phase_engine_map[phase] = (csv, density)

        self._phase_engines = {}
        for phase, (csv, density) in self.phase_engine_map.items():
            opts = AviaryValues()
            opts.set_val(Aircraft.Fuel.DENSITY, density, 'lbm/galUS')
            self._phase_engines[phase] = EngineTableBuilder(
                name=f'{name}_{phase}', csv_path=csv, options=opts
            )

    def get_default_engine(self) -> EngineTableBuilder:
        """Return the first configured engine for use as Aviary's EngineModel.

        The returned ``EngineTableBuilder`` should be passed to
        ``AviaryProblem.load_external_subsystems`` alongside this builder so
        that Aviary's propulsion subsystem has an engine to build. "First" is
        defined by insertion order of ``phase_engine_map``.
        """
        if not self._phase_engines:
            raise ValueError(f'{self.name}: phase_engine_map is empty.')
        return next(iter(self._phase_engines.values()))

    def configure_phase_info(self, phase_info: dict) -> dict:
        """Inject phase name into each phase's ``subsystem_options`` entry.

        Modifies ``phase_info`` in place so that when the mission ODE calls
        ``build_mission`` on this subsystem for a given phase, the
        ``subsystem_options`` argument contains ``phase_name``. That key is
        used by ``build_mission`` to dispatch to the correct per-phase engine
        deck from ``phase_engine_map``.

        Parameters
        ----------
        phase_info : dict
            The phase_info dictionary to configure. ``pre_mission`` and
            ``post_mission`` keys are skipped.

        Returns
        -------
        dict
            The same ``phase_info`` dict, with ``phase_name`` set at
            ``phase_info[phase]['subsystem_options'][self.name]['phase_name']``.
        """
        skip = {'pre_mission', 'post_mission'}
        for phase_name, phase_opts in phase_info.items():
            if phase_name in skip:
                continue
            phase_opts.setdefault('subsystem_options', {}).setdefault(self.name, {})[
                'phase_name'
            ] = phase_name
        return phase_info

    def build_mission(self, num_nodes, aviary_inputs, user_options, subsystem_options):
        """Build the per-phase engine deck's mission component.

        Reads ``phase_name`` from ``subsystem_options`` (set by
        ``configure_phase_info``) and delegates to the ``EngineTableBuilder``
        configured for that phase in ``phase_engine_map``. The resulting engine
        group is added to the phase ODE; its outputs remain at local paths
        (see ``mission_outputs``) so they do not collide with the main
        propulsion subsystem that is built from the engine returned by
        ``get_default_engine``.
        """
        if not subsystem_options:
            return None
        phase_name = subsystem_options.get('phase_name')
        if phase_name is None:
            return None
        engine = self._phase_engines.get(phase_name)
        if engine is None:
            return None
        return engine.build_mission(
            num_nodes=num_nodes,
            aviary_inputs=aviary_inputs,
            user_options=user_options,
            subsystem_options={},
        )

    def mission_inputs(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        """Promote all inputs of the per-phase engine group.

        The per-phase engine requires standard dynamic inputs (Mach, altitude,
        throttle, etc.); promoting with ``['*']`` lets them connect to the
        same sources used by the main propulsion subsystem.
        """
        return ['*']

    def mission_outputs(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        """Do not promote the per-phase engine's outputs.

        PropulsionMission already promotes the canonical
        ``Dynamic.Vehicle.Propulsion.*`` outputs. Promoting the per-phase
        engine's outputs would collide with those, so we keep them local
        under ``<phase ODE>/<self.name>/`` instead.
        """
        return []

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
