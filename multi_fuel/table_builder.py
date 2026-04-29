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
    ``MultiEngineFuelBurnComp`` to the post-mission group.

Module-level helpers
--------------------
configure_phase_info
    Inject per-phase ``csv_path`` and ``fuel_density`` into ``phase_info``'s
    ``subsystem_options`` so ``MultiPhasePropulsionBuilder`` can build the
    per-phase engine on the fly.
install_propulsion
    Swap Aviary's default ``CorePropulsionBuilder`` for a
    ``MultiPhasePropulsionBuilder`` after ``check_and_preprocess_inputs()``.
wire_trajectory
    Connect phase mass timeseries to the ``MultiEngineFuelBurnComp`` after
    ``build_model()``.

Variable names
--------------
TOTAL_MULTI_FUEL_MASS, TOTAL_MULTI_FUEL_VOLUME
    Local output variable names used by ``MultiEngineFuelBurnComp``. These are
    defined here rather than in Aviary's Mission namespace so the module works
    with an unmodified Aviary install.

Usage
-----
    from multi_fuel.table_builder import (
        MultiEngineTableBuilder,
        configure_phase_info,
        install_propulsion,
        wire_trajectory,
    )

    phase_engine_map = {
        'climb':   ('models/engines/turbofan_28k.csv', 6.7),
        'cruise':  ('models/engines/turbofan_22k.csv', 6.4),
        'descent': ('models/engines/turbofan_22k.csv', 6.4),
    }
    engine = MultiEngineTableBuilder(phase_engine_map=phase_engine_map)
    phase_info = configure_phase_info(phase_info, phase_engine_map)

    prob = AviaryProblem()
    prob.load_inputs(inputs, phase_info)
    prob.load_external_subsystems([engine])
    prob.check_and_preprocess_inputs()
    install_propulsion(prob.model, phase_engine_map)
    prob.build_model()
    wire_trajectory(prob.model, phase_engine_map, engine.name)
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
    every phase. This subclass overrides ``build_mission`` to instead build a
    fresh engine from ``subsystem_options['csv_path']`` and
    ``subsystem_options['fuel_density']`` (populated upstream by
    ``MultiEngineTableBuilder.configure_phase_info``) and wraps it in a
    ``PropulsionMission`` for that phase.

    Falling back to ``default_engine`` when ``csv_path`` is absent keeps
    callers from accidentally breaking when a phase is not configured (e.g., a
    reserve phase added later).

    Parameters
    ----------
    name : str
        Builder name. Must remain ``'propulsion'`` so it slots into
        ``AviaryGroup.subsystems[0]`` correctly.
    default_engine : EngineModel
        Engine used when ``subsystem_options`` lacks ``csv_path``; also used by
        ``build_pre_mission`` (which is called once and is not phase-aware).
    """

    def __init__(
        self,
        name: str = 'propulsion',
        meta_data=None,
        default_engine=None,
    ):
        super().__init__(
            name=name, meta_data=meta_data, engine_models=[default_engine]
        )
        self._default_engine = default_engine

    def build_mission(self, num_nodes, aviary_inputs, user_options, subsystem_options):
        opts = subsystem_options or {}
        csv_path = opts.get('csv_path')
        if csv_path is not None:
            engine_options = AviaryValues()
            engine_options.set_val(
                Aircraft.Fuel.DENSITY, opts['fuel_density'], 'lbm/galUS'
            )
            engine = EngineTableBuilder(
                name=f'{self.name}_{opts.get("phase_name", "phase")}',
                csv_path=csv_path,
                options=engine_options,
            )
        else:
            engine = self._default_engine
        return PropulsionMission(
            num_nodes=num_nodes,
            aviary_options=aviary_inputs,
            engine_models=[engine],
            user_options=user_options,
            engine_options={},
        )


class MultiEngineTableBuilder(SubsystemBuilder):
    """SubsystemBuilder that contributes the post-mission fuel-burn component.

    Holds a mapping of mission phases to engine deck CSV files and per-phase
    fuel densities. Registered as an external subsystem so its
    ``build_post_mission`` adds the ``MultiEngineFuelBurnComp`` to the
    post-mission group.

    The trajectory wiring (``wire_trajectory``), phase_info configuration
    (``configure_phase_info``), and propulsion swap (``install_propulsion``)
    live as module-level helpers; they take ``phase_engine_map`` directly so
    callers don't have to thread the builder instance through them.

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
                density_lbm_per_gal = _DEFAULT_FUEL_DENSITY_LBM_GAL
            else:
                csv_path, density_lbm_per_gal = phase_entry
            self.phase_engine_map[phase] = (csv_path, density_lbm_per_gal)

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


def configure_phase_info(
    phase_info: dict,
    phase_engine_map: dict,
    propulsion_name: str = 'propulsion',
) -> dict:
    """Inject per-phase engine config into ``subsystem_options[propulsion_name]``.

    Aviary's mission ODE passes ``phase_info[phase]['subsystem_options']
    [propulsion_name]`` to the propulsion builder's ``build_mission`` as
    ``subsystem_options``. ``MultiPhasePropulsionBuilder`` reads ``csv_path``
    and ``fuel_density`` from there to construct the per-phase engine on the
    fly.

    Parameters
    ----------
    phase_info : dict
        The phase_info dictionary to configure. ``pre_mission`` and
        ``post_mission`` keys are skipped.
    phase_engine_map : dict
        Mapping ``{phase_name: (csv_path, density)}``.
    propulsion_name : str
        Name of the propulsion subsystem in Aviary (default ``'propulsion'``).
        Must match the ``name`` of the ``MultiPhasePropulsionBuilder``
        installed in ``install_propulsion``.

    Returns
    -------
    dict
        The same ``phase_info`` dict, with ``phase_name``, ``csv_path``, and
        ``fuel_density`` set at
        ``phase_info[phase]['subsystem_options'][propulsion_name]``.
    """
    skip = {'pre_mission', 'post_mission'}
    for phase_name, phase_opts in phase_info.items():
        if phase_name in skip:
            continue
        opts = phase_opts.setdefault('subsystem_options', {}).setdefault(
            propulsion_name, {}
        )
        opts['phase_name'] = phase_name
        if phase_name in phase_engine_map:
            csv_path, density = phase_engine_map[phase_name]
            opts['csv_path'] = csv_path
            opts['fuel_density'] = density
    return phase_info


def install_propulsion(
    aviary_group,
    phase_engine_map: dict,
    propulsion_name: str = 'propulsion',
):
    """Swap Aviary's ``CorePropulsionBuilder`` for a phase-aware one.

    Must be called after ``check_and_preprocess_inputs()`` (which adds
    ``CorePropulsionBuilder`` to ``aviary_group.subsystems``) and before
    ``build_model()`` (which materializes phase ODEs from those subsystems).

    Parameters
    ----------
    aviary_group : AviaryGroup
        Typically ``prob.model``. Its ``subsystems`` list is mutated in place;
        the first builder named ``propulsion_name`` is replaced.
    phase_engine_map : dict
        Mapping ``{phase_name: (csv_path, density)}``. The first entry's
        ``(csv_path, density)`` is used to build the default engine that
        ``MultiPhasePropulsionBuilder`` falls back to when a phase is missing
        engine config in its subsystem_options.
    propulsion_name : str
        Name of the propulsion subsystem to replace.
    """
    if not phase_engine_map:
        raise ValueError('phase_engine_map is empty.')
    first_csv, first_density = next(iter(phase_engine_map.values()))
    default_options = AviaryValues()
    default_options.set_val(Aircraft.Fuel.DENSITY, first_density, 'lbm/galUS')
    default_engine = EngineTableBuilder(
        name='default_engine',
        csv_path=first_csv,
        options=default_options,
    )
    for subsystem_index, subsystem in enumerate(aviary_group.subsystems):
        if getattr(subsystem, 'name', None) == propulsion_name:
            aviary_group.subsystems[subsystem_index] = MultiPhasePropulsionBuilder(
                name=propulsion_name,
                default_engine=default_engine,
            )
            return
    raise RuntimeError(
        f'could not find a subsystem named {propulsion_name!r} to replace; '
        'call after check_and_preprocess_inputs().'
    )


def wire_trajectory(
    aviary_group,
    phase_engine_map: dict,
    subsystem_name: str = 'multi_engine_table',
):
    """Connect phase mass timeseries to the post-mission fuel burn component.

    Must be called after ``AviaryProblem.build_model()`` (so the trajectory
    and post-mission group exist) and before ``setup()``. Only phases that
    appear in both ``phase_engine_map`` and ``aviary_group.regular_phases``
    are connected.

    Input paths are resolved at the AviaryGroup level since the
    ``post_mission`` group promotes its inputs with ``*``.

    Parameters
    ----------
    aviary_group : AviaryGroup
        Typically ``prob.model``.
    phase_engine_map : dict
        Mapping ``{phase_name: (csv_path, density)}``; only the keys are used.
    subsystem_name : str
        Name of the ``MultiEngineTableBuilder`` whose
        ``MultiEngineFuelBurnComp`` lives at ``post_mission.<subsystem_name>``.
        Must match the ``name`` of the registered builder.
    """
    regular_phases = set(aviary_group.regular_phases)
    for phase in phase_engine_map:
        if phase not in regular_phases:
            continue
        aviary_group.connect(
            f'traj.{phase}.timeseries.mass',
            f'{subsystem_name}.mass_start_{phase}',
            src_indices=[0],
        )
        aviary_group.connect(
            f'traj.{phase}.timeseries.mass',
            f'{subsystem_name}.mass_end_{phase}',
            src_indices=[-1],
        )
