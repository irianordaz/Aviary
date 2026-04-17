"""External subsystem builders that create engine decks from CSV data files."""

import numpy as np
import openmdao.api as om

from aviary.subsystems.propulsion.engine_deck import EngineDeck
from aviary.subsystems.propulsion.engine_model import EngineModel
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import get_path
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.variable_info.variables import Aircraft, Mission

# Aviary default fuel density used when no override is provided.
_DEFAULT_FUEL_DENSITY_LBM_GAL = _MetaData[Aircraft.Fuel.DENSITY]['default_value']


class MultiEngineFuelBurnComp(om.ExplicitComponent):
    """
    Compute per-engine total fuel burn (mass and volume) from phase start/end
    gross-mass inputs.

    For each phase in ``phase_engine_map`` the component receives two scalar
    inputs (``mass_start_{phase}`` and ``mass_end_{phase}``) and computes the
    fuel burned as their difference.  Results are summed across phases that share
    the same ``(csv, density)`` entry.

    Outputs
    -------
    Mission.TOTAL_FUEL_MULTI : array, lbm
        Fuel mass burned per unique ``(csv, density)`` pair, ordered by first
        appearance in ``phase_engine_map``.
    Mission.TOTAL_FUEL_VOLUME_MULTI : array, galUS
        Fuel volume burned per unique ``(csv, density)`` pair.
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
        self.add_output(Mission.TOTAL_FUEL_MULTI, val=np.zeros(n), units='lbm')
        self.add_output(Mission.TOTAL_FUEL_VOLUME_MULTI, val=np.zeros(n), units='galUS')

    def setup_partials(self):
        phase_engine_map = self.options['phase_engine_map']
        unique_entries = self._unique_entries

        for phase, entry in phase_engine_map.items():
            idx = unique_entries.index(entry)
            density = entry[1]

            self.declare_partials(
                Mission.TOTAL_FUEL_MULTI, f'mass_start_{phase}',
                rows=[idx], cols=[0], val=1.0,
            )
            self.declare_partials(
                Mission.TOTAL_FUEL_MULTI, f'mass_end_{phase}',
                rows=[idx], cols=[0], val=-1.0,
            )
            self.declare_partials(
                Mission.TOTAL_FUEL_VOLUME_MULTI, f'mass_start_{phase}',
                rows=[idx], cols=[0], val=1.0 / density,
            )
            self.declare_partials(
                Mission.TOTAL_FUEL_VOLUME_MULTI, f'mass_end_{phase}',
                rows=[idx], cols=[0], val=-1.0 / density,
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
        outputs[Mission.TOTAL_FUEL_MULTI] = mass_totals
        outputs[Mission.TOTAL_FUEL_VOLUME_MULTI] = mass_totals / densities


class EngineTableBuilder(EngineDeck):
    """
    EngineDeck builder that reads engine performance data from a CSV file.

    Parameters
    ----------
    name : str
        Label for this engine model.
    csv_path : str
        Path to engine performance CSV. Accepts paths relative to the Aviary models
        directory (e.g. 'models/engines/turbofan_22k.csv') or absolute paths.
    options : AviaryValues, optional
        Additional engine options forwarded to EngineDeck. All required options not
        provided here are filled from Aviary metadata defaults by _preprocess_inputs.
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


class MultiEngineTableBuilder(EngineModel):
    """
    Engine builder that uses a different CSV engine deck for each mission phase,
    with an optional per-phase fuel density.

    Parameters
    ----------
    name : str
        Label for this engine model. Also used as the subsystem key when looking up
        per-phase options in phase_info's subsystem_options.
    phase_engine_map : dict
        Mapping of phase name to either a CSV path string or a
        ``(csv_path, density_lbm_per_galUS)`` tuple.  Using tuples allows the same
        engine CSV to carry a different fuel density in different phases.  Paths may
        be Aviary-relative or absolute.  String values fall back to Aviary's default
        fuel density (``Aircraft.Fuel.DENSITY``, currently 6.7 lbm/galUS).

    Usage
    -----
    Build the engine, then call ``configure_phase_info`` so each phase's
    ``subsystem_options`` carries the phase name for dispatch in ``build_mission``::

        engine = MultiEngineTableBuilder(
            phase_engine_map={
                'climb':   ('models/engines/turbofan_28k.csv', 6.7),   # Jet-A
                'cruise':  ('models/engines/turbofan_22k.csv', 6.5),   # SAF blend
                'descent': ('models/engines/turbofan_22k.csv', 6.5),
            },
        )
        local_phase_info = engine.configure_phase_info(deepcopy(phase_info))
        prob.load_external_subsystems([engine])
    """

    # Match EngineDeck: the builder itself computes max-throttle values per condition
    compute_max_values = True

    def __init__(
        self,
        name: str = 'multi_engine_table',
        phase_engine_map: dict = None,
    ):
        super().__init__(name=name)

        # Normalize values to (csv, density) tuples.
        raw_map = phase_engine_map or {}
        self.phase_engine_map = {}
        for phase, val in raw_map.items():
            if isinstance(val, str):
                self.phase_engine_map[phase] = (val, _DEFAULT_FUEL_DENSITY_LBM_GAL)
            else:
                csv, density = val
                self.phase_engine_map[phase] = (csv, density)

        # Pre-create one EngineTableBuilder per phase; set Aircraft.Fuel.DENSITY
        # on each engine's options so downstream sizing code sees the right value.
        self._phase_engines = {}
        for phase, (csv, density) in self.phase_engine_map.items():
            opts = AviaryValues()
            opts.set_val(Aircraft.Fuel.DENSITY, density, 'lbm/galUS')
            self._phase_engines[phase] = EngineTableBuilder(
                name=f'{name}_{phase}', csv_path=csv, options=opts
            )

    def configure_phase_info(
        self, phase_info: dict, propulsion_name: str = 'propulsion'
    ) -> dict:
        """
        Inject phase names into each phase's subsystem_options so build_mission
        can dispatch to the correct engine deck.

        Phase name is nested under the core propulsion builder's subsystem_options key
        so it flows through PropulsionMission's engine_options lookup by engine name.

        Modifies phase_info in-place and returns it.  Call this before passing
        phase_info to AviaryProblem.

        Parameters
        ----------
        phase_info : dict
            The phase_info dict to configure.
        propulsion_name : str
            Name of the CorePropulsionBuilder subsystem (default 'propulsion').
        """
        skip = {'pre_mission', 'post_mission'}
        for phase_name, phase_opts in phase_info.items():
            if phase_name in skip:
                continue
            phase_opts.setdefault('subsystem_options', {})
            phase_opts['subsystem_options'].setdefault(propulsion_name, {})
            phase_opts['subsystem_options'][propulsion_name].setdefault(self.name, {})
            phase_opts['subsystem_options'][propulsion_name][self.name][
                'phase_name'
            ] = phase_name
        return phase_info

    def get_post_mission_promotes_outputs(self):
        return [Mission.TOTAL_FUEL_MULTI]

    def build_post_mission(
        self,
        aviary_inputs=None,
        mission_info=None,
        subsystem_options=None,
        phase_mission_bus_lengths=None,
    ):
        """Return a component that sums per-phase fuel burn into Mission.TOTAL_FUEL_MULTI
        and Mission.TOTAL_FUEL_VOLUME_MULTI."""
        if not self.phase_engine_map:
            return None
        return MultiEngineFuelBurnComp(
            phase_engine_map=self.phase_engine_map,
        )

    def get_traj_connections(self, regular_phases):
        """
        Return connection specs for wiring trajectory mass timeseries into the
        MultiEngineFuelBurnComp inputs (start and end mass for every configured phase).
        """
        connections = []
        for phase in self.phase_engine_map:
            if phase not in regular_phases:
                continue
            connections.append((
                f'traj.{phase}.timeseries.mass',
                f'{self.name}.mass_start_{phase}',
                [0],
            ))
            connections.append((
                f'traj.{phase}.timeseries.mass',
                f'{self.name}.mass_end_{phase}',
                [-1],
            ))
        return connections

    def _default_engine(self) -> EngineTableBuilder:
        """Return the first configured engine; used for pre-mission sizing."""
        if not self._phase_engines:
            raise ValueError(f'{self.name}: phase_engine_map is empty.')
        return next(iter(self._phase_engines.values()))

    def build_pre_mission(self, aviary_inputs, subsystem_options=None):
        """Delegate pre-mission sizing to the first configured engine deck."""
        return self._default_engine().build_pre_mission(aviary_inputs, subsystem_options)

    def build_mission(self, num_nodes, aviary_inputs, user_options, subsystem_options):
        """Select the engine deck for the current phase and build its mission system."""
        phase_name = subsystem_options.get('phase_name')
        engine = self._phase_engines.get(phase_name)
        if engine is None:
            available = list(self._phase_engines)
            raise KeyError(
                f'{self.name}: no engine configured for phase "{phase_name}". '
                f'Available phases: {available}'
            )
        return engine.build_mission(num_nodes, aviary_inputs, user_options, subsystem_options)
