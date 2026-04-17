"""External subsystem builders that create engine decks from CSV data files."""

from aviary.subsystems.propulsion.engine_deck import EngineDeck
from aviary.subsystems.propulsion.engine_model import EngineModel
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import get_path
from aviary.variable_info.variables import Aircraft


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
    Engine builder that uses a different CSV engine deck for each mission phase.

    Parameters
    ----------
    name : str
        Label for this engine model. Also used as the subsystem key when looking up
        per-phase options in phase_info's subsystem_options.
    phase_csv_map : dict
        Mapping of phase name to engine CSV path, e.g.
        ``{'climb': 'models/engines/engine1.csv', 'cruise': 'models/engines/engine2.csv'}``.
        Paths may be Aviary-relative or absolute.

    Usage
    -----
    Build the engine, then call ``configure_phase_info`` so each phase's
    ``subsystem_options`` carries the phase name for dispatch in ``build_mission``::

        engine = MultiEngineTableBuilder(
            phase_csv_map={
                'climb':  'models/engines/turbofan_22k.csv',
                'cruise': 'models/engines/turbofan_24k_1.csv',
                'descent': 'models/engines/turbofan_22k.csv',
            }
        )
        local_phase_info = engine.configure_phase_info(deepcopy(phase_info))
        prob.load_external_subsystems([engine])
    """

    # Match EngineDeck: the builder itself computes max-throttle values per condition
    compute_max_values = True

    def __init__(self, name: str = 'multi_engine_table', phase_csv_map: dict = None):
        super().__init__(name=name)
        self.phase_csv_map = phase_csv_map or {}
        # Pre-create and fully initialise one EngineTableBuilder per phase so CSV
        # data is read once at construction time, not on every build_mission call.
        self._phase_engines = {
            phase: EngineTableBuilder(name=f'{name}_{phase}', csv_path=csv)
            for phase, csv in self.phase_csv_map.items()
        }

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
            phase_opts['subsystem_options'][propulsion_name][self.name]['phase_name'] = phase_name
        return phase_info

    def _default_engine(self) -> EngineTableBuilder:
        """Return the first configured engine; used for pre-mission sizing."""
        if not self._phase_engines:
            raise ValueError(f'{self.name}: phase_csv_map is empty.')
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
