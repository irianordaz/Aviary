"""External subsystem builders for multi-fuel engine simulation in Aviary.

This module provides OpenMDAO components and Aviary subsystem builders that enable
simulating aircraft operations with different engine decks and fuel types across
different mission phases. It supports the common scenario where an aircraft uses
different fuel types (e.g., Jet-A for climb, SAF blend for cruise, LNG for descent)
each with its own performance characteristics and fuel density.

The module provides three main classes:

    MultiEngineFuelBurnComp
        An OpenMDAO ExplicitComponent that computes per-engine total fuel burn
        (both mass and volume) from phase start/end gross-mass inputs. It sums
        fuel burned across phases that share the same engine deck and fuel density.

    EngineTableBuilder
        A thin wrapper around Aviary's EngineDeck that reads engine performance
        data from a CSV file. Used internally by MultiEngineTableBuilder to create
        individual engine models for each mission phase.

    MultiEngineTableBuilder
        The primary entry point for multi-fuel simulation. Extends EngineModel to
        manage a mapping of mission phases to different engine decks (CSV files) with
        optional per-phase fuel densities. It configures phase info, builds the
        post-mission fuel burn component, and wires up the necessary trajectory
        connections.

Usage
-----
The typical workflow involves creating a MultiEngineTableBuilder with a
phase_engine_map that maps each mission phase to an engine deck CSV file and
optional fuel density, configuring the phase_info dict, and then using the
engine as an external subsystem in an AviaryProblem::

    from copy import deepcopy
    from multi_fuel.table_builder import MultiEngineTableBuilder

    engine = MultiEngineTableBuilder(
        phase_engine_map={
            'climb':   ('models/engines/turbofan_28k.csv', 6.7),
            'cruise':  ('models/engines/turbofan_22k.csv', 6.4),
            'descent': ('models/engines/turbofan_22k.csv', 6.4),
        },
    )
    phase_info = engine.configure_phase_info(deepcopy(phase_info))

See multi_fuel/run_single_aisle.py for a complete end-to-end example.

Variables
---------
_DEFAULT_FUEL_DENSITY_LBM_GAL : float
    The Aviary default fuel density (typically 6.7 lbm/galUS for Jet-A) used
    when no per-phase override is provided.
"""

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
    """Compute per-engine total fuel burn from phase start/end gross-mass inputs.

    This OpenMDAO component calculates the total fuel consumed during each mission
    phase by taking the difference between the aircraft's gross mass at the start
    and end of each phase. It supports multiple engine decks with different fuel
    densities by grouping phases that share the same ``(csv_path, fuel_density)``
    pair and summing their fuel burn separately.

    The component is designed to be used as a post-mission component that receives
    mass timeseries data from trajectory phases. For each phase, it expects two
    scalar inputs representing the aircraft mass at the beginning and end of the
    phase. The fuel burned for each phase is computed as:

        fuel_burned = mass_start - mass_end

    These values are then aggregated by unique ``(csv, density)`` pair and output
    as both mass (lbm) and volume (galUS) using the appropriate fuel density.

    The component supports gradient computation for use with gradient-based
    optimizers such as IPOPT. Each output's derivative with respect to its
    corresponding mass inputs is either +1.0 (for mass_start) or -1.0
    (for mass_end), and the volume derivatives are scaled by the inverse of
    the fuel density.

    Parameters
    ----------
    phase_engine_map : dict
        Mapping of phase name to ``(csv_path, fuel_density_lbm_per_galUS)`` tuple.
        This is passed via the ``options`` argument during component construction.

    Inputs
    ------
    mass_start_{phase} : float, units='lbm'
        Aircraft gross mass at the start of the named phase. One input per phase
        in ``phase_engine_map``.

    mass_end_{phase} : float, units='lbm'
        Aircraft gross mass at the end of the named phase. One input per phase
        in ``phase_engine_map``.

    Outputs
    -------
    Mission.TOTAL_FUEL_MULTI : ndarray, units='lbm'
        Total fuel mass burned per-engine, organized by unique ``(csv, density)``
        pair in the order of their first appearance in ``phase_engine_map``. Each
        element represents the sum of fuel burned across all phases sharing that
        engine deck and fuel density.

    Mission.TOTAL_FUEL_VOLUME_MULTI : ndarray, units='galUS'
        Total fuel volume burned per-engine, computed by dividing the mass totals
        by the corresponding fuel density. Useful for fuel system sizing and
        reporting in gallons.

    Examples
    --------
    >>> from multi_fuel.table_builder import MultiEngineFuelBurnComp
    >>> phase_map = {
    ...     'climb': ('engines/turbofan.csv', 6.7),
    ...     'cruise': ('engines/turbofan.csv', 6.4),
    ... }
    >>> comp = MultiEngineFuelBurnComp(phase_engine_map=phase_map)

    Notes
    -----
    Phases sharing the same ``(csv, density)`` tuple have their fuel burn summed
    into a single output element. This allows the same engine model to be used
    across multiple phases while tracking fuel consumption separately for different
    fuel types.

    The component assumes that mass_start >= mass_end for each phase (i.e., the
    aircraft loses mass due to fuel burn). Negative fuel values may result if
    this assumption is violated.
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
    """EngineDeck builder that reads engine performance data from a CSV file.

    A convenience wrapper around Aviary's EngineDeck that simplifies the creation
    of engine models from CSV data files. It automatically resolves relative paths
    to absolute paths within the Aviary models directory and accepts an optional
    AviaryValues object for additional engine configuration.

    This class is primarily used internally by MultiEngineTableBuilder to create
    individual engine models for each mission phase. It inherits all functionality
    from EngineDeck, including the ability to interpolate engine performance data
    across operating conditions such as throttle setting, Mach number, and altitude.

    The CSV file is expected to contain engine performance data in a format
    compatible with Aviary's EngineDeck parser. Typical columns include throttle,
    Mach number, altitude, and various performance outputs such as thrust, fuel flow,
    and emissions indices.

    Parameters
    ----------
    name : str, optional
        Label for this engine model used in OpenMDAO component naming and output
        variable identification. Default is 'engine_table'.
    csv_path : str, optional
        Path to the engine performance CSV data file. Accepts paths relative to the
        Aviary models directory (e.g. 'models/engines/turbofan_22k.csv') or
        absolute file system paths. Relative paths are resolved using Aviary's
        path resolution utilities. Default is 'models/engines/turbofan_22k.csv'.
    options : AviaryValues, optional
        Additional engine configuration options to pass to the underlying EngineDeck.
        All required options not explicitly provided here are filled from Aviary
        metadata defaults via the _preprocess_inputs mechanism. If None, an empty
        AviaryValues container is created.

    Notes
    -----
    This class extends EngineDeck and delegates all engine deck construction logic
    to the parent class. Its primary purpose is to provide a simpler interface for
    creating individual engine models within a multi-engine configuration.

    The resolved CSV path is stored in the AviaryValues under the
    Aircraft.Engine.DATA_FILE key before being passed to the parent constructor.

    See Also
    --------
    MultiEngineTableBuilder : Uses this class internally to create per-phase engines.
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
    """Engine builder with per-phase engine decks and fuel densities.

    A comprehensive engine subsystem builder that manages a collection of engine
    models, each associated with a specific mission phase. This enables simulation
    of aircraft operations where different engine configurations and fuel types are
    used across different phases of flight (e.g., Jet-A for climb, SAF blend for
    cruise, LNG for descent).

    The class extends Aviary's EngineModel base class and integrates with the
    Aviary framework's subsystem architecture. It handles engine deck selection,
    phase info configuration, post-mission fuel burn computation, and trajectory
    variable connections.

    The core data structure is the ``phase_engine_map``, which maps mission phase
    names to engine deck CSV files and optional fuel densities. Each entry can be
    either a simple string path (using Aviary's default fuel density) or a tuple
    of ``(csv_path, fuel_density)`` for custom fuel density per phase.

    Attributes
    ----------
    phase_engine_map : dict
        Mapping of phase name to ``(csv_path, fuel_density_lbm_per_galUS)`` tuple.
        String values from the constructor are automatically converted to tuples
        using the default fuel density.
    _phase_engines : dict
        Internal cache of EngineTableBuilder instances, one per phase. Created
        during initialization with the appropriate fuel density set on each
        engine's options.
    compute_max_values : bool
        Class attribute set to True to match EngineDeck convention, indicating
        that this builder computes max-throttle values per condition.

    Parameters
    ----------
    name : str, optional
        Label for this engine model. Also serves as the subsystem key when looking
        up per-phase options in phase_info's subsystem_options. Default is
        'multi_engine_table'.
    phase_engine_map : dict, optional
        Mapping of phase name to engine configuration. Each value can be:

        - A string containing the CSV path (uses default fuel density)
        - A ``(csv_path, density)`` tuple for custom fuel density

        Paths may be Aviary-relative (e.g. 'models/engines/turbofan_22k.csv')
        or absolute file system paths. Default is None, resulting in an empty
        phase_engine_map.

    Examples
    --------
    Create a multi-engine configuration with different fuel types per phase::

        from copy import deepcopy
        from multi_fuel.table_builder import MultiEngineTableBuilder

        engine = MultiEngineTableBuilder(
            phase_engine_map={
                'climb':   ('models/engines/turbofan_28k.csv', 6.7),
                'cruise':  ('models/engines/turbofan_22k.csv', 6.4),
                'descent': ('models/engines/turbofan_22k.csv', 6.4),
            },
        )

    Configure phase info and integrate with AviaryProblem::

        phase_info = engine.configure_phase_info(deepcopy(phase_info))
        prob = AviaryProblem()
        prob.load_external_subsystems([engine])

    Methods
    -------
    configure_phase_info
        Inject phase names into subsystem_options for engine deck dispatch.
    build_post_mission
        Create the MultiEngineFuelBurnComp for post-mission fuel accounting.
    build_mission
        Select and build the appropriate engine deck for the current phase.
    get_traj_connections
        Return trajectory mass timeseries connections for fuel burn computation.
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
        """Inject phase names into subsystem_options for engine deck dispatch.

        Modifies the provided phase_info dictionary in-place by injecting the
        appropriate phase name into each phase's subsystem_options hierarchy.
        This enables the build_mission method to select the correct engine deck
        for each phase during simulation.

        The phase name is nested under a specific hierarchy of subsystem options:
        first under the core propulsion builder's key (typically 'propulsion'),
        then under this builder's name (self.name). This structure allows the
        PropulsionMission subsystem to look up engine options by engine name
        and retrieve the phase name for dispatch.

        Pre-mission and post-mission phases are skipped since they do not
        correspond to trajectory phases that use engine decks.

        Parameters
        ----------
        phase_info : dict
            The phase_info dictionary to configure. Contains phase names as keys
            and phase options as values. Modified in-place.
        propulsion_name : str, optional
            The key name for the core propulsion builder subsystem within each
            phase's subsystem_options. Default is 'propulsion'.

        Returns
        -------
        dict
            The modified phase_info dictionary with phase names injected into
            the subsystem_options hierarchy. Same object as the input phase_info.

        Notes
        -----
        This method must be called before passing phase_info to AviaryProblem's
        load_inputs method. It creates the necessary nested dictionary structure
        if it does not already exist, using setdefault to avoid overwriting
        existing options.

        The resulting subsystem_options structure for each phase looks like:

            phase_options['subsystem_options'][propulsion_name][self.name]['phase_name']

        Examples
        --------
        >>> engine = MultiEngineTableBuilder(phase_engine_map={'climb': 'engines.csv'})
        >>> configured = engine.configure_phase_info(deepcopy(phase_info))
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
        """Return the list of output variables to promote from the post-mission subsystem.

        Returns the OpenMDAO variable name for the total multi-fuel mass burn
        computed by the MultiEngineFuelBurnComp post-mission component. These
        variables are promoted to the top-level problem namespace for easy
        access after optimization.

        Returns
        -------
        list of str
            A list containing the Mission.TOTAL_FUEL_MULTI variable name. The
            volume output (Mission.TOTAL_FUEL_VOLUME_MULTI) is not promoted
            since mass is the primary optimization objective.

        See Also
        --------
        build_post_mission : Creates the component that produces these outputs.
        """
        return [Mission.TOTAL_FUEL_MULTI]

    def build_post_mission(
        self,
        aviary_inputs=None,
        mission_info=None,
        subsystem_options=None,
        phase_mission_bus_lengths=None,
    ):
        """Build the post-mission fuel burn accounting component.

        Creates and returns a MultiEngineFuelBurnComp OpenMDAO component that
        computes total per-engine fuel burn (both mass and volume) from the
    mass inputs provided by trajectory phase timeseries. This component is
        added to the post-mission subsystem and receives mass data from each
        configured mission phase.

        The component sums fuel burned across phases that share the same engine
        deck and fuel density pair, producing separate outputs for each unique
        combination. This enables tracking fuel consumption by fuel type when
        different fuels are used across phases.

        Parameters
        ----------
        aviary_inputs : AviaryValues, optional
            The Aviary inputs containing aircraft and engine configuration.
            Not used by this method but included for interface compatibility.
        mission_info : dict, optional
            Information about the mission configuration. Not used by this method.
        subsystem_options : dict, optional
            Options for subsystems. Not used by this method.
        phase_mission_bus_lengths : list, optional
            Lengths of mission bus segments. Not used by this method.

        Returns
        -------
        MultiEngineFuelBurnComp or None
            A configured MultiEngineFuelBurnComp instance if phase_engine_map
            is non-empty, or None if no phases are configured.

        Notes
        -----
        The returned component must have its inputs connected to trajectory
        mass timeseries via get_traj_connections. The component computes:

            fuel_mass = mass_start - mass_end

        for each phase and aggregates by unique (csv, density) pair.

        See Also
        --------
        get_traj_connections : Returns the connections needed for this component.
        MultiEngineFuelBurnComp : The component class that performs the computation.
        """
        if not self.phase_engine_map:
            return None
        return MultiEngineFuelBurnComp(
            phase_engine_map=self.phase_engine_map,
        )

    def get_traj_connections(self, regular_phases):
        """Return trajectory variable connections for fuel burn component inputs.

        Generates a list of OpenMDAO connection specifications that wire the
        mass timeseries data from each configured mission phase to the corresponding
        inputs of the MultiEngineFuelBurnComp post-mission component.

        For each phase in ``phase_engine_map`` that also appears in ``regular_phases``,
        two connections are created:
        1. Mass at the start of the phase (index 0 of the timeseries)
        2. Mass at the end of the phase (last index -1 of the timeseries)

        These connections enable the fuel burn component to compute fuel consumed
        as the difference between phase start and end masses.

        Parameters
        ----------
        regular_phases : iterable of str
            The set of regular (non-pre/post) mission phase names that have
            trajectory mass timeseries available. Phases not in this set are
            skipped even if they appear in phase_engine_map.

        Returns
        -------
        list of tuple
            Each tuple contains three elements:
                (source_path, target_path, indices)
            where:
                source_path : str
                    The trajectory mass timeseries variable path, e.g.
                    'traj.climb.timeseries.mass'.
                target_path : str
                    The fuel burn component input path, e.g.
                    'multi_engine_table.mass_start_climb'.
                indices : list of int
                    The indices into the timeseries array: [0] for start mass,
                    [-1] for end mass.

        Examples
        --------
        >>> connections = engine.get_traj_connections(['climb', 'cruise', 'descent'])
        >>> for src, tgt, idx in connections:
        ...     print(f'{src} -> {tgt}[{idx}]')
        traj.climb.timeseries.mass -> multi_engine_table.mass_start_climb[0]
        traj.climb.timeseries.mass -> multi_engine_table.mass_end_climb[-1]
        ...
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
        """Return the first configured engine for pre-mission sizing operations.

        Retrieves the first EngineTableBuilder instance from the internal
        _phase_engines cache. This engine is used for pre-mission sizing
        calculations such as determining required thrust and fuel for
        takeoff and initial climb.

        The "first" engine is determined by the insertion order of the
        phase_engine_map dictionary, which follows Python 3.7+ dict ordering
        guarantees. In practice, this is typically the climb phase engine
        since it is usually the first phase defined.

        Returns
        -------
        EngineTableBuilder
            The first EngineTableBuilder instance from the configured phases.
            This engine provides the baseline configuration for pre-mission
            sizing calculations.

        Raises
        ------
        ValueError
            Raised if no engines have been configured (empty phase_engine_map).
            This indicates a configuration error.

        Notes
        -----
        This is a private method (indicated by the leading underscore) used
        internally by build_pre_mission. The engine returned should typically
        represent the most demanding operating condition for sizing purposes.

        See Also
        --------
        build_pre_mission : Uses this method to delegate pre-mission sizing.
        """
        if not self._phase_engines:
            raise ValueError(f'{self.name}: phase_engine_map is empty.')
        return next(iter(self._phase_engines.values()))

    def build_pre_mission(self, aviary_inputs, subsystem_options=None):
        """Build the pre-mission subsystem using the first configured engine deck.

        Delegates pre-mission subsystem construction to the default engine
        (the first engine in phase_engine_map). The pre-mission subsystem
 typically handles operations such as takeoff, climb to cruise altitude,
        and any other activities that occur before the main mission phases.

        The method uses _default_engine to select the appropriate engine model
        and forwards all arguments to its build_pre_mission method. This ensures
        consistent engine configuration between pre-mission and mission phases.

        Parameters
        ----------
        aviary_inputs : AviaryValues
            The Aviary inputs containing aircraft configuration, engine sizing
            data, and other mission parameters. Forwarded to the engine's
            build_pre_mission method.
        subsystem_options : dict, optional
            Additional subsystem options for the pre-mission phase. Forwarded
            to the engine's build_pre_mission method.

        Returns
        -------
        OpenMDAO Component or Group
            The pre-mission subsystem built by the default engine. This may be
            a component or a group containing multiple components depending on
            the engine configuration.

        See Also
        --------
        _default_engine : Selects the engine used for this operation.
        EngineTableBuilder.build_pre_mission : The underlying implementation.
        """
        return self._default_engine().build_pre_mission(aviary_inputs, subsystem_options)

    def build_mission(self, num_nodes, aviary_inputs, user_options, subsystem_options):
        """Select and build the engine deck system for the current mission phase.

        Looks up the engine deck configuration for the phase specified in
        ``subsystem_options['phase_name']`` and delegates mission system
        construction to the corresponding EngineTableBuilder instance.

        This method is called by the Aviary framework during mission model
        construction for each trajectory phase. The phase name is retrieved
        from the subsystem_options dictionary, which was previously populated
        by configure_phase_info.

        If the requested phase has no configured engine, a KeyError is raised
        with a descriptive message listing the available phases. This helps
        diagnose configuration errors where a phase name does not match any
        entry in phase_engine_map.

        Parameters
        ----------
        num_nodes : int
            The number of discretization nodes for the mission phase. Used by
            the engine's build_mission method to shape the output variables.
        aviary_inputs : AviaryValues
            The Aviary inputs containing aircraft configuration and mission
            parameters. Forwarded to the engine's build_mission method.
        user_options : dict
            User-specified options for the mission phase. Forwarded to the
            engine's build_mission method.
        subsystem_options : dict
            Options for the current subsystem, expected to contain a 'phase_name'
            key that identifies which engine deck to use.

        Returns
        -------
        OpenMDAO Component or Group
            The mission subsystem built by the selected engine. This may be a
            component or a group containing multiple components depending on the
            engine deck configuration and mission method.

        Raises
        ------
        KeyError
            Raised if no engine is configured for the requested phase. The error
            message includes the requested phase name and a list of available
            phases to help diagnose configuration issues.

        Examples
        --------
        When configured with phases 'climb', 'cruise', and 'descent', calling
        this method with subsystem_options={'phase_name': 'cruise'} will return
        the engine system built from the cruise phase's engine deck.

        See Also
        --------
        configure_phase_info : Populates subsystem_options with phase names.
        EngineTableBuilder.build_mission : The underlying implementation.
        """
        phase_name = subsystem_options.get('phase_name')
        engine = self._phase_engines.get(phase_name)
        if engine is None:
            available = list(self._phase_engines)
            raise KeyError(
                f'{self.name}: no engine configured for phase "{phase_name}". '
                f'Available phases: {available}'
            )
        return engine.build_mission(num_nodes, aviary_inputs, user_options, subsystem_options)
