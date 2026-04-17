"""Multi-fuel engine subsystem for Aviary.

This package provides external subsystem builders that enable Aviary to use
different engine decks (CSV-based performance data) for different mission phases,
along with a component that computes per-engine total fuel burn from phase
start/end gross-mass inputs.

The primary use case is simulating aircraft operations where different fuel types
(Jet-A, SAF blend, LNG, etc.) are used across mission phases, each with its own
engine performance characteristics and fuel density.

Modules
-------
table_builder : Contains MultiEngineFuelBurnComp, EngineTableBuilder, and
    MultiEngineTableBuilder classes for multi-fuel engine simulation.

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
    phase_info = engine.configure_phase_info(deepcopy(phase_info))

See the run_single_aisle.py script for a complete end-to-end example.
"""
