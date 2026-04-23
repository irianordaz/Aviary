"""End-to-end benchmark for the multi-fuel single-aisle optimization.

Builds the full ``AviaryProblem`` used in ``run_single_aisle.py`` and verifies:

- Each mission phase's ODE contains a per-phase engine subsystem built from the
  CSV mapped to that phase in ``phase_engine_map``.
- Total fuel volume equals total fuel mass divided by the per-phase density
  (i.e., the density provided in ``phase_engine_map`` is the one used).
- The optimization converges and produces physically reasonable totals.
"""

import unittest
from copy import deepcopy

import openmdao.api as om
from openmdao.core.problem import _clear_problem_names
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

from aviary.core.aviary_problem import AviaryProblem
from aviary.models.aircraft.large_single_aisle_2.large_single_aisle_2_FLOPS_data import (
    inputs as lsa2_inputs,
)
from aviary.models.missions.energy_state_default import phase_info
from aviary.utils.functions import get_path
from aviary.variable_info.variables import Aircraft, Mission
from multi_fuel.table_builder import (
    TOTAL_MULTI_FUEL_MASS,
    TOTAL_MULTI_FUEL_VOLUME,
    MultiEngineTableBuilder,
)

_PHASE_ENGINE_MAP = {
    'climb': ('multi_fuel/engines/turbofan_28k.csv', 6.7),
    'cruise': ('multi_fuel/engines/turbofan_22k.csv', 6.4),
    'descent': ('multi_fuel/engines/turbofan_24k_1.csv', 6.4),
}


def _set_engine_inputs(inputs, phase_engine_map):
    first_csv = next(iter(phase_engine_map.values()))
    if isinstance(first_csv, tuple):
        first_csv = first_csv[0]
    inputs.set_val(Aircraft.Engine.DATA_FILE, get_path(first_csv))
    inputs.set_val(Aircraft.Engine.MASS, 6293.8, 'lbm')
    inputs.set_val(Aircraft.Engine.REFERENCE_MASS, 6293.8, 'lbm')
    inputs.set_val(Aircraft.Engine.REFERENCE_SLS_THRUST, 22200.5, 'lbf')
    inputs.set_val(Aircraft.Engine.SCALED_SLS_THRUST, 22200.5, 'lbf')
    inputs.set_val(Aircraft.Engine.SCALE_FACTOR, 1.0)


def _build_problem(phase_engine_map):
    """Build (but do not yet run) an AviaryProblem mirroring ``run_single_aisle``."""
    inputs = deepcopy(lsa2_inputs)
    _set_engine_inputs(inputs, phase_engine_map)
    phases = deepcopy(phase_info)

    engine = MultiEngineTableBuilder(phase_engine_map=phase_engine_map)
    phases = engine.configure_phase_info(phases)

    prob = AviaryProblem(verbosity=0)
    prob.load_inputs(inputs, phases)
    prob.load_external_subsystems([engine])
    prob.check_and_preprocess_inputs()
    # Swap the default CorePropulsionBuilder for a phase-aware one so each
    # phase's ODE is built around that phase's engine.
    engine.install_propulsion(prob.model)
    prob.build_model()
    engine.wire_trajectory(prob.model)
    prob.add_driver('IPOPT', max_iter=50, use_coloring=True)
    prob.add_design_variables()
    prob.add_objective()
    prob.setup()
    return prob, engine


@use_tempdirs
class MultiFuelBenchmarkTest(unittest.TestCase):
    """End-to-end benchmark for the multi-fuel LSA2 optimization."""

    def setUp(self):
        om.clear_reports()
        _clear_problem_names()

    def test_each_phase_ode_contains_its_engine_subsystem(self):
        """After ``build_model``, each phase's ODE must contain the per-phase engine.

        ``MultiPhasePropulsionBuilder`` (installed via
        ``engine.install_propulsion``) builds a per-phase ``PropulsionMission``
        using the engine registered for that phase. The mission group adds each
        engine as a subgroup under the engine's own name, giving the path
        ``traj.phases.{phase}.rhs_all.solver_sub.propulsion.{engine_name}``.
        The builder's per-phase engine is identified by the CSV it resolves
        ``Aircraft.Engine.DATA_FILE`` to; we cross-check that each phase's
        engine has the expected CSV.
        """
        prob, engine = _build_problem(_PHASE_ENGINE_MAP)

        for phase, (csv, _) in _PHASE_ENGINE_MAP.items():
            phase_engine = engine._phase_engines[phase]
            path = (
                f'traj.phases.{phase}.rhs_all.solver_sub.propulsion.'
                f'{phase_engine.name}'
            )
            subsys = prob.model._get_subsystem(path)
            self.assertIsNotNone(
                subsys,
                f'Expected per-phase engine subsystem at {path}',
            )
            self.assertIsInstance(subsys, om.Group)
            # Per-phase engine group must contain an EngineDeck interpolator,
            # confirming the engine was actually constructed from a CSV.
            self.assertIsNotNone(
                subsys._get_subsystem('interpolation'),
                f'Phase {phase!r} engine should contain an interpolation component '
                f'built from CSV {csv}',
            )

            # The builder holds the engine that was dispatched for this phase;
            # its DATA_FILE must match the CSV requested in phase_engine_map.
            self.assertEqual(
                str(phase_engine.get_val(Aircraft.Engine.DATA_FILE)),
                str(get_path(csv)),
                f'Phase {phase!r} engine should use CSV {csv}',
            )

    @require_pyoptsparse(optimizer='IPOPT')
    def test_optimization_converges_and_volumes_use_per_phase_density(self):
        """Run the optimization and verify volume = mass / density per entry.

        ``_PHASE_ENGINE_MAP`` maps each phase to a distinct CSV; with three
        distinct CSVs the output arrays have three entries, one per
        (csv, density) pair in insertion order.
        """
        prob, _ = _build_problem(_PHASE_ENGINE_MAP)
        prob.run_aviary_problem(suppress_solver_print=True)

        self.assertTrue(prob.result.success)

        total_fuel = prob.get_val(Mission.TOTAL_FUEL, units='lbm')[0]
        self.assertGreater(total_fuel, 10000.0)
        self.assertLess(total_fuel, 100000.0)

        fuel_mass = prob.get_val(TOTAL_MULTI_FUEL_MASS, units='lbm')
        fuel_volume = prob.get_val(TOTAL_MULTI_FUEL_VOLUME, units='galUS')
        densities = [d for _, d in _PHASE_ENGINE_MAP.values()]
        self.assertEqual(fuel_mass.shape, (len(densities),))
        self.assertEqual(fuel_volume.shape, (len(densities),))

        for mass, volume, density in zip(fuel_mass, fuel_volume, densities):
            self.assertGreater(mass, 0.0)
            assert_near_equal(volume, mass / density, tolerance=1e-6)


if __name__ == '__main__':
    unittest.main()
