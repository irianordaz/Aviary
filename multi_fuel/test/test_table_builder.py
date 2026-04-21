"""Tests for the multi_fuel table builder module.

These tests verify the two core requirements of ``MultiEngineTableBuilder``:

1. Each engine CSV provided in ``phase_engine_map`` is assigned to and used in
   the requested mission phase (via ``build_mission`` dispatch).
2. The fuel density provided per phase is used when computing the fuel volume.
"""

import unittest
from copy import deepcopy
from unittest import mock

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from aviary.utils.functions import get_path
from aviary.variable_info.variables import Aircraft
from multi_fuel.table_builder import (
    _DEFAULT_FUEL_DENSITY_LBM_GAL,
    TOTAL_FUEL_MULTI,
    TOTAL_FUEL_VOLUME_MULTI,
    EngineTableBuilder,
    MultiEngineFuelBurnComp,
    MultiEngineTableBuilder,
)

_CSV_28K = 'multi_fuel/engines/turbofan_28k.csv'
_CSV_22K = 'multi_fuel/engines/turbofan_22k.csv'
_CSV_24K = 'multi_fuel/engines/turbofan_24k_1.csv'


def _set_phase_masses(prob, masses):
    """Set mass_start_<phase> / mass_end_<phase> on a fuel-burn problem."""
    for phase, (start, end) in masses.items():
        prob.set_val(f'mass_start_{phase}', start)
        prob.set_val(f'mass_end_{phase}', end)


class TestMultiEngineFuelBurnComp(unittest.TestCase):
    """Verify per-phase fuel density is used in the volume calculation."""

    def test_volume_uses_per_phase_density(self):
        # Three unique (csv, density) entries → each volume uses its own density.
        phase_engine_map = {
            'climb': (_CSV_28K, 6.7),
            'cruise': (_CSV_22K, 6.4),
            'descent': (_CSV_24K, 4.2),
        }
        comp = MultiEngineFuelBurnComp(phase_engine_map=phase_engine_map)
        prob = om.Problem()
        prob.model.add_subsystem('fuel_burn', comp, promotes=['*'])
        prob.setup()
        _set_phase_masses(
            prob,
            {
                'climb': (100000.0, 95000.0),
                'cruise': (95000.0, 91000.0),
                'descent': (91000.0, 88000.0),
            },
        )
        prob.run_model()

        mass = prob.get_val(TOTAL_FUEL_MULTI, units='lbm')
        volume = prob.get_val(TOTAL_FUEL_VOLUME_MULTI, units='galUS')

        assert_near_equal(mass, [5000.0, 4000.0, 3000.0], tolerance=1e-8)
        assert_near_equal(
            volume,
            [5000.0 / 6.7, 4000.0 / 6.4, 3000.0 / 4.2],
            tolerance=1e-8,
        )

    def test_same_csv_different_density_creates_separate_entries(self):
        phase_engine_map = {
            'climb': (_CSV_22K, 6.7),
            'cruise': (_CSV_22K, 6.4),
        }
        comp = MultiEngineFuelBurnComp(phase_engine_map=phase_engine_map)
        prob = om.Problem()
        prob.model.add_subsystem('fuel_burn', comp, promotes=['*'])
        prob.setup()
        _set_phase_masses(
            prob,
            {'climb': (100000.0, 95000.0), 'cruise': (95000.0, 91000.0)},
        )
        prob.run_model()

        mass = prob.get_val(TOTAL_FUEL_MULTI, units='lbm')
        volume = prob.get_val(TOTAL_FUEL_VOLUME_MULTI, units='galUS')

        self.assertEqual(mass.shape, (2,))
        assert_near_equal(mass, [5000.0, 4000.0], tolerance=1e-8)
        assert_near_equal(volume, [5000.0 / 6.7, 4000.0 / 6.4], tolerance=1e-8)

    def test_same_csv_same_density_aggregates(self):
        phase_engine_map = {
            'climb': (_CSV_22K, 6.4),
            'cruise': (_CSV_22K, 6.4),
        }
        comp = MultiEngineFuelBurnComp(phase_engine_map=phase_engine_map)
        prob = om.Problem()
        prob.model.add_subsystem('fuel_burn', comp, promotes=['*'])
        prob.setup()
        _set_phase_masses(
            prob,
            {'climb': (100000.0, 95000.0), 'cruise': (95000.0, 91000.0)},
        )
        prob.run_model()

        mass = prob.get_val(TOTAL_FUEL_MULTI, units='lbm')
        volume = prob.get_val(TOTAL_FUEL_VOLUME_MULTI, units='galUS')

        self.assertEqual(mass.shape, (1,))
        assert_near_equal(mass, [9000.0], tolerance=1e-8)
        assert_near_equal(volume, [9000.0 / 6.4], tolerance=1e-8)

    def test_partials(self):
        phase_engine_map = {
            'climb': (_CSV_28K, 6.7),
            'cruise': (_CSV_22K, 6.4),
        }
        comp = MultiEngineFuelBurnComp(phase_engine_map=phase_engine_map)
        prob = om.Problem()
        prob.model.add_subsystem('fuel_burn', comp, promotes=['*'])
        prob.setup()
        _set_phase_masses(
            prob,
            {'climb': (100000.0, 95000.0), 'cruise': (95000.0, 91000.0)},
        )
        prob.run_model()

        data = prob.check_partials(
            method='fd', out_stream=None, compact_print=True
        )
        for comp_data in data.values():
            for direction_data in comp_data.values():
                if 'forward' in direction_data:
                    assert_near_equal(
                        direction_data['forward']['abs error'][0][0],
                        0.0,
                        tolerance=1e-5,
                    )


class TestEngineTableBuilder(unittest.TestCase):
    """Verify ``EngineTableBuilder`` resolves the CSV into DATA_FILE."""

    def test_default_name(self):
        self.assertEqual(EngineTableBuilder().name, 'engine_table')

    def test_custom_name(self):
        self.assertEqual(EngineTableBuilder(name='x').name, 'x')

    def test_data_file_option_set_from_csv_path(self):
        builder = EngineTableBuilder(csv_path=_CSV_28K)
        self.assertEqual(
            str(builder.get_val(Aircraft.Engine.DATA_FILE)),
            str(get_path(_CSV_28K)),
        )


class TestMultiEngineTableBuilder(unittest.TestCase):
    """Verify per-phase engines are created with the correct CSV and density."""

    def _make_builder(self):
        return MultiEngineTableBuilder(
            phase_engine_map={
                'climb': (_CSV_28K, 6.7),
                'cruise': (_CSV_22K, 6.4),
                'descent': (_CSV_24K, 4.2),
            },
        )

    def test_each_phase_engine_uses_the_requested_csv(self):
        """Each per-phase engine must resolve DATA_FILE to the mapped CSV."""
        builder = self._make_builder()
        expected = {
            'climb': _CSV_28K,
            'cruise': _CSV_22K,
            'descent': _CSV_24K,
        }
        for phase, csv in expected.items():
            engine = builder._phase_engines[phase]
            self.assertEqual(
                str(engine.get_val(Aircraft.Engine.DATA_FILE)),
                str(get_path(csv)),
                f'phase {phase!r} should use engine CSV {csv}',
            )

    def test_each_phase_engine_uses_the_requested_density(self):
        """Each per-phase engine must carry the requested Aircraft.Fuel.DENSITY."""
        builder = self._make_builder()
        expected = {'climb': 6.7, 'cruise': 6.4, 'descent': 4.2}
        for phase, density in expected.items():
            engine = builder._phase_engines[phase]
            self.assertAlmostEqual(
                engine.get_val(Aircraft.Fuel.DENSITY, units='lbm/galUS'),
                density,
                places=9,
                msg=f'phase {phase!r} should have density {density}',
            )

    def test_string_value_applies_default_density(self):
        builder = MultiEngineTableBuilder(phase_engine_map={'climb': _CSV_28K})
        self.assertAlmostEqual(
            builder._phase_engines['climb'].get_val(
                Aircraft.Fuel.DENSITY, units='lbm/galUS'
            ),
            _DEFAULT_FUEL_DENSITY_LBM_GAL,
            places=9,
        )

    def test_per_phase_engine_names_are_distinct(self):
        builder = self._make_builder()
        names = {e.name for e in builder._phase_engines.values()}
        self.assertEqual(
            names,
            {
                'multi_engine_table_climb',
                'multi_engine_table_cruise',
                'multi_engine_table_descent',
            },
        )

    def test_get_default_engine_returns_first_entry(self):
        builder = self._make_builder()
        default = builder.get_default_engine()
        self.assertEqual(
            str(default.get_val(Aircraft.Engine.DATA_FILE)),
            str(get_path(_CSV_28K)),
        )

    def test_get_default_engine_empty_raises(self):
        with self.assertRaises(ValueError):
            MultiEngineTableBuilder().get_default_engine()

    def test_default_builder_name(self):
        self.assertEqual(MultiEngineTableBuilder().name, 'multi_engine_table')

    def test_custom_builder_name(self):
        self.assertEqual(MultiEngineTableBuilder(name='x').name, 'x')


class TestConfigurePhaseInfo(unittest.TestCase):
    """Verify ``configure_phase_info`` tags each phase with its name."""

    def test_injects_phase_name_into_subsystem_options(self):
        builder = MultiEngineTableBuilder(
            phase_engine_map={'climb': _CSV_28K, 'cruise': _CSV_22K},
        )
        phase_info = {
            'climb': {},
            'cruise': {'subsystem_options': {'other': {'k': 1}}},
            'pre_mission': {},
            'post_mission': {},
        }
        configured = builder.configure_phase_info(deepcopy(phase_info))

        self.assertEqual(
            configured['climb']['subsystem_options'][builder.name][
                'phase_name'
            ],
            'climb',
        )
        self.assertEqual(
            configured['cruise']['subsystem_options'][builder.name][
                'phase_name'
            ],
            'cruise',
        )
        # Existing subsystem_options entries are preserved.
        self.assertEqual(
            configured['cruise']['subsystem_options']['other'], {'k': 1}
        )

    def test_skips_pre_and_post_mission(self):
        builder = MultiEngineTableBuilder(phase_engine_map={'climb': _CSV_28K})
        phase_info = {'climb': {}, 'pre_mission': {}, 'post_mission': {}}
        configured = builder.configure_phase_info(deepcopy(phase_info))

        self.assertNotIn('subsystem_options', configured['pre_mission'])
        self.assertNotIn('subsystem_options', configured['post_mission'])


class TestBuildMissionDispatch(unittest.TestCase):
    """Verify ``build_mission`` routes to the engine configured for that phase."""

    def test_dispatches_to_the_phase_engine(self):
        builder = MultiEngineTableBuilder(
            phase_engine_map={
                'climb': (_CSV_28K, 6.7),
                'cruise': (_CSV_22K, 6.4),
                'descent': (_CSV_24K, 4.2),
            },
        )

        # Replace each per-phase engine's build_mission with a distinct sentinel
        # group. build_mission on the builder should return the sentinel for the
        # engine registered at the requested phase.
        sentinels = {}
        for phase, engine in builder._phase_engines.items():
            sentinel = om.Group()
            sentinels[phase] = sentinel
            engine.build_mission = mock.MagicMock(return_value=sentinel)

        for phase in ('climb', 'cruise', 'descent'):
            result = builder.build_mission(
                num_nodes=3,
                aviary_inputs=None,
                user_options={},
                subsystem_options={'phase_name': phase},
            )
            self.assertIs(
                result,
                sentinels[phase],
                f'build_mission for phase {phase!r} should return that phase engine',
            )
            builder._phase_engines[phase].build_mission.assert_called_once()

        # And only the phase's engine should have been called (not cross-phase).
        for phase, engine in builder._phase_engines.items():
            self.assertEqual(engine.build_mission.call_count, 1)

    def test_returns_none_for_unknown_phase(self):
        builder = MultiEngineTableBuilder(phase_engine_map={'climb': _CSV_28K})
        self.assertIsNone(
            builder.build_mission(
                num_nodes=3,
                aviary_inputs=None,
                user_options={},
                subsystem_options={'phase_name': 'not_a_phase'},
            )
        )

    def test_returns_none_with_empty_subsystem_options(self):
        builder = MultiEngineTableBuilder(phase_engine_map={'climb': _CSV_28K})
        self.assertIsNone(
            builder.build_mission(
                num_nodes=3,
                aviary_inputs=None,
                user_options={},
                subsystem_options={},
            )
        )


class TestBuildPostMission(unittest.TestCase):
    def test_returns_none_when_empty(self):
        self.assertIsNone(MultiEngineTableBuilder().build_post_mission())

    def test_returns_fuel_burn_comp(self):
        builder = MultiEngineTableBuilder(
            phase_engine_map={'climb': (_CSV_28K, 6.7)},
        )
        comp = builder.build_post_mission()
        self.assertIsInstance(comp, MultiEngineFuelBurnComp)
        self.assertEqual(
            comp.options['phase_engine_map'], builder.phase_engine_map
        )


if __name__ == '__main__':
    unittest.main()
