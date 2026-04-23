"""Tests for the multi_fuel table builder module.

These tests verify the two core requirements of ``MultiEngineTableBuilder``:

1. Each engine CSV provided in ``phase_engine_map`` is assigned to and used in
   the requested mission phase (via ``MultiPhasePropulsionBuilder`` dispatch).
2. The fuel density provided per phase is used when computing the fuel volume.
"""

import unittest
from copy import deepcopy
from unittest import mock

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from aviary.subsystems.propulsion.propulsion_builder import CorePropulsionBuilder
from aviary.utils.functions import get_path
from aviary.variable_info.variables import Aircraft
from multi_fuel.table_builder import (
    _DEFAULT_FUEL_DENSITY_LBM_GAL,
    TOTAL_MULTI_FUEL_MASS,
    TOTAL_MULTI_FUEL_VOLUME,
    EngineTableBuilder,
    MultiEngineFuelBurnComp,
    MultiEngineTableBuilder,
    MultiPhasePropulsionBuilder,
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
        fuel_burn_comp = MultiEngineFuelBurnComp(phase_engine_map=phase_engine_map)
        prob = om.Problem()
        prob.model.add_subsystem('fuel_burn', fuel_burn_comp, promotes=['*'])
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

        mass = prob.get_val(TOTAL_MULTI_FUEL_MASS, units='lbm')
        volume = prob.get_val(TOTAL_MULTI_FUEL_VOLUME, units='galUS')

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
        fuel_burn_comp = MultiEngineFuelBurnComp(phase_engine_map=phase_engine_map)
        prob = om.Problem()
        prob.model.add_subsystem('fuel_burn', fuel_burn_comp, promotes=['*'])
        prob.setup()
        _set_phase_masses(
            prob,
            {'climb': (100000.0, 95000.0), 'cruise': (95000.0, 91000.0)},
        )
        prob.run_model()

        mass = prob.get_val(TOTAL_MULTI_FUEL_MASS, units='lbm')
        volume = prob.get_val(TOTAL_MULTI_FUEL_VOLUME, units='galUS')

        self.assertEqual(mass.shape, (2,))
        assert_near_equal(mass, [5000.0, 4000.0], tolerance=1e-8)
        assert_near_equal(volume, [5000.0 / 6.7, 4000.0 / 6.4], tolerance=1e-8)

    def test_same_csv_same_density_aggregates(self):
        phase_engine_map = {
            'climb': (_CSV_22K, 6.4),
            'cruise': (_CSV_22K, 6.4),
        }
        fuel_burn_comp = MultiEngineFuelBurnComp(phase_engine_map=phase_engine_map)
        prob = om.Problem()
        prob.model.add_subsystem('fuel_burn', fuel_burn_comp, promotes=['*'])
        prob.setup()
        _set_phase_masses(
            prob,
            {'climb': (100000.0, 95000.0), 'cruise': (95000.0, 91000.0)},
        )
        prob.run_model()

        mass = prob.get_val(TOTAL_MULTI_FUEL_MASS, units='lbm')
        volume = prob.get_val(TOTAL_MULTI_FUEL_VOLUME, units='galUS')

        self.assertEqual(mass.shape, (1,))
        assert_near_equal(mass, [9000.0], tolerance=1e-8)
        assert_near_equal(volume, [9000.0 / 6.4], tolerance=1e-8)

    def test_partials(self):
        phase_engine_map = {
            'climb': (_CSV_28K, 6.7),
            'cruise': (_CSV_22K, 6.4),
        }
        fuel_burn_comp = MultiEngineFuelBurnComp(phase_engine_map=phase_engine_map)
        prob = om.Problem()
        prob.model.add_subsystem('fuel_burn', fuel_burn_comp, promotes=['*'])
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
        builder = EngineTableBuilder(engine_csv=_CSV_28K)
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
        for phase, fuel_density in expected.items():
            engine = builder._phase_engines[phase]
            self.assertAlmostEqual(
                engine.get_val(Aircraft.Fuel.DENSITY, units='lbm/galUS'),
                fuel_density,
                places=9,
                msg=f'phase {phase!r} should have fuel_density {fuel_density}',
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
        engine_names = {engine.name for engine in builder._phase_engines.values()}
        self.assertEqual(
            engine_names,
            {
                'multi_engine_table_climb',
                'multi_engine_table_cruise',
                'multi_engine_table_descent',
            },
        )

    def test_default_builder_name(self):
        self.assertEqual(MultiEngineTableBuilder().name, 'multi_engine_table')

    def test_custom_builder_name(self):
        self.assertEqual(MultiEngineTableBuilder(name='x').name, 'x')


class TestConfigurePhaseInfo(unittest.TestCase):
    """Verify ``configure_phase_info`` tags each phase with its name under propulsion."""

    def test_injects_phase_name_into_propulsion_subsystem_options(self):
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
            configured['climb']['subsystem_options']['propulsion']['phase_name'],
            'climb',
        )
        self.assertEqual(
            configured['cruise']['subsystem_options']['propulsion']['phase_name'],
            'cruise',
        )
        # Existing subsystem_options entries are preserved.
        self.assertEqual(
            configured['cruise']['subsystem_options']['other'], {'k': 1}
        )

    def test_custom_propulsion_name(self):
        builder = MultiEngineTableBuilder(phase_engine_map={'climb': _CSV_28K})
        phase_info = {'climb': {}}
        configured = builder.configure_phase_info(
            deepcopy(phase_info), propulsion_name='custom_prop'
        )
        self.assertEqual(
            configured['climb']['subsystem_options']['custom_prop'][
                'phase_name'
            ],
            'climb',
        )

    def test_skips_pre_and_post_mission(self):
        builder = MultiEngineTableBuilder(phase_engine_map={'climb': _CSV_28K})
        phase_info = {'climb': {}, 'pre_mission': {}, 'post_mission': {}}
        configured = builder.configure_phase_info(deepcopy(phase_info))

        self.assertNotIn('subsystem_options', configured['pre_mission'])
        self.assertNotIn('subsystem_options', configured['post_mission'])


class _StubAviaryGroup:
    """Minimal stub with a ``subsystems`` list in the shape AviaryGroup provides."""

    def __init__(self, subsystems):
        self.subsystems = list(subsystems)


class TestMultiPhasePropulsionBuilderDispatch(unittest.TestCase):
    """Verify ``MultiPhasePropulsionBuilder.build_mission`` selects per phase.

    The builder wraps a dict ``{phase_name: engine}``. On each
    ``build_mission`` call, it should pick the engine whose key matches
    ``subsystem_options['phase_name']`` and pass *only* that engine to the
    ``PropulsionMission`` it constructs.
    """

    def _make_propulsion_builder(self):
        outer_builder = MultiEngineTableBuilder(
            phase_engine_map={
                'climb': (_CSV_28K, 6.7),
                'cruise': (_CSV_22K, 6.4),
                'descent': (_CSV_24K, 4.2),
            },
        )
        propulsion_builder = MultiPhasePropulsionBuilder(
            name='propulsion',
            phase_engines=outer_builder._phase_engines,
            default_engine=next(iter(outer_builder._phase_engines.values())),
        )
        return propulsion_builder, outer_builder

    def test_picks_engine_registered_for_phase(self):
        propulsion_builder, outer_builder = self._make_propulsion_builder()
        with mock.patch(
            'multi_fuel.table_builder.PropulsionMission'
        ) as mock_propulsion_mission:
            for phase in ('climb', 'cruise', 'descent'):
                propulsion_builder.build_mission(
                    num_nodes=3,
                    aviary_inputs=None,
                    user_options={},
                    subsystem_options={'phase_name': phase},
                )
                call_kwargs = mock_propulsion_mission.call_args.kwargs
                self.assertEqual(
                    call_kwargs['engine_models'],
                    [outer_builder._phase_engines[phase]],
                    f'phase {phase!r} should dispatch to its engine',
                )

    def test_unknown_phase_falls_back_to_default(self):
        propulsion_builder, outer_builder = self._make_propulsion_builder()
        with mock.patch(
            'multi_fuel.table_builder.PropulsionMission'
        ) as mock_propulsion_mission:
            propulsion_builder.build_mission(
                num_nodes=3,
                aviary_inputs=None,
                user_options={},
                subsystem_options={'phase_name': 'not_a_phase'},
            )
            self.assertEqual(
                mock_propulsion_mission.call_args.kwargs['engine_models'],
                [next(iter(outer_builder._phase_engines.values()))],
            )

    def test_missing_subsystem_options_falls_back_to_default(self):
        propulsion_builder, outer_builder = self._make_propulsion_builder()
        with mock.patch(
            'multi_fuel.table_builder.PropulsionMission'
        ) as mock_propulsion_mission:
            propulsion_builder.build_mission(
                num_nodes=3,
                aviary_inputs=None,
                user_options={},
                subsystem_options=None,
            )
            self.assertEqual(
                mock_propulsion_mission.call_args.kwargs['engine_models'],
                [next(iter(outer_builder._phase_engines.values()))],
            )


class TestInstallPropulsion(unittest.TestCase):
    """Verify ``install_propulsion`` swaps ``CorePropulsionBuilder`` in place."""

    def test_replaces_default_propulsion_builder(self):
        builder = MultiEngineTableBuilder(
            phase_engine_map={'climb': _CSV_28K, 'cruise': _CSV_22K},
        )
        original_propulsion = CorePropulsionBuilder(
            'propulsion', engine_models=[next(iter(builder._phase_engines.values()))]
        )
        aviary_group = _StubAviaryGroup(
            [original_propulsion, object(), object()]
        )

        builder.install_propulsion(aviary_group)

        self.assertIsInstance(
            aviary_group.subsystems[0], MultiPhasePropulsionBuilder
        )
        self.assertEqual(aviary_group.subsystems[0].name, 'propulsion')
        # The replacement carries the same per-phase engine mapping.
        self.assertEqual(
            aviary_group.subsystems[0]._phase_engines, builder._phase_engines
        )

    def test_raises_when_no_propulsion_subsystem(self):
        builder = MultiEngineTableBuilder(phase_engine_map={'climb': _CSV_28K})
        aviary_group = _StubAviaryGroup([object(), object()])

        with self.assertRaises(RuntimeError):
            builder.install_propulsion(aviary_group)


class TestBuildPostMission(unittest.TestCase):
    def test_returns_none_when_empty(self):
        self.assertIsNone(MultiEngineTableBuilder().build_post_mission())

    def test_returns_fuel_burn_comp(self):
        builder = MultiEngineTableBuilder(
            phase_engine_map={'climb': (_CSV_28K, 6.7)},
        )
        fuel_burn_component = builder.build_post_mission()
        self.assertIsInstance(fuel_burn_component, MultiEngineFuelBurnComp)
        self.assertEqual(
            fuel_burn_component.options['phase_engine_map'],
            builder.phase_engine_map,
        )


if __name__ == '__main__':
    unittest.main()
