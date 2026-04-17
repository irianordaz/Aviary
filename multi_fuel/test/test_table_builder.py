"""Tests for multi_fuel table builder module."""

import unittest
from copy import deepcopy
from unittest.mock import MagicMock, patch

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from multi_fuel.table_builder import (
    EngineTableBuilder,
    MultiEngineFuelBurnComp,
    MultiEngineTableBuilder,
)
from aviary.variable_info.variables import Mission

_ENGINE_CSV = "models/engines/turbofan_28k.csv"
_ENGINE_CSV_2 = "models/engines/turbofan_22k.csv"


class TestMultiEngineFuelBurnComp(unittest.TestCase):
    """Tests for MultiEngineFuelBurnComp OpenMDAO component."""

    def setUp(self):
        self.phase_engine_map = {
            "climb": (_ENGINE_CSV, 6.7),
            "cruise": (_ENGINE_CSV_2, 6.4),
            "descent": (_ENGINE_CSV_2, 3.5),
        }

    def test_component_creation(self):
        """Test that the component can be created with a phase_engine_map."""
        comp = MultiEngineFuelBurnComp(phase_engine_map=self.phase_engine_map)
        self.assertEqual(comp.options["phase_engine_map"], self.phase_engine_map)

    def test_component_setup(self):
        """Test that the component sets up correct inputs and outputs."""
        comp = MultiEngineFuelBurnComp(phase_engine_map=self.phase_engine_map)
        prob = om.Problem()
        prob.model.add_subsystem("fuel_burn", comp, promotes_inputs=["*"], promotes_outputs=["*"])
        prob.setup()

        expected_outputs = [
            "fuel_burn.mission:total_fuel_multi",
            "fuel_burn.mission:total_fuel_volume_multi",
        ]
        for output in expected_outputs:
            self.assertIn(output, prob.model._var_allprocs_abs2meta["output"])

    def test_compute_single_phase(self):
        """Test fuel burn computation with a single phase."""
        single_map = {"climb": (_ENGINE_CSV, 6.7)}
        comp = MultiEngineFuelBurnComp(phase_engine_map=single_map)
        prob = om.Problem()
        prob.model.add_subsystem("fuel_burn", comp, promotes_inputs=["*"], promotes_outputs=["*"])
        prob.setup()
        prob.set_val("mass_start_climb", 100000.0)
        prob.set_val("mass_end_climb", 90000.0)
        prob.run_model()

        expected_fuel = 10000.0
        assert_near_equal(prob.get_val(Mission.TOTAL_FUEL_MULTI)[0], expected_fuel, tolerance=1e-6)

    def test_compute_multiple_phases_same_engine(self):
        """Test fuel burn with multiple phases using the same engine."""
        same_engine_map = {
            "climb": (_ENGINE_CSV, 6.7),
            "cruise": (_ENGINE_CSV, 6.7),
        }
        comp = MultiEngineFuelBurnComp(phase_engine_map=same_engine_map)
        prob = om.Problem()
        prob.model.add_subsystem("fuel_burn", comp, promotes_inputs=["*"], promotes_outputs=["*"])
        prob.setup()
        prob.set_val("mass_start_climb", 100000.0)
        prob.set_val("mass_end_climb", 95000.0)
        prob.set_val("mass_start_cruise", 95000.0)
        prob.set_val("mass_end_cruise", 90000.0)
        prob.run_model()

        expected_fuel = 5000.0 + 5000.0
        assert_near_equal(prob.get_val(Mission.TOTAL_FUEL_MULTI)[0], expected_fuel, tolerance=1e-6)

    def test_compute_multiple_phases_different_engines(self):
        """Test fuel burn with multiple phases using different engines."""
        comp = MultiEngineFuelBurnComp(phase_engine_map=self.phase_engine_map)
        prob = om.Problem()
        prob.model.add_subsystem("fuel_burn", comp, promotes_inputs=["*"], promotes_outputs=["*"])
        prob.setup()
        prob.set_val("mass_start_climb", 100000.0)
        prob.set_val("mass_end_climb", 95000.0)
        prob.set_val("mass_start_cruise", 95000.0)
        prob.set_val("mass_end_cruise", 91000.0)
        prob.set_val("mass_start_descent", 91000.0)
        prob.set_val("mass_end_descent", 88000.0)
        prob.run_model()

        fuel_climb = 5000.0
        fuel_cruise = 4000.0
        fuel_descent = 3000.0

        assert_near_equal(prob.get_val(Mission.TOTAL_FUEL_MULTI)[0], fuel_climb, tolerance=1e-6)
        assert_near_equal(prob.get_val(Mission.TOTAL_FUEL_MULTI)[1], fuel_cruise, tolerance=1e-6)
        assert_near_equal(prob.get_val(Mission.TOTAL_FUEL_MULTI)[2], fuel_descent, tolerance=1e-6)

    def test_compute_volume_output(self):
        """Test that volume output is correctly computed from mass and density."""
        comp = MultiEngineFuelBurnComp(phase_engine_map=self.phase_engine_map)
        prob = om.Problem()
        prob.model.add_subsystem("fuel_burn", comp, promotes_inputs=["*"], promotes_outputs=["*"])
        prob.setup()
        prob.set_val("mass_start_climb", 100000.0)
        prob.set_val("mass_end_climb", 95000.0)
        prob.set_val("mass_start_cruise", 95000.0)
        prob.set_val("mass_end_cruise", 91000.0)
        prob.set_val("mass_start_descent", 91000.0)
        prob.set_val("mass_end_descent", 88000.0)
        prob.run_model()

        volume_climb = 5000.0 / 6.7
        volume_cruise = 4000.0 / 6.4
        volume_descent = 3000.0 / 3.5

        assert_near_equal(prob.get_val(Mission.TOTAL_FUEL_VOLUME_MULTI)[0], volume_climb, tolerance=1e-4)
        assert_near_equal(prob.get_val(Mission.TOTAL_FUEL_VOLUME_MULTI)[1], volume_cruise, tolerance=1e-4)
        assert_near_equal(prob.get_val(Mission.TOTAL_FUEL_VOLUME_MULTI)[2], volume_descent, tolerance=1e-4)

    def test_compute_with_zero_fuel(self):
        """Test fuel burn computation when mass_start equals mass_end."""
        comp = MultiEngineFuelBurnComp(phase_engine_map=self.phase_engine_map)
        prob = om.Problem()
        prob.model.add_subsystem("fuel_burn", comp, promotes_inputs=["*"], promotes_outputs=["*"])
        prob.setup()
        prob.set_val("mass_start_climb", 100000.0)
        prob.set_val("mass_end_climb", 100000.0)
        prob.set_val("mass_start_cruise", 95000.0)
        prob.set_val("mass_end_cruise", 95000.0)
        prob.set_val("mass_start_descent", 90000.0)
        prob.set_val("mass_end_descent", 90000.0)
        prob.run_model()

        assert_near_equal(prob.get_val(Mission.TOTAL_FUEL_MULTI)[0], 0.0, tolerance=1e-6)
        assert_near_equal(prob.get_val(Mission.TOTAL_FUEL_MULTI)[1], 0.0, tolerance=1e-6)
        assert_near_equal(prob.get_val(Mission.TOTAL_FUEL_MULTI)[2], 0.0, tolerance=1e-6)

    def test_compute_negative_fuel(self):
        """Test fuel burn computation when mass increases (negative fuel burn)."""
        comp = MultiEngineFuelBurnComp(phase_engine_map=self.phase_engine_map)
        prob = om.Problem()
        prob.model.add_subsystem("fuel_burn", comp, promotes_inputs=["*"], promotes_outputs=["*"])
        prob.setup()
        prob.set_val("mass_start_climb", 95000.0)
        prob.set_val("mass_end_climb", 100000.0)
        prob.run_model()

        expected_fuel = -5000.0
        assert_near_equal(prob.get_val(Mission.TOTAL_FUEL_MULTI)[0], expected_fuel, tolerance=1e-6)

    def test_partials_compute(self):
        """Test that partial derivatives are computed correctly."""
        comp = MultiEngineFuelBurnComp(phase_engine_map=self.phase_engine_map)
        prob = om.Problem()
        prob.model.add_subsystem("fuel_burn", comp, promotes_inputs=["*"], promotes_outputs=["*"])
        prob.setup()
        prob.set_val("mass_start_climb", 100000.0)
        prob.set_val("mass_end_climb", 95000.0)
        prob.set_val("mass_start_cruise", 95000.0)
        prob.set_val("mass_end_cruise", 91000.0)
        prob.set_val("mass_start_descent", 91000.0)
        prob.set_val("mass_end_descent", 88000.0)
        prob.run_model()

        rel_error = prob.check_partials(method="fd")
        for comp_name, comp_data in rel_error.items():
            for key, data in comp_data.items():
                if "forward" in data:
                    assert_near_equal(data["forward"]["abs error"][0][0], 0.0, tolerance=1e-4)
                if "cs" in data:
                    assert_near_equal(data["cs"]["abs error"][0][0], 0.0, tolerance=1e-6)


class TestEngineTableBuilder(unittest.TestCase):
    """Tests for EngineTableBuilder class."""

    def test_engine_table_builder_default_values(self):
        """Test EngineTableBuilder with default values."""
        builder = EngineTableBuilder()
        self.assertEqual(builder.name, "engine_table")

    def test_engine_table_builder_custom_name(self):
        """Test EngineTableBuilder with custom name."""
        builder = EngineTableBuilder(name="custom_engine")
        self.assertEqual(builder.name, "custom_engine")

    def test_engine_table_builder_custom_path(self):
        """Test EngineTableBuilder with custom CSV path."""
        builder = EngineTableBuilder(csv_path=_ENGINE_CSV)
        self.assertEqual(builder.name, "engine_table")


class TestMultiEngineTableBuilder(unittest.TestCase):
    """Tests for MultiEngineTableBuilder class."""

    def test_builder_creation_empty(self):
        """Test MultiEngineTableBuilder with empty phase_engine_map."""
        builder = MultiEngineTableBuilder()
        self.assertEqual(builder.phase_engine_map, {})
        self.assertEqual(builder._phase_engines, {})

    def test_builder_creation_with_phases(self):
        """Test MultiEngineTableBuilder with phase_engine_map."""
        phase_map = {
            "climb": (_ENGINE_CSV, 6.7),
            "cruise": (_ENGINE_CSV_2, 6.4),
        }
        builder = MultiEngineTableBuilder(phase_engine_map=phase_map)
        self.assertEqual(len(builder.phase_engine_map), 2)
        self.assertEqual(builder.phase_engine_map["climb"], (_ENGINE_CSV, 6.7))
        self.assertEqual(builder.phase_engine_map["cruise"], (_ENGINE_CSV_2, 6.4))

    def test_builder_string_values_converted(self):
        """Test that string values in phase_engine_map are converted to tuples."""
        phase_map = {
            "climb": _ENGINE_CSV,
            "cruise": _ENGINE_CSV_2,
        }
        builder = MultiEngineTableBuilder(phase_engine_map=phase_map)
        self.assertIsInstance(builder.phase_engine_map["climb"], tuple)
        self.assertEqual(builder.phase_engine_map["climb"][0], _ENGINE_CSV)
        self.assertEqual(builder.phase_engine_map["cruise"][0], _ENGINE_CSV_2)

    def test_builder_default_density_for_strings(self):
        """Test that string values use the default fuel density."""
        phase_map = {"climb": _ENGINE_CSV}
        builder = MultiEngineTableBuilder(phase_engine_map=phase_map)
        from multi_fuel.table_builder import _DEFAULT_FUEL_DENSITY_LBM_GAL
        self.assertEqual(builder.phase_engine_map["climb"][1], _DEFAULT_FUEL_DENSITY_LBM_GAL)

    def test_builder_phase_engines_created(self):
        """Test that _phase_engines is populated correctly."""
        phase_map = {
            "climb": (_ENGINE_CSV, 6.7),
            "cruise": (_ENGINE_CSV_2, 6.4),
        }
        builder = MultiEngineTableBuilder(phase_engine_map=phase_map)
        self.assertEqual(len(builder._phase_engines), 2)
        self.assertIn("climb", builder._phase_engines)
        self.assertIn("cruise", builder._phase_engines)

    def test_builder_name_default(self):
        """Test that the builder has the default name."""
        builder = MultiEngineTableBuilder()
        self.assertEqual(builder.name, "multi_engine_table")

    def test_builder_name_custom(self):
        """Test that the builder accepts a custom name."""
        builder = MultiEngineTableBuilder(name="custom_multi_engine")
        self.assertEqual(builder.name, "custom_multi_engine")

    def test_builder_compute_max_values(self):
        """Test that compute_max_values is set to True."""
        builder = MultiEngineTableBuilder()
        self.assertTrue(builder.compute_max_values)

    @use_tempdirs
    def test_configure_phase_info(self):
        """Test that configure_phase_info injects phase names correctly."""
        phase_map = {
            "climb": _ENGINE_CSV,
        }
        builder = MultiEngineTableBuilder(phase_engine_map=phase_map)
        phase_info = {
            "climb": {"subsystem_options": {}},
            "cruise": {"subsystem_options": {}},
            "descent": {"subsystem_options": {}},
            "pre_mission": {"subsystem_options": {}},
            "post_mission": {"subsystem_options": {}},
        }
        configured = builder.configure_phase_info(deepcopy(phase_info))

        self.assertIn("phase_name", configured["climb"]["subsystem_options"]["propulsion"]["multi_engine_table"])
        self.assertEqual(configured["climb"]["subsystem_options"]["propulsion"]["multi_engine_table"]["phase_name"], "climb")
        pre_mission_opts = configured["pre_mission"]["subsystem_options"].get("propulsion", {}).get("multi_engine_table", {})
        post_mission_opts = configured["post_mission"]["subsystem_options"].get("propulsion", {}).get("multi_engine_table", {})
        self.assertNotIn("phase_name", pre_mission_opts)
        self.assertNotIn("phase_name", post_mission_opts)

    @use_tempdirs
    def test_configure_phase_info_custom_propulsion_name(self):
        """Test configure_phase_info with custom propulsion name."""
        phase_map = {"climb": _ENGINE_CSV}
        builder = MultiEngineTableBuilder(phase_engine_map=phase_map, name="custom_engine")
        phase_info = {
            "climb": {"subsystem_options": {}},
        }
        configured = builder.configure_phase_info(deepcopy(phase_info), propulsion_name="custom_prop")
        phase_name_value = configured["climb"]["subsystem_options"]["custom_prop"]["custom_engine"]["phase_name"]
        self.assertEqual(phase_name_value, "climb")

    def test_get_post_mission_promotes_outputs(self):
        """Test that get_post_mission_promotes_outputs returns the correct list."""
        builder = MultiEngineTableBuilder()
        outputs = builder.get_post_mission_promotes_outputs()
        self.assertEqual(outputs, [Mission.TOTAL_FUEL_MULTI])

    def test_build_post_mission_empty(self):
        """Test that build_post_mission returns None for empty phase_engine_map."""
        builder = MultiEngineTableBuilder()
        result = builder.build_post_mission()
        self.assertIsNone(result)

    def test_build_post_mission_nonempty(self):
        """Test that build_post_mission returns a component for non-empty phase_engine_map."""
        phase_map = {"climb": _ENGINE_CSV}
        builder = MultiEngineTableBuilder(phase_engine_map=phase_map)
        result = builder.build_post_mission()
        self.assertIsInstance(result, MultiEngineFuelBurnComp)

    def test_get_traj_connections_empty(self):
        """Test that get_traj_connections returns empty list for no phases."""
        builder = MultiEngineTableBuilder()
        connections = builder.get_traj_connections(["climb", "cruise"])
        self.assertEqual(connections, [])

    def test_get_traj_connections_single_phase(self):
        """Test that get_traj_connections returns correct connections for single phase."""
        phase_map = {"climb": _ENGINE_CSV}
        builder = MultiEngineTableBuilder(phase_engine_map=phase_map)
        connections = builder.get_traj_connections(["climb", "cruise"])
        self.assertEqual(len(connections), 2)

        self.assertEqual(connections[0][0], "traj.climb.timeseries.mass")
        self.assertEqual(connections[0][1], "multi_engine_table.mass_start_climb")
        self.assertEqual(connections[0][2], [0])

        self.assertEqual(connections[1][0], "traj.climb.timeseries.mass")
        self.assertEqual(connections[1][1], "multi_engine_table.mass_end_climb")
        self.assertEqual(connections[1][2], [-1])

    def test_get_traj_connections_multiple_phases(self):
        """Test that get_traj_connections returns correct connections for multiple phases."""
        phase_map = {"climb": _ENGINE_CSV, "cruise": _ENGINE_CSV_2}
        builder = MultiEngineTableBuilder(phase_engine_map=phase_map)
        connections = builder.get_traj_connections(["climb", "cruise", "descent"])
        self.assertEqual(len(connections), 4)

    def test_get_traj_connections_skips_missing_phases(self):
        """Test that get_traj_connections skips phases not in regular_phases."""
        phase_map = {"climb": _ENGINE_CSV, "cruise": _ENGINE_CSV_2}
        builder = MultiEngineTableBuilder(phase_engine_map=phase_map)
        connections = builder.get_traj_connections(["descent"])
        self.assertEqual(connections, [])

    def test_default_engine_raises_on_empty(self):
        """Test that _default_engine raises ValueError when no engines configured."""
        builder = MultiEngineTableBuilder()
        with self.assertRaises(ValueError):
            builder._default_engine()

    def test_default_engine_returns_first_engine(self):
        """Test that _default_engine returns the first configured engine."""
        phase_map = {
            "climb": (_ENGINE_CSV, 6.7),
            "cruise": (_ENGINE_CSV_2, 6.4),
        }
        builder = MultiEngineTableBuilder(phase_engine_map=phase_map)
        default = builder._default_engine()
        self.assertEqual(default.name, "multi_engine_table_climb")

    def test_build_mission_phase_not_found(self):
        """Test that build_mission raises KeyError for unconfigured phase."""
        phase_map = {"climb": _ENGINE_CSV}
        builder = MultiEngineTableBuilder(phase_engine_map=phase_map)
        with self.assertRaises(KeyError):
            builder.build_mission(
                num_nodes=10,
                aviary_inputs=None,
                user_options={},
                subsystem_options={"phase_name": "cruise"},
            )


class TestMultiEngineFuelBurnCompIntegration(unittest.TestCase):
    """Integration tests for MultiEngineFuelBurnComp."""

    def test_full_workflow_with_problem(self):
        """Test the full workflow using an OpenMDAO Problem."""
        phase_engine_map = {
            "climb": (_ENGINE_CSV, 6.7),
            "cruise": (_ENGINE_CSV_2, 6.4),
        }
        comp = MultiEngineFuelBurnComp(phase_engine_map=phase_engine_map)
        prob = om.Problem()
        model = prob.model.add_subsystem("fuel_burn", comp, promotes_inputs=["*"], promotes_outputs=["*"])
        prob.setup()

        prob.set_val("mass_start_climb", 100000.0)
        prob.set_val("mass_end_climb", 95000.0)
        prob.set_val("mass_start_cruise", 95000.0)
        prob.set_val("mass_end_cruise", 90000.0)
        prob.run_model()

        assert_near_equal(prob.get_val(Mission.TOTAL_FUEL_MULTI)[0], 5000.0, tolerance=1e-6)
        assert_near_equal(prob.get_val(Mission.TOTAL_FUEL_MULTI)[1], 5000.0, tolerance=1e-6)

    def test_output_shapes(self):
        """Test that output shapes are correct."""
        phase_engine_map = {
            "climb": (_ENGINE_CSV, 6.7),
            "cruise": (_ENGINE_CSV_2, 6.4),
            "descent": (_ENGINE_CSV, 3.5),
        }
        comp = MultiEngineFuelBurnComp(phase_engine_map=phase_engine_map)
        prob = om.Problem()
        prob.model.add_subsystem("fuel_burn", comp, promotes_inputs=["*"], promotes_outputs=["*"])
        prob.setup()
        prob.run_model()

        fuel_shape = prob.get_val(Mission.TOTAL_FUEL_MULTI).shape
        volume_shape = prob.get_val(Mission.TOTAL_FUEL_VOLUME_MULTI).shape
        self.assertEqual(fuel_shape, (3,))
        self.assertEqual(volume_shape, (3,))


if __name__ == "__main__":
    unittest.main()
