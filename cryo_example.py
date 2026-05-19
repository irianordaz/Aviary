"""Aviary + HyTank integration example.

Demonstrates how to use HyTank's LH2 vacuum tank weight model
as an external subsystem in an Aviary advanced single aisle
problem. The tank weight computed by HyTank overrides
Aircraft.Fuel.FUEL_SYSTEM_MASS in Aviary's pre-mission mass
buildup.
"""

import openmdao.api as om

from aviary.api import Aircraft
from aviary.core.aviary_problem import AviaryProblem
from aviary.subsystems.subsystem_builder import SubsystemBuilder
from aviary.utils.functions import get_aviary_resource_path
from aviary.variable_info.functions import (
    add_aviary_output,
)

from aviary.models.aircraft.advanced_single_aisle.phase_info import (
    phase_info,
)

from hytank.weight import VacuumTankWeight

# ── Constants ──────────────────────────────────────────────────
KG_TO_LBM = 2.20462


# ── Pre-mission wrapper component ─────────────────────────────
class LH2TankWeightComp(om.Group):
    """Wraps HyTank VacuumTankWeight and outputs fuel system mass.

    The VacuumTankWeight component computes the structural
    weight of a vacuum-insulated LH2 tank in kilograms. This
    wrapper converts that weight to lbm and promotes it as
    Aircraft.Fuel.FUEL_SYSTEM_MASS so Aviary can consume it.
    """

    def setup(self):
        self.add_subsystem(
            'tank_weight',
            VacuumTankWeight(),
            promotes_inputs=[
                'environment_design_pressure',
                'max_expected_operating_pressure',
                'vacuum_gap',
                'radius',
                'length',
                'N_layers',
            ],
            promotes_outputs=[('weight', 'tank_weight_kg')],
        )

        self.add_subsystem(
            'unit_conversion',
            _TankWeightToFuelSystemMass(),
            promotes_inputs=['tank_weight_kg'],
            promotes_outputs=['*'],
        )

        # ── Sensible defaults for LH2 tank geometry ──
        self.set_input_defaults(
            'radius', 1.5, units='m',
        )
        self.set_input_defaults(
            'length', 6.0, units='m',
        )
        self.set_input_defaults(
            'N_layers', 30,
        )
        self.set_input_defaults(
            'vacuum_gap', 5, units='cm',
        )
        self.set_input_defaults(
            'environment_design_pressure', 1.0, units='bar',
        )
        self.set_input_defaults(
            'max_expected_operating_pressure',
            5.0,
            units='bar',
        )


class _TankWeightToFuelSystemMass(om.ExplicitComponent):
    """Convert tank weight (kg) to fuel system mass (lbm)."""

    def setup(self):
        self.add_input(
            'tank_weight_kg',
            val=0.0,
            units='kg',
            desc='LH2 tank structural weight from HyTank',
        )
        add_aviary_output(
            self,
            Aircraft.Fuel.FUEL_SYSTEM_MASS,
            units='lbm',
        )

    def compute(self, inputs, outputs):
        tank_weight_kg = inputs['tank_weight_kg']
        outputs[Aircraft.Fuel.FUEL_SYSTEM_MASS] = (
            tank_weight_kg * KG_TO_LBM
        )


# ── External subsystem builder ────────────────────────────────
class LH2TankBuilder(SubsystemBuilder):
    """Builder that integrates HyTank into Aviary pre-mission."""

    _default_name = 'lh2_tank'

    def build_pre_mission(
        self, aviary_inputs, subsystem_options=None,
    ):
        """Return the LH2 tank weight group for pre-mission."""
        return LH2TankWeightComp()

    def get_mass_names(self, aviary_inputs=None):
        """No extra mass roll-up names needed.

        The tank weight is already mapped to
        Aircraft.Fuel.FUEL_SYSTEM_MASS, which Aviary's core
        mass buildup already includes. Returning an empty list
        avoids double-counting.
        """
        return []


# ── Main script ───────────────────────────────────────────────
if __name__ == '__main__':
    # 1. Create the Aviary problem
    prob = AviaryProblem()

    # 2. Load the advanced single aisle aircraft definition
    csv_path = get_aviary_resource_path(
        'models/aircraft/advanced_single_aisle'
        '/advanced_single_aisle_FLOPS.csv'
    )
    prob.load_inputs(csv_path, phase_info)

    # 3. Register the LH2 tank external subsystem
    lh2_builder = LH2TankBuilder()
    prob.load_external_subsystems([lh2_builder])

    # 4. Preprocess, build, and configure the model
    prob.check_and_preprocess_inputs()
    prob.build_model()

    # 5. Add optimizer, design variables, and objective
    prob.add_driver('IPOPT')
    prob.add_design_variables()
    prob.add_objective('fuel_burned')

    # 6. Setup the problem
    prob.setup()

    # 7. Set LH2 tank design inputs (override defaults)
    prob.set_val(
        'pre_mission.lh2_tank.radius', 1.5, units='m',
    )
    prob.set_val(
        'pre_mission.lh2_tank.length', 6.0, units='m',
    )
    prob.set_val(
        'pre_mission.lh2_tank.N_layers', 30,
    )
    prob.set_val(
        'pre_mission.lh2_tank.vacuum_gap', 5, units='cm',
    )
    prob.set_val(
        'pre_mission.lh2_tank.environment_design_pressure',
        1.0,
        units='bar',
    )
    prob.set_val(
        'pre_mission.lh2_tank.max_expected_operating_pressure',
        5.0,
        units='bar',
    )

    # 8. Run the Aviary problem
    prob.run_aviary_problem()

    # 9. Print results
    tank_mass_lbm = prob.get_val(
        Aircraft.Fuel.FUEL_SYSTEM_MASS, units='lbm',
    ).item()
    print(
        f'\nLH2 Tank Weight (fuel system mass): '
        f'{tank_mass_lbm:.2f} lbm'
    )
    print(
        f'LH2 Tank Weight: '
        f'{tank_mass_lbm / KG_TO_LBM:.2f} kg'
    )
