"""Aviary + HyTank LH2 integration example.

Demonstrates how to use HyTank's LH2 vacuum tank model as an
external subsystem in an Aviary advanced single aisle problem.
The tank weight computed by HyTank overrides
``Aircraft.Fuel.FUEL_SYSTEM_MASS`` in Aviary's pre-mission mass
buildup, and the post-mission thermal/boil-off analysis consumes
the mission's climb/cruise/descent fuel-flow trajectory via
post-mission bus variables.
"""

import aviary.api as av
from aviary.api import Aircraft as _AviaryAircraft
from aviary.api import CoreMetaData
from aviary.core.aviary_problem import AviaryProblem
from aviary.utils.functions import get_aviary_resource_path
from aviary.variable_info.variables import Mission

from aviary.models.aircraft.advanced_single_aisle.phase_info import (
    phase_info,
)

from hytank.LH2_tank import LH2TankThermals

from cryo_builder import CryoTankBuilder, KG_TO_LBM


# ── Aviary data-hierarchy extension ───────────────────────────
class Aircraft(_AviaryAircraft):
    """Aircraft data hierarchy extended with LH2 tank inputs."""

    class Fuel(_AviaryAircraft.Fuel):
        class LH2Tank:
            RADIUS = 'aircraft:fuel:lh2_tank:radius'
            LENGTH = 'aircraft:fuel:lh2_tank:length'
            N_LAYERS = 'aircraft:fuel:lh2_tank:n_layers'
            VACUUM_GAP = 'aircraft:fuel:lh2_tank:vacuum_gap'
            ENV_DESIGN_PRESSURE = (
                'aircraft:fuel:lh2_tank:environment_design_pressure'
            )
            MAX_OPERATING_PRESSURE = (
                'aircraft:fuel:lh2_tank:max_expected_operating_pressure'
            )


ExtendedMetaData = CoreMetaData

av.add_meta_data(
    Aircraft.Fuel.LH2Tank.RADIUS,
    units='m',
    desc='Inner radius of the cylindrical LH2 tank.',
    default_value=1.5,
    meta_data=ExtendedMetaData,
)
av.add_meta_data(
    Aircraft.Fuel.LH2Tank.LENGTH,
    units='m',
    desc='Length of the cylindrical section of the LH2 tank.',
    default_value=6.0,
    meta_data=ExtendedMetaData,
)
av.add_meta_data(
    Aircraft.Fuel.LH2Tank.N_LAYERS,
    units='unitless',
    desc='Number of MLI reflective shield layers.',
    default_value=30.0,
    meta_data=ExtendedMetaData,
)
av.add_meta_data(
    Aircraft.Fuel.LH2Tank.VACUUM_GAP,
    units='cm',
    desc='Thickness of the vacuum insulation gap.',
    default_value=5.0,
    meta_data=ExtendedMetaData,
)
av.add_meta_data(
    Aircraft.Fuel.LH2Tank.ENV_DESIGN_PRESSURE,
    units='bar',
    desc='External environment design pressure for the tank.',
    default_value=1.0,
    meta_data=ExtendedMetaData,
)
av.add_meta_data(
    Aircraft.Fuel.LH2Tank.MAX_OPERATING_PRESSURE,
    units='bar',
    desc='Maximum expected operating pressure inside the tank.',
    default_value=5.0,
    meta_data=ExtendedMetaData,
)

# Mapping from generic CryoTankBuilder keys to LH2 Aviary variables.
_LH2_TANK_VARS = {
    'RADIUS': Aircraft.Fuel.LH2Tank.RADIUS,
    'LENGTH': Aircraft.Fuel.LH2Tank.LENGTH,
    'N_LAYERS': Aircraft.Fuel.LH2Tank.N_LAYERS,
    'VACUUM_GAP': Aircraft.Fuel.LH2Tank.VACUUM_GAP,
    'ENV_DESIGN_PRESSURE': Aircraft.Fuel.LH2Tank.ENV_DESIGN_PRESSURE,
    'MAX_OPERATING_PRESSURE': (
        Aircraft.Fuel.LH2Tank.MAX_OPERATING_PRESSURE
    ),
}


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

    # 3. Register the LH2 tank external subsystem.
    # ullage_T_init=22 K: the LH2TankThermals default of 21 K sits
    # just below the ~21.77 K saturation temperature at 1.5 bar,
    # which causes the boil-off Newton solver to fail at startup.
    lh2_builder = CryoTankBuilder(
        name='lh2_tank',
        meta_data=ExtendedMetaData,
        tank_vars=_LH2_TANK_VARS,
        thermals_class=LH2TankThermals,
        thermals_kwargs={'ullage_T_init': 22.0},
        t_env_default=273.0,
    )
    prob.load_external_subsystems([lh2_builder])

    # 4. Preprocess, build, and configure the model
    prob.check_and_preprocess_inputs()
    prob.build_model()

    # 4b. Let OpenMDAO auto-reorder the core pre-mission group from
    # its data connections rather than declaration order. Set here
    # (after build_model, before setup) so Aviary core stays
    # unmodified.
    prob.model.pre_mission.options['auto_order'] = True

    # 5. Add optimizer, design variables, and objective
    prob.add_driver('IPOPT')
    prob.add_design_variables()
    prob.add_objective('fuel_burned')

    # 6. Setup the problem
    prob.setup()

    # 7. Set LH2 tank design inputs once via the shared
    # Aircraft.* hierarchy. Each Aircraft.Fuel.LH2Tank.* variable
    # is promoted to the top of the model out of both the
    # pre-mission weight component and the post-mission thermal
    # component, so a single ``set_val`` propagates to both.
    # Tank is sized generously so HyTank's LiquidHeight geometry
    # solver stays well-conditioned over the full mission window
    # even with the un-converged optimizer's early iterations.
    prob.set_val(Aircraft.Fuel.LH2Tank.RADIUS, 2.0, units='m')
    prob.set_val(Aircraft.Fuel.LH2Tank.LENGTH, 5.0, units='m')
    prob.set_val(Aircraft.Fuel.LH2Tank.N_LAYERS, 30)
    prob.set_val(Aircraft.Fuel.LH2Tank.VACUUM_GAP, 5, units='cm')
    prob.set_val(
        Aircraft.Fuel.LH2Tank.ENV_DESIGN_PRESSURE, 1.0, units='bar',
    )
    prob.set_val(
        Aircraft.Fuel.LH2Tank.MAX_OPERATING_PRESSURE, 5.0, units='bar',
    )
    # Scale the mission fuel-flow rate seen by HyTank.
    # 1.0 = use mission values as-is; 0.25 = quarter the extraction rate.
    prob.set_val('lh2_tank.assembler.flow_rate_scale', 0.25)

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

    # 10. Post-mission tank state driven by climb/cruise/descent bus.
    duration_s = prob.get_val(
        'lh2_tank.assembler.mission_duration', units='s',
    ).item()
    m_dot_liq_kgps = prob.get_val('lh2_tank.m_dot_liq_out', units='kg/s')
    m_liq_kg = prob.get_val('lh2_tank.m_liq', units='kg')
    P_bar = prob.get_val('lh2_tank.P', units='bar')
    fill = prob.get_val('lh2_tank.fill_level')
    print(
        f'\nMission duration (all phases): '
        f'{duration_s:.1f} s ({duration_s / 3600:.2f} h)'
    )
    print(
        f'm_dot_liq_out (HyTank grid, kg/s): '
        f'min={m_dot_liq_kgps.min():.4f}, '
        f'max={m_dot_liq_kgps.max():.4f}'
    )
    print(
        f'LH2 liquid mass: '
        f'start={m_liq_kg[0]:.1f} kg, '
        f'end={m_liq_kg[-1]:.1f} kg, '
        f'burned={m_liq_kg[0] - m_liq_kg[-1]:.1f} kg'
    )
    print(
        f'Ullage pressure: '
        f'start={P_bar[0]:.2f} bar, '
        f'end={P_bar[-1]:.2f} bar'
    )
    print(
        f'Fill level: '
        f'start={fill[0] * 100:.1f}%, '
        f'end={fill[-1] * 100:.1f}%'
    )
    print(Mission.BLOCK_FUEL, prob.get_val(Mission.BLOCK_FUEL))
