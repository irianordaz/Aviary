"""Aviary + HyTank LNG integration example.

Demonstrates how to use HyTank's LNG (liquefied methane) vacuum
tank model as an external subsystem in an Aviary advanced single
aisle problem. The tank weight computed by HyTank overrides
``Aircraft.Fuel.FUEL_SYSTEM_MASS`` in Aviary's pre-mission mass
buildup, and the post-mission thermal/boil-off analysis consumes
the mission's climb/cruise/descent fuel-flow trajectory via
post-mission bus variables.

Methane is ~6x denser than liquid hydrogen (~422 vs ~71 kg/m³),
so a smaller tank is realistic. Saturation temperature is ~111 K
at 1 atm (vs ~20 K for LH2); ``LNGTankThermals`` pre-sets initial
conditions to match.
"""

from hytank.LNG_tank import LNGTankThermals

import aviary.api as av
from aviary.api import Aircraft as _AviaryAircraft
from aviary.api import CoreMetaData
from aviary.core.aviary_problem import AviaryProblem
from aviary.models.aircraft.advanced_single_aisle.phase_info import phase_info
from aviary.utils.functions import get_aviary_resource_path
from aviary.variable_info.variables import Mission
from cryo_builder import KG_TO_LBM, CryoTankBuilder


# ── Aviary data-hierarchy extension ───────────────────────────
class Aircraft(_AviaryAircraft):
    """Aircraft data hierarchy extended with LNG tank inputs."""

    class Fuel(_AviaryAircraft.Fuel):
        class LNGTank:
            RADIUS = 'aircraft:fuel:lng_tank:radius'
            LENGTH = 'aircraft:fuel:lng_tank:length'
            N_LAYERS = 'aircraft:fuel:lng_tank:n_layers'
            VACUUM_GAP = 'aircraft:fuel:lng_tank:vacuum_gap'
            ENV_DESIGN_PRESSURE = (
                'aircraft:fuel:lng_tank:environment_design_pressure'
            )
            MAX_OPERATING_PRESSURE = (
                'aircraft:fuel:lng_tank:max_expected_operating_pressure'
            )


ExtendedMetaData = CoreMetaData

av.add_meta_data(
    Aircraft.Fuel.LNGTank.RADIUS,
    units='m',
    desc='Inner radius of the cylindrical LNG tank.',
    default_value=1.5,
    meta_data=ExtendedMetaData,
)
av.add_meta_data(
    Aircraft.Fuel.LNGTank.LENGTH,
    units='m',
    desc='Length of the cylindrical section of the LNG tank.',
    default_value=6.0,
    meta_data=ExtendedMetaData,
)
av.add_meta_data(
    Aircraft.Fuel.LNGTank.N_LAYERS,
    units='unitless',
    desc='Number of MLI reflective shield layers.',
    default_value=30.0,
    meta_data=ExtendedMetaData,
)
av.add_meta_data(
    Aircraft.Fuel.LNGTank.VACUUM_GAP,
    units='cm',
    desc='Thickness of the vacuum insulation gap.',
    default_value=5.0,
    meta_data=ExtendedMetaData,
)
av.add_meta_data(
    Aircraft.Fuel.LNGTank.ENV_DESIGN_PRESSURE,
    units='bar',
    desc='External environment design pressure for the tank.',
    default_value=1.0,
    meta_data=ExtendedMetaData,
)
av.add_meta_data(
    Aircraft.Fuel.LNGTank.MAX_OPERATING_PRESSURE,
    units='bar',
    desc='Maximum expected operating pressure inside the tank.',
    default_value=5.0,
    meta_data=ExtendedMetaData,
)

# Mapping from generic CryoTankBuilder keys to LNG Aviary variables.
_LNG_TANK_VARS = {
    'RADIUS': Aircraft.Fuel.LNGTank.RADIUS,
    'LENGTH': Aircraft.Fuel.LNGTank.LENGTH,
    'N_LAYERS': Aircraft.Fuel.LNGTank.N_LAYERS,
    'VACUUM_GAP': Aircraft.Fuel.LNGTank.VACUUM_GAP,
    'ENV_DESIGN_PRESSURE': Aircraft.Fuel.LNGTank.ENV_DESIGN_PRESSURE,
    'MAX_OPERATING_PRESSURE': (Aircraft.Fuel.LNGTank.MAX_OPERATING_PRESSURE),
}


# ── Main script ───────────────────────────────────────────────
if __name__ == '__main__':
    # 1. Create the Aviary problem
    prob = AviaryProblem()

    # 2. Load the advanced single aisle aircraft definition
    csv_path = get_aviary_resource_path(
        'models/aircraft/advanced_single_aisle/advanced_single_aisle_FLOPS.csv'
    )
    prob.load_inputs(csv_path, phase_info)

    # 3. Register the LNG tank external subsystem.
    # ullage_T_init=117 K and liquid_T_init=117 K: the LNGTankThermals
    # defaults (111 K / 111 K) sit below the ~116.66 K saturation
    # temperature of methane at the 1.5 bar default ullage pressure.
    # At sub-saturation temperatures CoolProp returns liquid density
    # for the ullage (~1200 kg of "gas"), and saturated-property
    # lookups fall outside CoolProp's valid two-phase range, raising
    # "max() iterable argument is empty" and failing the boil-off IVP
    # initial-guess solver. 117 K keeps both ullage and liquid in the
    # proper two-phase region.
    lng_builder = CryoTankBuilder(
        name='lng_tank',
        meta_data=ExtendedMetaData,
        tank_vars=_LNG_TANK_VARS,
        thermals_class=LNGTankThermals,
        thermals_kwargs={
            'ullage_T_init': 117.0,
            'liquid_T_init': 117.0,
        },
        t_env_default=300.0,
    )
    prob.load_external_subsystems([lng_builder])

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
    prob.driver.opt_settings['max_iter'] = 200
    prob.add_design_variables()
    prob.add_objective('fuel_burned')

    # 6. Setup the problem
    prob.setup()

    # 7. Set LNG tank design inputs once via the shared
    # Aircraft.* hierarchy. Each Aircraft.Fuel.LNGTank.* variable
    # is promoted to the top of the model out of both the
    # pre-mission weight component and the post-mission thermal
    # component, so a single ``set_val`` propagates to both.
    #
    # Methane is ~6x denser than LH2 (~422 vs ~71 kg/m³) so a
    # 1.5 m x 6 m tank holds ~22.6 tonnes of LNG at 95% fill —
    # ample headroom for the un-converged optimizer's early
    # iterations while keeping the tank realistically sized.
    prob.set_val(Aircraft.Fuel.LNGTank.RADIUS, 1.5, units='m')
    prob.set_val(Aircraft.Fuel.LNGTank.LENGTH, 6.0, units='m')
    prob.set_val(Aircraft.Fuel.LNGTank.N_LAYERS, 30)
    prob.set_val(Aircraft.Fuel.LNGTank.VACUUM_GAP, 5, units='cm')
    prob.set_val(
        Aircraft.Fuel.LNGTank.ENV_DESIGN_PRESSURE,
        1.0,
        units='bar',
    )
    prob.set_val(
        Aircraft.Fuel.LNGTank.MAX_OPERATING_PRESSURE,
        5.0,
        units='bar',
    )
    # Scale the mission fuel-flow rate seen by HyTank.
    # 1.0 = use mission values as-is; 2.0 = double the extraction rate.
    prob.set_val('lng_tank.assembler.flow_rate_scale', 1.0)

    # Takeoff inputs required by the simplified takeoff group when
    # phase_info enables ``include_takeoff``. The advanced single
    # aisle CSV ships ``thrust_takeoff_per_eng = 0`` and omits
    # ``lift_over_drag`` (whose metadata default is 0). The takeoff
    # rolling-distance formula divides by lift_over_drag, so a 0 value
    # yields inf/NaN that poisons the takeoff->climb mach/altitude
    # connection constraints and aborts IPOPT on the first iteration.
    # These values match the sibling FLOPS benchmark for this airframe.
    prob.set_val(Aircraft.Design.THRUST_TAKEOFF_PER_ENG, 28928.1, units='lbf')
    prob.set_val(Mission.Takeoff.LIFT_OVER_DRAG, 17.354)

    # 8. Run the Aviary problem
    prob.run_aviary_problem()

    # 9. Print results
    tank_mass_lbm = prob.get_val(
        Aircraft.Fuel.FUEL_SYSTEM_MASS,
        units='lbm',
    ).item()
    print(f'\nLNG Tank Weight (fuel system mass): {tank_mass_lbm:.2f} lbm')
    print(f'LNG Tank Weight: {tank_mass_lbm / KG_TO_LBM:.2f} kg')

    # 10. Post-mission tank state driven by climb/cruise/descent bus.
    duration_s = prob.get_val(
        'lng_tank.assembler.mission_duration',
        units='s',
    ).item()
    m_dot_liq_kgps = prob.get_val(
        'lng_tank.m_dot_liq_out',
        units='kg/s',
    )
    m_liq_kg = prob.get_val('lng_tank.m_liq', units='kg')
    P_bar = prob.get_val('lng_tank.P', units='bar')
    T_gas_K = prob.get_val('lng_tank.T_gas', units='K')
    T_liq_K = prob.get_val('lng_tank.T_liq', units='K')
    fill = prob.get_val('lng_tank.fill_level')
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
        f'LNG liquid mass: '
        f'start={m_liq_kg[0]:.1f} kg, '
        f'end={m_liq_kg[-1]:.1f} kg, '
        f'burned={m_liq_kg[0] - m_liq_kg[-1]:.1f} kg'
    )
    print(f'Ullage pressure: start={P_bar[0]:.2f} bar, end={P_bar[-1]:.2f} bar')
    print(
        f'Ullage temperature: start={T_gas_K[0]:.2f} K, end={T_gas_K[-1]:.2f} K'
    )
    print(
        f'Liquid temperature: start={T_liq_K[0]:.2f} K, end={T_liq_K[-1]:.2f} K'
    )
    print(f'Fill level: start={fill[0] * 100:.1f}%, end={fill[-1] * 100:.1f}%')
    print(Mission.BLOCK_FUEL, prob.get_val(Mission.BLOCK_FUEL))
