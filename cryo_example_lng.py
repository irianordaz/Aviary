"""Aviary + HyTank LNG integration example.

Demonstrates how to use HyTank's LNG (liquefied methane) vacuum
tank model as an external subsystem in an Aviary advanced single
aisle problem. The tank weight computed by HyTank overrides
Aircraft.Fuel.FUEL_SYSTEM_MASS in Aviary's pre-mission mass
buildup, and the post-mission thermal/boil-off analysis consumes
the mission's climb/cruise/descent fuel-flow trajectory via
post-mission bus variables.

Mirrors ``cryo_example.py`` but selects LNG via HyTank's
``propellant`` option (through :class:`LNGTankThermals`). Methane
is ~6x denser than liquid hydrogen, so a smaller tank is
realistic; saturation temperatures are ~111 K at 1 atm (vs ~20 K
for LH2) and the LNG tank wrappers pre-set initial conditions to
match.
"""

import numpy as np
import openmdao.api as om

import aviary.api as av
from aviary.api import Aircraft as _AviaryAircraft
from aviary.api import CoreMetaData
from aviary.core.aviary_problem import AviaryProblem
from aviary.subsystems.subsystem_builder import SubsystemBuilder
from aviary.utils.functions import get_aviary_resource_path
from aviary.variable_info.functions import (
    add_aviary_output,
)
from aviary.variable_info.variables import Dynamic

from aviary.models.aircraft.advanced_single_aisle.phase_info import (
    phase_info,
)

from hytank.weight import VacuumTankWeight
from hytank.LNG_tank import LNGTankThermals

# ── Constants ──────────────────────────────────────────────────
KG_TO_LBM = 2.20462


# ── Aviary data-hierarchy extension ───────────────────────────
# Add LNG tank geometry to the Aircraft.Fuel namespace so the
# pre-mission and post-mission tank wrappers can share a single
# source of truth via Aviary's standard ``aircraft:*`` promotion
# mechanism (see ``promote_aircraft_and_mission_vars``).
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


# ── Pre-mission wrapper component ─────────────────────────────
class LNGTankWeightComp(om.Group):
    """Wraps HyTank VacuumTankWeight and outputs fuel system mass.

    The VacuumTankWeight component computes the structural
    weight of a vacuum-insulated tank in kilograms. The
    structural model is geometry-driven; the same vacuum-tank
    wall sizing applies for LNG as for LH2 (cryogenic aluminum
    properties are actually conservative at LNG's warmer
    ~111 K vs LH2's ~20 K). This wrapper converts that weight
    to lbm and promotes it as Aircraft.Fuel.FUEL_SYSTEM_MASS so
    Aviary can consume it.
    """

    def setup(self):
        self.options['auto_order'] = True

        self.add_subsystem(
            'tank_weight',
            VacuumTankWeight(),
            promotes_inputs=[
                (
                    'environment_design_pressure',
                    Aircraft.Fuel.LNGTank.ENV_DESIGN_PRESSURE,
                ),
                (
                    'max_expected_operating_pressure',
                    Aircraft.Fuel.LNGTank.MAX_OPERATING_PRESSURE,
                ),
                ('vacuum_gap', Aircraft.Fuel.LNGTank.VACUUM_GAP),
                ('radius', Aircraft.Fuel.LNGTank.RADIUS),
                ('length', Aircraft.Fuel.LNGTank.LENGTH),
                ('N_layers', Aircraft.Fuel.LNGTank.N_LAYERS),
            ],
            promotes_outputs=[('weight', 'tank_weight_kg')],
        )

        self.add_subsystem(
            'unit_conversion',
            _TankWeightToFuelSystemMass(),
            promotes_inputs=['tank_weight_kg'],
            promotes_outputs=['*'],
        )


class _TankWeightToFuelSystemMass(om.ExplicitComponent):
    """Convert tank weight (kg) to fuel system mass (lbm)."""

    def setup(self):
        self.add_input(
            'tank_weight_kg',
            val=0.0,
            units='kg',
            desc='LNG tank structural weight from HyTank',
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


_PHASES = ('climb', 'cruise', 'descent')

# Sensible default phase time windows (s) for setup-time before
# the bus connections are made. Approx values for a single-aisle
# mission: 30 min climb, 2 h cruise, 30 min descent.
_PHASE_TIME_DEFAULTS = {
    'climb': (0.0, 1800.0),
    'cruise': (1800.0, 9000.0),
    'descent': (9000.0, 10800.0),
}


# ── Mission fuel-flow assembler ───────────────────────────────
class _MissionFuelFlowAssembler(om.ExplicitComponent):
    """Concatenate climb/cruise/descent fuel-flow into HyTank's grid.

    Joins the three phase ``fuel_flow_rate_negative_total``
    trajectories along their absolute time axes, flips sign so
    consumption is positive (HyTank convention), and linearly
    resamples onto an odd-sized grid of ``num_nodes`` spanning
    [t_climb_start, t_descent_end]. Also outputs the total
    mission duration for HyTank's integrator.
    """

    def initialize(self):
        for phase in _PHASES:
            self.options.declare(f'{phase}_length', types=int)
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        for phase in _PHASES:
            length = self.options[f'{phase}_length']
            t0, t1 = _PHASE_TIME_DEFAULTS[phase]
            self.add_input(
                f'fuel_flow_rate_{phase}',
                val=np.zeros(length),
                units='kg/s',
            )
            self.add_input(
                f'time_{phase}',
                val=np.linspace(t0, t1, length),
                units='s',
            )
        self.add_output(
            'm_dot_liq_out', val=np.zeros(nn), units='kg/s',
        )
        self.add_output(
            'mission_duration', val=1.0, units='s',
        )
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']

        t_segments = [inputs[f'time_{p}'] for p in _PHASES]
        ff_segments = [
            -inputs[f'fuel_flow_rate_{p}'] for p in _PHASES
        ]
        t_all = np.concatenate(t_segments)
        ff_all = np.concatenate(ff_segments)

        # Sort by time so the interp grid is monotonic even if
        # phases overlap on a boundary node.
        order = np.argsort(t_all)
        t_all = t_all[order]
        ff_all = ff_all[order]

        t_start = t_all[0]
        t_end = t_all[-1]
        outputs['mission_duration'] = t_end - t_start

        if t_end > t_start:
            t_target = np.linspace(t_start, t_end, nn)
            outputs['m_dot_liq_out'] = np.interp(
                t_target, t_all, ff_all,
            )
        else:
            outputs['m_dot_liq_out'] = np.full(nn, ff_all.mean())


# ── Post-mission wrapper component ────────────────────────────
class LNGTankPostMissionComp(om.Group):
    """HyTank LNG tank thermal analysis driven by the full mission.

    Pulls ``fuel_flow_rate_negative_total`` and ``time`` from
    each of climb, cruise, and descent via post-mission bus
    variables, concatenates and resamples them onto an
    odd-sized grid (``num_nodes``), and feeds them to HyTank's
    :class:`LNGTankThermals`. The mission duration (descent end
    minus climb start) drives HyTank's bdf3 integrator.

    :class:`LNGTankThermals` ships with methane-appropriate
    initial conditions (118 K ullage / 113 K liquid at 1.5 bar,
    above/below the ~117 K saturation point) so no extra
    overrides are needed here.
    """

    def initialize(self):
        self.options.declare(
            'num_nodes',
            types=int,
            default=11,
            desc=(
                'Number of post-mission analysis nodes; '
                'must be odd (2N+1).'
            ),
        )
        for phase in _PHASES:
            self.options.declare(
                f'{phase}_bus_length',
                types=int,
                default=4,
                desc=f'Length of the {phase}-phase trajectory bus.',
            )

    def setup(self):
        self.options['auto_order'] = True
        nn = self.options['num_nodes']
        if (nn - 1) % 2 != 0:
            raise ValueError(
                f'num_nodes must be odd, got {nn}.'
            )

        assembler_kwargs = {
            f'{p}_length': self.options[f'{p}_bus_length']
            for p in _PHASES
        }
        promoted_inputs = []
        for p in _PHASES:
            promoted_inputs.append(f'fuel_flow_rate_{p}')
            promoted_inputs.append(f'time_{p}')

        self.add_subsystem(
            'assembler',
            _MissionFuelFlowAssembler(
                num_nodes=nn, **assembler_kwargs,
            ),
            promotes_inputs=promoted_inputs,
            promotes_outputs=['m_dot_liq_out'],
        )

        self.add_subsystem(
            'tank_thermals',
            LNGTankThermals(num_nodes=nn),
            promotes_inputs=[
                ('radius', Aircraft.Fuel.LNGTank.RADIUS),
                ('length', Aircraft.Fuel.LNGTank.LENGTH),
                ('N_layers', Aircraft.Fuel.LNGTank.N_LAYERS),
                'P_heater',
                'm_dot_gas_out',
                'm_dot_liq_out',
                'T_env',
            ],
            promotes_outputs=[
                'm_gas',
                'm_liq',
                'T_gas',
                'T_liq',
                'P',
                'fill_level',
            ],
        )

        # Drive HyTank's integrator window with the mission duration.
        self.connect(
            'assembler.mission_duration',
            'tank_thermals.boil_off.integ.duration',
        )

        # ── Defaults for inputs not wired to the trajectory ──
        self.set_input_defaults(
            'P_heater', np.zeros(nn), units='W',
        )
        self.set_input_defaults(
            'm_dot_gas_out', np.zeros(nn), units='kg/s',
        )
        self.set_input_defaults(
            'T_env', np.full(nn, 300.0), units='K',
        )
        for p in _PHASES:
            bl = self.options[f'{p}_bus_length']
            t0, t1 = _PHASE_TIME_DEFAULTS[p]
            self.set_input_defaults(
                f'fuel_flow_rate_{p}',
                np.zeros(bl),
                units='kg/s',
            )
            self.set_input_defaults(
                f'time_{p}',
                np.linspace(t0, t1, bl),
                units='s',
            )


# ── External subsystem builder ────────────────────────────────
class LNGTankBuilder(SubsystemBuilder):
    """Builder that integrates HyTank LNG into Aviary pre/post-mission."""

    _default_name = 'lng_tank'
    _default_metadata = ExtendedMetaData

    def build_pre_mission(
        self, aviary_inputs, subsystem_options=None,
    ):
        """Return the LNG tank weight group for pre-mission."""
        return LNGTankWeightComp()

    def build_post_mission(
        self,
        aviary_inputs=None,
        mission_info=None,
        subsystem_options=None,
        phase_mission_bus_lengths=None,
    ):
        """Return the LNG tank thermal analysis for post-mission.

        Embedding HyTank's BoilOff inside a Dymos mission ODE
        would double-integrate time (Dymos collocates the phase,
        while HyTank's Integrator does its own bdf3 quadrature)
        and clashes on parity (HyTank needs odd num_nodes, Dymos
        gives even). Post-mission lets HyTank run its own time
        integration after the mission trajectory has been solved
        while still consuming the climb/cruise/descent fuel-flow
        and time profiles via bus variables.
        """
        bus_lengths = {p: 4 for p in _PHASES}
        if phase_mission_bus_lengths is not None:
            for p in _PHASES:
                if p in phase_mission_bus_lengths:
                    bus_lengths[p] = phase_mission_bus_lengths[p]
        return LNGTankPostMissionComp(
            num_nodes=11,
            climb_bus_length=bus_lengths['climb'],
            cruise_bus_length=bus_lengths['cruise'],
            descent_bus_length=bus_lengths['descent'],
        )

    def get_post_mission_bus_variables(
        self, aviary_inputs=None, mission_info=None,
    ):
        """Expose climb/cruise/descent fuel-flow and time to post-mission.

        Aviary's flight phase ODE outputs
        ``fuel_flow_rate_negative_total`` (mass-rate convention,
        negative on consumption). The assembler negates it to
        match HyTank's positive-on-extraction convention. Time
        is needed to assemble a single absolute-time trajectory
        across phases and to drive HyTank's integrator duration.
        """
        bus = {}
        for p in _PHASES:
            bus[p] = {
                Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL: {
                    'post_mission_name': f'lng_tank.fuel_flow_rate_{p}',
                },
                'time': {
                    'post_mission_name': f'lng_tank.time_{p}',
                },
            }
        return bus

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

    # 3. Register the LNG tank external subsystem
    lng_builder = LNGTankBuilder()
    prob.load_external_subsystems([lng_builder])

    # 4. Preprocess, build, and configure the model
    prob.check_and_preprocess_inputs()
    prob.build_model()

    # 5. Add optimizer, design variables, and objective
    prob.add_driver('IPOPT')
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
    # Methane is ~6x denser than LH2 (~422 vs ~71 kg/m^3) so a
    # 1.5 m x 6 m tank holds ~22.6 tonnes of LNG at 95% fill —
    # ample headroom for the un-converged optimizer's early
    # iterations while keeping the tank realistically sized.
    prob.set_val(
        Aircraft.Fuel.LNGTank.RADIUS, 1.5, units='m',
    )
    prob.set_val(
        Aircraft.Fuel.LNGTank.LENGTH, 6.0, units='m',
    )
    prob.set_val(
        Aircraft.Fuel.LNGTank.N_LAYERS, 30,
    )
    prob.set_val(
        Aircraft.Fuel.LNGTank.VACUUM_GAP, 5, units='cm',
    )
    prob.set_val(
        Aircraft.Fuel.LNGTank.ENV_DESIGN_PRESSURE,
        1.0, units='bar',
    )
    prob.set_val(
        Aircraft.Fuel.LNGTank.MAX_OPERATING_PRESSURE,
        5.0, units='bar',
    )

    # 8. Run the Aviary problem
    prob.run_aviary_problem()

    # 9. Print results
    tank_mass_lbm = prob.get_val(
        Aircraft.Fuel.FUEL_SYSTEM_MASS, units='lbm',
    ).item()
    print(
        f'\nLNG Tank Weight (fuel system mass): '
        f'{tank_mass_lbm:.2f} lbm'
    )
    print(
        f'LNG Tank Weight: '
        f'{tank_mass_lbm / KG_TO_LBM:.2f} kg'
    )

    # 10. Post-mission tank state driven by climb/cruise/descent bus.
    duration_s = prob.get_val(
        'lng_tank.assembler.mission_duration', units='s',
    ).item()
    m_dot_liq_kgps = prob.get_val(
        'lng_tank.m_dot_liq_out', units='kg/s',
    )
    m_liq_kg = prob.get_val(
        'lng_tank.m_liq', units='kg',
    )
    P_bar = prob.get_val(
        'lng_tank.P', units='bar',
    )
    T_gas_K = prob.get_val(
        'lng_tank.T_gas', units='K',
    )
    T_liq_K = prob.get_val(
        'lng_tank.T_liq', units='K',
    )
    fill = prob.get_val('lng_tank.fill_level')
    print(
        f'\nMission duration (climb+cruise+descent): '
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
    print(
        f'Ullage pressure: '
        f'start={P_bar[0]:.2f} bar, '
        f'end={P_bar[-1]:.2f} bar'
    )
    print(
        f'Ullage temperature: '
        f'start={T_gas_K[0]:.2f} K, '
        f'end={T_gas_K[-1]:.2f} K'
    )
    print(
        f'Liquid temperature: '
        f'start={T_liq_K[0]:.2f} K, '
        f'end={T_liq_K[-1]:.2f} K'
    )
    print(
        f'Fill level: '
        f'start={fill[0] * 100:.1f}%, '
        f'end={fill[-1] * 100:.1f}%'
    )
