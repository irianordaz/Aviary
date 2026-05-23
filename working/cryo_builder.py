"""Shared infrastructure for cryogenic tank external subsystems in Aviary.

Provides the generic builder, weight component, post-mission thermal
component, and fuel-flow assembler used by both the LH2 and LNG
integration examples. Each propellant-specific example imports from
this module and configures a :class:`CryoTankBuilder` with its own:

  * Aircraft variable name mapping (``tank_vars``)
  * HyTank thermals class (e.g. ``LH2TankThermals`` or ``LNGTankThermals``)
  * Extra thermals constructor kwargs (e.g. ``ullage_T_init`` for LH2)
  * Default environment temperature (``t_env_default``)
"""

import numpy as np
import openmdao.api as om

from aviary.api import Aircraft as _Aircraft
from aviary.subsystems.subsystem_builder import SubsystemBuilder
from aviary.variable_info.functions import add_aviary_output
from aviary.variable_info.variables import Dynamic

from hytank.weight import VacuumTankWeight

KG_TO_LBM = 2.20462


def _phase_time_defaults(phase_names, bus_lengths):
    """Compute sequential default time arrays for each phase.

    Used before bus connections are established so the model has
    non-degenerate time inputs. Each phase gets 100 s per node,
    placed back-to-back with a 100 s gap between phases.

    Args:
        phase_names: Ordered sequence of phase name strings.
        bus_lengths: Mapping from phase name to node count.

    Returns:
        Dict mapping each phase name to a 1-D numpy time array.
    """
    defaults = {}
    offset = 0.0
    for p in phase_names:
        n = bus_lengths[p]
        defaults[p] = np.linspace(
            offset, offset + 100.0 * max(n - 1, 1), n
        )
        offset += 100.0 * max(n, 1) + 100.0
    return defaults


class _TankWeightToFuelSystemMass(om.ExplicitComponent):
    """Convert tank structural weight (kg) to fuel system mass (lbm)."""

    def setup(self):
        self.add_input(
            'tank_weight_kg',
            val=0.0,
            units='kg',
            desc='Cryogenic tank structural weight from HyTank',
        )
        add_aviary_output(
            self,
            _Aircraft.Fuel.FUEL_SYSTEM_MASS,
            units='lbm',
        )

    def compute(self, inputs, outputs):
        outputs[_Aircraft.Fuel.FUEL_SYSTEM_MASS] = (
            inputs['tank_weight_kg'] * KG_TO_LBM
        )


class _MissionFuelFlowAssembler(om.ExplicitComponent):
    """Concatenate per-phase fuel-flow timeseries into HyTank's grid.

    Joins all phase ``fuel_flow_rate_negative_total`` trajectories
    along their absolute time axes, flips sign so consumption is
    positive (HyTank convention), and linearly resamples onto an
    odd-sized grid of ``num_nodes`` spanning the full mission.
    Also outputs the total mission duration for HyTank's integrator.
    """

    def initialize(self):
        self.options.declare(
            'phase_names',
            desc='Ordered sequence of trajectory phase names.',
        )
        self.options.declare(
            'bus_lengths',
            desc='Mapping from phase name to trajectory node count.',
        )
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        phase_names = self.options['phase_names']
        bus_lengths = self.options['bus_lengths']
        time_defaults = _phase_time_defaults(phase_names, bus_lengths)
        for phase in phase_names:
            length = bus_lengths[phase]
            self.add_input(
                f'fuel_flow_rate_{phase}',
                val=np.zeros(length),
                units='kg/s',
            )
            self.add_input(
                f'time_{phase}',
                val=time_defaults[phase],
                units='s',
            )
        self.add_input('flow_rate_scale', val=1.0, units='unitless')
        self.add_output(
            'm_dot_liq_out', val=np.zeros(nn), units='kg/s',
        )
        self.add_output('mission_duration', val=1.0, units='s')
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        phase_names = self.options['phase_names']

        t_segments = [inputs[f'time_{p}'] for p in phase_names]
        ff_segments = [
            -inputs[f'fuel_flow_rate_{p}'] for p in phase_names
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

        scale = inputs['flow_rate_scale']
        if t_end > t_start:
            t_target = np.linspace(t_start, t_end, nn)
            outputs['m_dot_liq_out'] = scale * np.interp(
                t_target, t_all, ff_all,
            )
        else:
            outputs['m_dot_liq_out'] = scale * np.full(nn, ff_all.mean())


class CryoTankWeightComp(om.Group):
    """Generic vacuum-tank weight wrapper for Aviary pre-mission.

    Wraps HyTank's ``VacuumTankWeight`` and promotes the result as
    ``Aircraft.Fuel.FUEL_SYSTEM_MASS``. The mapping from HyTank's
    generic input names to propellant-specific Aviary variable names
    is provided via the ``tank_vars`` option.
    """

    def initialize(self):
        self.options.declare(
            'tank_vars',
            desc=(
                'Dict mapping the keys RADIUS, LENGTH, N_LAYERS, '
                'VACUUM_GAP, ENV_DESIGN_PRESSURE, and '
                'MAX_OPERATING_PRESSURE to Aviary variable name strings.'
            ),
        )

    def setup(self):
        self.options['auto_order'] = True
        tv = self.options['tank_vars']

        self.add_subsystem(
            'tank_weight',
            VacuumTankWeight(),
            promotes_inputs=[
                (
                    'environment_design_pressure',
                    tv['ENV_DESIGN_PRESSURE'],
                ),
                (
                    'max_expected_operating_pressure',
                    tv['MAX_OPERATING_PRESSURE'],
                ),
                ('vacuum_gap', tv['VACUUM_GAP']),
                ('radius', tv['RADIUS']),
                ('length', tv['LENGTH']),
                ('N_layers', tv['N_LAYERS']),
            ],
            promotes_outputs=[('weight', 'tank_weight_kg')],
        )
        self.add_subsystem(
            'unit_conversion',
            _TankWeightToFuelSystemMass(),
            promotes_inputs=['tank_weight_kg'],
            promotes_outputs=['*'],
        )


class CryoTankPostMissionComp(om.Group):
    """Generic HyTank thermal analysis driven by the full mission.

    Pulls ``fuel_flow_rate_negative_total`` and ``time`` from each
    trajectory phase via post-mission bus variables, concatenates
    and resamples them onto an odd-sized grid (``num_nodes``), and
    feeds them to the propellant-specific thermals class. The mission
    duration (last phase end minus first phase start) drives HyTank's
    bdf3 integrator.
    """

    def initialize(self):
        self.options.declare(
            'num_nodes',
            types=int,
            default=11,
            desc='Post-mission analysis nodes; must be odd (2N+1).',
        )
        self.options.declare(
            'phase_names',
            desc='Ordered sequence of trajectory phase names.',
        )
        self.options.declare(
            'bus_lengths',
            desc='Mapping from phase name to trajectory node count.',
        )
        self.options.declare(
            'thermals_class',
            desc='HyTank thermals class to instantiate.',
        )
        self.options.declare(
            'thermals_kwargs',
            default=None,
            desc='Extra keyword args forwarded to thermals_class.',
        )
        self.options.declare(
            'tank_vars',
            desc='Tank variable name mapping (see CryoTankWeightComp).',
        )
        self.options.declare(
            't_env_default',
            default=300.0,
            desc='Default environment temperature (K).',
        )

    def setup(self):
        self.options['auto_order'] = True
        nn = self.options['num_nodes']
        if (nn - 1) % 2 != 0:
            raise ValueError(f'num_nodes must be odd, got {nn}.')
        phase_names = self.options['phase_names']
        bus_lengths = self.options['bus_lengths']
        thermals_class = self.options['thermals_class']
        thermals_kwargs = self.options['thermals_kwargs'] or {}
        tv = self.options['tank_vars']
        t_env_default = self.options['t_env_default']
        time_defaults = _phase_time_defaults(phase_names, bus_lengths)

        promoted_inputs = []
        for p in phase_names:
            promoted_inputs.append(f'fuel_flow_rate_{p}')
            promoted_inputs.append(f'time_{p}')

        self.add_subsystem(
            'assembler',
            _MissionFuelFlowAssembler(
                phase_names=phase_names,
                bus_lengths=bus_lengths,
                num_nodes=nn,
            ),
            promotes_inputs=promoted_inputs,
            promotes_outputs=['m_dot_liq_out'],
        )

        self.add_subsystem(
            'tank_thermals',
            thermals_class(num_nodes=nn, **thermals_kwargs),
            promotes_inputs=[
                ('radius', tv['RADIUS']),
                ('length', tv['LENGTH']),
                ('N_layers', tv['N_LAYERS']),
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
        self.set_input_defaults('P_heater', np.zeros(nn), units='W')
        self.set_input_defaults(
            'm_dot_gas_out', np.zeros(nn), units='kg/s',
        )
        self.set_input_defaults(
            'T_env', np.full(nn, t_env_default), units='K',
        )
        for p in phase_names:
            self.set_input_defaults(
                f'fuel_flow_rate_{p}',
                np.zeros(bus_lengths[p]),
                units='kg/s',
            )
            self.set_input_defaults(
                f'time_{p}',
                time_defaults[p],
                units='s',
            )


class CryoTankBuilder(SubsystemBuilder):
    """Generic Aviary external subsystem builder for cryogenic tanks.

    Configures HyTank's vacuum-tank weight model for pre-mission and
    its thermal boil-off model for post-mission analysis. All
    propellant-specific details are supplied at construction time so
    the same builder works for both LH2 and LNG.

    The phase list and bus lengths are derived automatically from
    ``phase_mission_bus_lengths`` / ``mission_info`` at build time,
    so adding, removing, or renaming phases in ``phase_info`` requires
    no changes to this builder.

    Args:
        name: Aviary subsystem name (e.g. ``'lh2_tank'``). Used as
            the prefix in post-mission bus variable paths.
        meta_data: Extended Aviary metadata dict that includes the
            propellant-specific tank variable declarations.
        tank_vars: Dict with keys RADIUS, LENGTH, N_LAYERS,
            VACUUM_GAP, ENV_DESIGN_PRESSURE, MAX_OPERATING_PRESSURE
            mapping to Aviary variable name strings.
        thermals_class: HyTank thermals class to instantiate
            (e.g. ``LH2TankThermals`` or ``LNGTankThermals``).
        thermals_kwargs: Extra keyword arguments forwarded to
            ``thermals_class`` (e.g. ``{'ullage_T_init': 22.0}``).
        t_env_default: Default environment temperature in K used
            before bus connections are established. Defaults to 300.
        hytank_num_nodes: Odd number of HyTank BDF3 analysis nodes.
            Defaults to 11. Intentionally decoupled from the trajectory
            node count — HyTank's BDF3 Newton solver has dt-dependent
            stability regions; larger node counts reduce dt and can
            push h×λ outside the stability region. The assembler
            resamples the mission trajectory onto this fixed odd grid.
    """

    def __init__(
        self,
        name,
        meta_data,
        tank_vars,
        thermals_class,
        thermals_kwargs=None,
        t_env_default=300.0,
        hytank_num_nodes=11,
    ):
        """Initialize the cryogenic tank builder.

        Args:
            name: Aviary subsystem name string.
            meta_data: Extended Aviary metadata dict.
            tank_vars: Tank variable name mapping.
            thermals_class: HyTank thermals class.
            thermals_kwargs: Extra kwargs for thermals_class.
            t_env_default: Default environment temperature (K).
            hytank_num_nodes: Odd HyTank analysis node count.
        """
        super().__init__(name=name, meta_data=meta_data)
        if hytank_num_nodes < 1 or (hytank_num_nodes - 1) % 2 != 0:
            raise ValueError(
                f'hytank_num_nodes must be odd, got {hytank_num_nodes}.'
            )
        self.tank_vars = tank_vars
        self.thermals_class = thermals_class
        self.thermals_kwargs = (
            thermals_kwargs if thermals_kwargs is not None else {}
        )
        self.t_env_default = t_env_default
        self.hytank_num_nodes = hytank_num_nodes

    def build_pre_mission(self, aviary_inputs, subsystem_options=None):
        """Return the tank weight group for pre-mission."""
        return CryoTankWeightComp(tank_vars=self.tank_vars)

    def build_post_mission(
        self,
        aviary_inputs=None,
        mission_info=None,
        subsystem_options=None,
        phase_mission_bus_lengths=None,
    ):
        """Return the tank thermal analysis for post-mission.

        Embedding HyTank's BoilOff inside a Dymos mission ODE
        would double-integrate time (Dymos collocates the phase,
        while HyTank's Integrator does its own bdf3 quadrature)
        and clashes on parity (HyTank needs odd num_nodes, Dymos
        gives even). Post-mission lets HyTank run its own time
        integration after the trajectory is solved, consuming
        the per-phase fuel-flow and time profiles via bus variables.
        """
        if phase_mission_bus_lengths is not None:
            phase_names = tuple(phase_mission_bus_lengths.keys())
            bus_lengths = dict(phase_mission_bus_lengths)
        elif mission_info is not None:
            phase_names = tuple(mission_info.keys())
            bus_lengths = {p: 4 for p in phase_names}
        else:
            phase_names = ('climb', 'cruise', 'descent')
            bus_lengths = {p: 4 for p in phase_names}
        return CryoTankPostMissionComp(
            num_nodes=self.hytank_num_nodes,
            phase_names=phase_names,
            bus_lengths=bus_lengths,
            thermals_class=self.thermals_class,
            thermals_kwargs=self.thermals_kwargs,
            tank_vars=self.tank_vars,
            t_env_default=self.t_env_default,
        )

    def get_post_mission_bus_variables(
        self, aviary_inputs=None, mission_info=None,
    ):
        """Expose per-phase fuel-flow and time arrays to post-mission.

        Aviary's flight phase ODE outputs
        ``fuel_flow_rate_negative_total`` (mass-rate convention,
        negative on consumption). The assembler negates it to match
        HyTank's positive-on-extraction convention. Time is needed
        to assemble a single absolute-time trajectory across all
        phases and to drive HyTank's integrator duration. The phase
        list is read from ``mission_info`` so any change to
        ``phase_info`` is reflected automatically.
        """
        phases = (
            tuple(mission_info.keys())
            if mission_info is not None
            else ('climb', 'cruise', 'descent')
        )
        bus = {}
        for p in phases:
            bus[p] = {
                Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL: {
                    'post_mission_name': (
                        f'{self.name}.fuel_flow_rate_{p}'
                    ),
                },
                'time': {
                    'post_mission_name': f'{self.name}.time_{p}',
                },
            }
        return bus

    def get_mass_names(self, aviary_inputs=None):
        """Return empty list — tank weight already maps to FUEL_SYSTEM_MASS.

        The tank weight is already promoted as
        ``Aircraft.Fuel.FUEL_SYSTEM_MASS``, which Aviary's core mass
        buildup already includes. Returning an empty list avoids
        double-counting.
        """
        return []
