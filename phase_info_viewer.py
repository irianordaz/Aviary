"""Visualize an Aviary trajectory definition from a phase_info file.

Renders two side-by-side plots from a phase_info dict:
  Left  – Altitude [ft]  vs Time [min]
  Right – Mach [-]       vs Time [min]

Each phase is drawn as a piecewise-linear segment connecting its initial
and final values.  Error-bar markers on each segment show:
  ●  time_initial_bounds  — the window in which the phase may start
  ▒  time_duration_bounds — the shaded region of possible end times

Handles energy-state, height-energy, and two-DOF phase_info styles.

Usage:
    python phase_info_viewer.py path/to/phase_info.py
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SKIP_KEYS = frozenset({'pre_mission', 'post_mission'})

_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]

_TIME_TO_MIN: dict[str, float] = {
    's': 1 / 60, 'min': 1.0, 'h': 60.0, 'hr': 60.0, 'hour': 60.0,
}
_ALT_TO_FT: dict[str, float] = {
    'ft': 1.0, 'm': 3.280839895, 'km': 3280.839895, 'kft': 1000.0,
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _scalar(v) -> float:
    """Numeric part of (value, unit) tuple or plain number."""
    return float(v[0]) if isinstance(v, (list, tuple)) else float(v)


def _unit(v) -> str:
    if isinstance(v, (list, tuple)) and len(v) >= 2 and isinstance(v[-1], str):
        return v[-1]
    return 'unitless'


def _to_min(value, unit: str) -> float:
    return float(value) * _TIME_TO_MIN.get(unit, 1 / 60)


def _to_ft(value, unit: str) -> float:
    return float(value) * _ALT_TO_FT.get(unit, 1.0)


# ---------------------------------------------------------------------------
# Phase data extraction
# ---------------------------------------------------------------------------

def _parse_phase(name: str, d: dict) -> dict:
    """Return a flat dict of timing, altitude, and Mach info for one phase.

    Extraction priority (later wins for the same field):
      initial_guesses  →  user_options explicit values  →  constant shortcuts
    """
    uo = d.get('user_options', {})
    ig = d.get('initial_guesses', {})

    # ── Time ────────────────────────────────────────────────────────────────
    t_initial = None
    if 'time_initial' in uo:
        v = uo['time_initial']
        t_initial = _to_min(_scalar(v), _unit(v))

    # initial_guesses.time = ([abs_start, duration], unit)
    t_start_ig = t_dur_ig = None
    if 'time' in ig:
        v = ig['time']
        vals, unit = v[0], _unit(v)
        if isinstance(vals, (list, tuple)) and len(vals) == 2:
            t_start_ig = _to_min(vals[0], unit)
            t_dur_ig   = _to_min(vals[1], unit)

    # ── Altitude ───────────────────────────────────────────────────────────
    alt_i_ig = alt_f_ig = None
    if 'altitude' in ig:
        v = ig['altitude']
        vals, unit = v[0], _unit(v)
        if not isinstance(vals, (list, tuple)):
            vals = [vals, vals]
        alt_i_ig = _to_ft(vals[0], unit)
        alt_f_ig = _to_ft(vals[1], unit)

    t_initial_bounds = None
    if 'time_initial_bounds' in uo:
        v = uo['time_initial_bounds']
        b, u = v[0], v[1]
        t_initial_bounds = (_to_min(b[0], u), _to_min(b[1], u))

    t_dur_bounds = None
    if 'time_duration_bounds' in uo:
        v = uo['time_duration_bounds']
        b, u = v[0], v[1]
        t_dur_bounds = (_to_min(b[0], u), _to_min(b[1], u))

    # Nominal duration: from initial_guesses first, then midpoint of bounds
    t_dur_nom = t_dur_ig
    if t_dur_nom is None and t_dur_bounds is not None:
        t_dur_nom = (t_dur_bounds[0] + t_dur_bounds[1]) / 2

    # ── Altitude ────────────────────────────────────────────────────────────
    alt_i = alt_f = None

    # Level 1: initial_guesses.altitude = ([alt_start, alt_end], unit)
    if 'altitude' in ig:
        v = ig['altitude']
        vals, unit = v[0], _unit(v)
        if not isinstance(vals, (list, tuple)):
            vals = [vals, vals]
        alt_i = _to_ft(vals[0], unit)
        alt_f = _to_ft(vals[1], unit)

    # Level 2: explicit user_options values (override guesses)
    if 'altitude_initial' in uo:
        alt_i = _to_ft(_scalar(uo['altitude_initial']), _unit(uo['altitude_initial']))
    if 'altitude_final' in uo:
        alt_f = _to_ft(_scalar(uo['altitude_final']), _unit(uo['altitude_final']))

    # Level 3: constant altitude shortcuts (only if still unknown)
    if alt_i is None:
        for key in ('alt', 'alt_cruise'):
            if key in uo:
                c = _to_ft(_scalar(uo[key]), _unit(uo[key]))
                alt_i = alt_f = c
                break

    # ── Altitude bounds ──────────────────────────────────────────────────────
    alt_bounds = None
    if 'altitude_bounds' in uo:
        v = uo['altitude_bounds']
        b, u = v[0], v[1]
        alt_bounds = (_to_ft(b[0], u), _to_ft(b[1], u))

    # ── Mach ────────────────────────────────────────────────────────────────
    mach_i = mach_f = None
    mach_i_ig = mach_f_ig = None

    # Level 1: initial_guesses.mach = ([mach_start, mach_end], unit)
    if 'mach' in ig:
        v = ig['mach']
        vals, unit = v[0], _unit(v)
        if not isinstance(vals, (list, tuple)):
            vals = [vals, vals]
        mach_i_ig = float(vals[0])
        mach_f_ig = float(vals[1])

    # Level 2: explicit user_options values
    if 'mach_initial' in uo:
        mach_i = _scalar(uo['mach_initial'])
    if 'mach_final' in uo:
        mach_f = _scalar(uo['mach_final'])

    # Level 3: constant Mach shortcuts
    if mach_i is None:
        if 'mach_cruise' in uo:
            v = uo['mach_cruise']
            c = float(v[0]) if isinstance(v, (tuple, list)) else float(v)
            mach_i = mach_f = c
        elif 'mach_target' in uo:
            mach_i = mach_f = float(uo['mach_target'])

    # ── Mach bounds ──────────────────────────────────────────────────────────
    mach_bounds = None
    if 'mach_bounds' in uo:
        v = uo['mach_bounds']
        b, u = v[0], v[1]
        mach_bounds = (float(b[0]), float(b[1]))

    return {
        'name': name,
        # timing
        't_initial':        t_initial,
        't_start_ig':       t_start_ig,
        't_dur_ig':         t_dur_ig,
        't_initial_bounds': t_initial_bounds,
        't_dur_nom':        t_dur_nom,
        't_dur_bounds':     t_dur_bounds,
        # altitude
        'alt_i':      alt_i,
        'alt_f':      alt_f,
        'alt_i_ig':   alt_i_ig,
        'alt_f_ig':   alt_f_ig,
        'alt_bounds': alt_bounds,
        # mach
        'mach_i':      mach_i,
        'mach_f':      mach_f,
        'mach_i_ig':   mach_i_ig,
        'mach_f_ig':   mach_f_ig,
        'mach_bounds': mach_bounds,
    }


def _propagate_final_states(
    phases: list[dict], raw_phase_info: dict,
) -> None:
    """Propagate final states and fill missing initial Mach for the first phase.

    If a phase lacks final altitude/Mach, inherit from the next phase's
    initial values.  If the first phase lacks ``mach_i``, fall back to a
    default takeoff Mach when ``include_takeoff`` is True, or to a default
    low-speed Mach otherwise.
    """
    # ── Propagate next-phase final → current-phase final ─────────────────
    for i in range(len(phases) - 1):
        if phases[i]['alt_f'] is None:
            phases[i]['alt_f'] = phases[i + 1]['alt_i']
        if phases[i]['mach_f'] is None:
            phases[i]['mach_f'] = phases[i + 1]['mach_i']

    # ── First-phase missing mach_i fallback ───────────────────────────────
    if phases and phases[0]['mach_i'] is None:
        pre_mission = raw_phase_info.get('pre_mission', {})
        include_takeoff = pre_mission.get('include_takeoff', False)
        # Typical takeoff clears 35 ft at ~Mach 0.17; otherwise use 0.2.
        phases[0]['mach_i'] = 0.17 if include_takeoff else 0.20


def _chain_times(phases: list[dict]) -> list[dict]:
    """Compute absolute t_start / t_end for each phase by chaining."""
    t_cursor = 0.0
    for p in phases:
        if p['t_start_ig'] is not None:
            # initial_guesses.time provides an absolute start
            t_start = p['t_start_ig']
        elif p['t_initial'] is not None:
            t_start = p['t_initial']
        elif p['t_initial_bounds'] is not None:
            t_start = max(t_cursor, p['t_initial_bounds'][0])
        else:
            t_start = t_cursor

        p['t_start'] = t_start
        dur = p['t_dur_nom'] if p['t_dur_nom'] is not None else 0.0
        p['t_end'] = t_start + dur
        t_cursor = p['t_end']
    return phases


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _draw_profile(
    ax: plt.Axes,
    phases: list[dict],
    key_i: str,
    key_f: str,
    ylabel: str,
    y_formatter=None,
) -> None:
    """Draw a piecewise-linear profile for every phase that has data.

    For each phase with valid y-values:
      - Dashed line + 'x' markers for the initial-guess trajectory
      - Solid line from (t_start, y_i) to (t_end, y_f)
      - ● at (t_start, y_i)  with horizontal error bar for time_initial_bounds
      - Shaded region at (t_end) spanning the full plot height for
        time_duration_bounds
      - Phase name label centred above the segment midpoint
    """
    t_extremes: list[float] = []
    dur_bounds: list[tuple[float, float]] = []

    for i, p in enumerate(phases):
        color = _PALETTE[i % len(_PALETTE)]
        y_i = p[key_i]
        y_f = p[key_f]

        if y_i is None and y_f is None:
            continue
        if y_i is None:
            y_i = y_f
        if y_f is None:
            y_f = y_i

        t_s = p['t_start']
        t_e = p['t_end']

        t_extremes.extend([t_s, t_e])

        # ── Initial-guess trajectory (dashed) ─────────────────────────────
        if p['t_start_ig'] is not None and p['t_dur_ig'] is not None:
            t_ig_start = p['t_start_ig']
            t_ig_end   = t_ig_start + p['t_dur_ig']
            y_ig_i     = p.get(f'{key_i}_ig')
            y_ig_f     = p.get(f'{key_f}_ig')
            if y_ig_i is not None and y_ig_f is not None:
                ax.plot([t_ig_start, t_ig_end], [y_ig_i, y_ig_f],
                        color=color, linewidth=1.2, linestyle='--',
                        alpha=0.45, zorder=6)
                ax.plot([t_ig_start, t_ig_end], [y_ig_i, y_ig_f],
                        'x', color=color, markersize=6,
                        markeredgewidth=1.5, alpha=0.45, zorder=6)

        # Nominal line
        ax.plot([t_s, t_e], [y_i, y_f],
                color=color, linewidth=2.2, solid_capstyle='round', zorder=3)

        # Phase label at segment midpoint
        t_mid = (t_s + t_e) / 2
        y_mid = (y_i + y_f) / 2
        ax.text(
            t_mid, y_mid, p['name'],
            ha='center', va='bottom', fontsize=7.5,
            color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.75),
            zorder=5,
        )

        # ── Start-time bounds ●  ──────────────────────────────────────────
        tib = p['t_initial_bounds']
        if tib is not None:
            xerr_lo = max(t_s - tib[0], 0.0)
            xerr_hi = max(tib[1] - t_s, 0.0)
            ax.errorbar(
                t_s, y_i,
                xerr=[[xerr_lo], [xerr_hi]],
                fmt='o', ms=6, color=color,
                ecolor=color, capsize=8, capthick=1.5, elinewidth=1.5,
                alpha=0.80, zorder=4,
            )
            t_extremes.extend([tib[0], tib[1]])
        else:
            ax.plot(t_s, y_i, 'o', ms=6, color=color, zorder=4, alpha=0.85)

        # ── Duration bounds (shaded) ───────────────────────────────────
        db = p['t_dur_bounds']
        if db is not None:
            t_end_min = t_s + db[0]
            t_end_max = t_s + db[1]
            dur_bounds.append((t_end_min, t_end_max, color))
            t_extremes.extend([t_end_min, t_end_max])
        else:
            ax.plot(t_e, y_f, 's', ms=5, color=color, zorder=4, alpha=0.85)

    # ── Axes styling ─────────────────────────────────────────────────────────
    ax.set_xlabel('Time [min]', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(linestyle='--', linewidth=0.5, alpha=0.55)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if y_formatter is not None:
        ax.yaxis.set_major_formatter(y_formatter)

    # ── Duration-bounds shading ──────────────────────────────────────────
    y_lo, y_hi = ax.get_ylim()
    for t_lo_d, t_hi_d, color in dur_bounds:
        ax.fill_betweenx(
            (y_lo, y_hi),
            t_lo_d, t_hi_d,
            color=color, alpha=0.10, zorder=1,
        )

    if t_extremes:
        t_lo, t_hi = min(t_extremes), max(t_extremes)
        margin = max((t_hi - t_lo) * 0.06, 5.0)
        ax.set_xlim(t_lo - margin, t_hi + margin)


def _build_legend(ax: plt.Axes, phases: list[dict]) -> None:
    """Add a combined legend (phase colours + bound symbols) below *ax*."""
    # ── Phase-colour patches ──────────────────────────────────────────────
    phase_handles = [
        mpatches.Patch(facecolor=_PALETTE[i % len(_PALETTE)],
                       label=p['name'], edgecolor='white')
        for i, p in enumerate(phases)
    ]

    # ── Bound-symbol markers ──────────────────────────────────────────────
    from matplotlib.lines import Line2D

    # initial_guesses: x marker with dashed line
    ig_handle = Line2D(
        [-0.4, 0.4], [0, 0],
        color='#555555', linewidth=1.2, linestyle='--',
        marker='x', markersize=8,
        markeredgecolor='#555555', markeredgewidth=1.5,
        label='initial_guesses',
    )

    # time_initial_bounds: circle with horizontal error bar (drawn as a line)
    initial_bounds_handle = Line2D(
        [-0.4, 0.4], [0, 0],
        color='#555555', linewidth=1.5,
        marker='o', markersize=8,
        markerfacecolor='none', markeredgecolor='#555555', markeredgewidth=1.5,
        label='time_initial_bounds',
    )
    # time_duration_bounds: shaded rectangle
    dur_bounds_handle = mpatches.Patch(
        facecolor='#888888', alpha=0.30,
        label='time_duration_bounds', edgecolor='#888888',
    )

    all_handles = [ig_handle, initial_bounds_handle, dur_bounds_handle] + phase_handles

    ax.legend(
        handles=all_handles,
        fontsize=8, framealpha=0.88, edgecolor='#cccccc',
        loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(len(phases) + 2, 6),
    )


# ---------------------------------------------------------------------------
# Phase info loader
# ---------------------------------------------------------------------------

def load_phase_info(path: str) -> dict:
    """Import *path* as a Python module and return its ``phase_info`` dict."""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f'File not found: {path}')
    spec = importlib.util.spec_from_file_location('_phase_info_mod', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, 'phase_info'):
        raise AttributeError(f"No 'phase_info' variable found in: {path}")
    return mod.phase_info


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_phase_info(
    phase_info: dict,
    *,
    title: str = '',
    phases: Optional[list[str]] = None,
) -> None:
    """Render altitude-vs-time and Mach-vs-time for *phase_info*.

    Parameters
    ----------
    phase_info:
        Aviary phase_info dict (supports energy-state, height-energy, 2DOF).
    title:
        Optional string appended to the figure suptitle.
    phases:
        Optional list of phase names to plot. If ``None``, all phases are
        plotted.
    """
    all_phases = [
        _parse_phase(k, v)
        for k, v in phase_info.items()
        if k not in _SKIP_KEYS and isinstance(v, dict)
    ]

    # Propagate next phase's initial values to current phase's final values
    # so that the trajectory is continuous even when only a subset of phases
    # is plotted.  Also fill first-phase mach_i when not specified.
    _propagate_final_states(all_phases, phase_info)

    if phases is not None:
        phase_names_lower = {p.lower() for p in phases}
        selected = [
            p for p in all_phases if p['name'].lower() in phase_names_lower
        ]
        if not selected:
            print(
                f'Warning: none of the requested phases '
                f'{phases} match any phases in the file. '
                f'Available: {[p["name"] for p in all_phases]}',
                file=sys.stderr,
            )
            return
        phases_to_plot = selected
    else:
        phases_to_plot = all_phases

    phases_to_plot = _chain_times(phases_to_plot)

    fig, (ax_alt, ax_mach) = plt.subplots(
        1, 2, figsize=(16, 6), layout='constrained',
    )
    fig.patch.set_facecolor('#ffffff')
    ax_alt.set_facecolor('#ffffff')
    ax_mach.set_facecolor('#ffffff')

    _draw_profile(
        ax_alt, phases_to_plot, 'alt_i', 'alt_f',
        ylabel='Altitude [ft]',
        y_formatter=mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'),
    )
    ax_alt.set_title('Altitude vs Time', fontsize=11, fontweight='bold', pad=8)
    _build_legend(ax_alt, phases_to_plot)

    _draw_profile(
        ax_mach, phases_to_plot, 'mach_i', 'mach_f',
        ylabel='Mach [-]',
    )
    ax_mach.set_title('Mach Number vs Time', fontsize=11, fontweight='bold', pad=8)
    _build_legend(ax_mach, phases_to_plot)

    suptitle = 'Trajectory Definition'
    if title:
        suptitle += f' — {title}'
    fig.suptitle(suptitle, fontsize=13, fontweight='bold')

    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Visualize an Aviary phase_info trajectory definition. '
            'Produces altitude-vs-time and Mach-vs-time plots side by side.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'phase_info_path',
        metavar='PHASE_INFO',
        help='Path to a Python file containing a phase_info dict',
    )
    parser.add_argument(
        '--phases',
        nargs='+',
        metavar='PHASE',
        help=(
            'One or more phase names to plot. If omitted, all phases are '
            'plotted. Phase names are matched case-insensitively.'
        ),
    )
    args = parser.parse_args()

    pi = load_phase_info(args.phase_info_path)
    title = Path(args.phase_info_path).stem
    plot_phase_info(pi, title=title, phases=args.phases)


if __name__ == '__main__':
    main()
