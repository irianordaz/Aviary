"""Mass fraction dashboard for Aviary / OpenMDAO SQLite case databases.

Reads the last solved case and renders a horizontal bar chart of mass
fractions (each component mass divided by aircraft gross mass). Bars are
sorted largest-to-smallest from top to bottom. Each category of component
is given a distinct colour.

One database → single chart.
Two databases → grouped comparison chart with one pair of bars per item.

Add trajectory phases on the right by using --show-trajectory.

Usage:
    python mass_fraction.py path/to/problem_history.db
    python mass_fraction.py path/to/db_a.db path/to/db_b.db
    python mass_fraction.py path/to/db_a.db path/to/db_b.db --show-trajectory
"""

import argparse
import json
import re
import sqlite3
import zlib
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.gridspec import GridSpecFromSubplotSpec

# ---------------------------------------------------------------------------
# Variable catalog: (promoted_name, display_label, category)
# Entries with zero or missing values are dropped automatically.
# ---------------------------------------------------------------------------

_CAT_FUEL = 'Fuel'
_CAT_PAYLOAD = 'Payload'
_CAT_CREW = 'Crew & Service'
_CAT_STRUCTURE = 'Structure'
_CAT_PROPULSION = 'Propulsion'
_CAT_SYSTEMS = 'Systems'
_CAT_MARGIN = 'Margin'

_VARIABLE_CATALOG: list[tuple[str, str, str]] = [
    # --- Fuel ---
    ('mission:fuel', 'Mission Fuel (Block)', _CAT_FUEL),
    ('mission:reserve_fuel', 'Reserve Fuel', _CAT_FUEL),
    ('mission:reserve_fuel_additional', 'Additional Reserve', _CAT_FUEL),
    # --- Payload ---
    ('aircraft:crew_and_payload:passenger_mass_total', 'Passengers', _CAT_PAYLOAD),
    ('aircraft:crew_and_payload:baggage_mass', 'Baggage', _CAT_PAYLOAD),
    ('aircraft:crew_and_payload:cargo_mass', 'Cargo', _CAT_PAYLOAD),
    ('aircraft:crew_and_payload:misc_cargo', 'Misc Cargo', _CAT_PAYLOAD),
    ('aircraft:crew_and_payload:wing_cargo', 'Wing Cargo', _CAT_PAYLOAD),
    # --- Crew & Service ---
    ('aircraft:crew_and_payload:flight_crew_mass', 'Flight Crew', _CAT_CREW),
    ('aircraft:crew_and_payload:cabin_crew_mass', 'Cabin Crew', _CAT_CREW),
    ('aircraft:crew_and_payload:passenger_service_mass', 'Passenger Service', _CAT_CREW),
    ('aircraft:crew_and_payload:cargo_container_mass', 'Cargo Containers', _CAT_CREW),
    ('aircraft:fuel:unusable_fuel_mass', 'Unusable Fuel', _CAT_CREW),
    ('aircraft:propulsion:total_engine_oil_mass', 'Engine Oil', _CAT_CREW),
    # --- Structure ---
    ('aircraft:fuselage:mass', 'Fuselage', _CAT_STRUCTURE),
    ('aircraft:wing:mass', 'Wing', _CAT_STRUCTURE),
    ('aircraft:landing_gear:main_gear_mass', 'Main Landing Gear', _CAT_STRUCTURE),
    ('aircraft:landing_gear:nose_gear_mass', 'Nose Landing Gear', _CAT_STRUCTURE),
    ('aircraft:horizontal_tail:mass', 'Horizontal Tail', _CAT_STRUCTURE),
    ('aircraft:vertical_tail:mass', 'Vertical Tail', _CAT_STRUCTURE),
    ('aircraft:canard:mass', 'Canard', _CAT_STRUCTURE),
    ('aircraft:fins:mass', 'Fins', _CAT_STRUCTURE),
    ('aircraft:nacelle:mass', 'Nacelle', _CAT_STRUCTURE),
    ('aircraft:paint:mass', 'Paint', _CAT_STRUCTURE),
    # --- Propulsion ---
    ('aircraft:propulsion:total_engine_mass', 'Engines', _CAT_PROPULSION),
    ('aircraft:fuel:fuel_system_mass', 'Fuel System', _CAT_PROPULSION),
    ('aircraft:propulsion:total_thrust_reversers_mass', 'Thrust Reversers', _CAT_PROPULSION),
    ('aircraft:propulsion:total_engine_controls_mass', 'Engine Controls', _CAT_PROPULSION),
    ('aircraft:propulsion:total_starter_mass', 'Engine Starters', _CAT_PROPULSION),
    ('aircraft:battery:mass', 'Battery', _CAT_PROPULSION),
    # --- Systems ---
    ('aircraft:furnishings:mass', 'Furnishings', _CAT_SYSTEMS),
    ('aircraft:wing:surface_control_mass', 'Surface Controls', _CAT_SYSTEMS),
    ('aircraft:electrical:mass', 'Electrical', _CAT_SYSTEMS),
    ('aircraft:avionics:mass', 'Avionics', _CAT_SYSTEMS),
    ('aircraft:air_conditioning:mass', 'Air Conditioning', _CAT_SYSTEMS),
    ('aircraft:apu:mass', 'APU', _CAT_SYSTEMS),
    ('aircraft:instruments:mass', 'Instruments', _CAT_SYSTEMS),
    ('aircraft:hydraulics:mass', 'Hydraulics', _CAT_SYSTEMS),
    ('aircraft:anti_icing:mass', 'Anti-Icing', _CAT_SYSTEMS),
    # --- Margin ---
    ('aircraft:design:empty_mass_margin', 'Empty Mass Margin', _CAT_MARGIN),
]

_GROSS_MASS_PROM = 'aircraft:design:gross_mass'

_CATEGORY_COLOR: dict[str, str] = {
    _CAT_FUEL: '#E8871A',
    _CAT_PAYLOAD: '#2E9E4F',
    _CAT_CREW: '#1A9E8E',
    _CAT_STRUCTURE: '#2060C8',
    _CAT_PROPULSION: '#C82020',
    _CAT_SYSTEMS: '#8030B0',
    _CAT_MARGIN: '#6E8090',
}

_MAX_BLEND = 0.65
_BAR_SPACING = 0.8
_PAIR_BAR_SPACING = 1.0

# Distinct colors for trajectory phases (tab10 palette)
_PHASE_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]

# Timeseries discovery: traj.phases.{phase}.timeseries.{var}
_TRAJ_VAR_RE = re.compile(r'traj\.(?:phases\.)?([^.]+)\.timeseries\.(.+)')

_ALT_TO_FT: dict[str, float] = {
    'ft': 1.0,
    'm': 3.280839895,
    'km': 3280.839895,
    'kft': 1000.0,
}
_TIME_TO_MIN: dict[str, float] = {
    's': 1 / 60,
    'min': 1.0,
    'h': 60.0,
    'hr': 60.0,
    'hour': 60.0,
}


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def _decompress_blob(blob) -> dict:
    if blob is None:
        return {}
    if isinstance(blob, (bytes, bytearray)):
        return json.loads(zlib.decompress(blob))
    return json.loads(blob)


def _scalar(value) -> float:
    if isinstance(value, (list, tuple)):
        return float(value[0])
    return float(value)


def load_mass_data(db_path: str) -> dict:
    """Read mass variables from the last solved case in *db_path*.

    Returns a dict with:
      - 'gross_mass': float in lbm
      - 'entries': list of (label, category, mass_lbm) tuples,
        only for variables with positive values
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute('SELECT abs2prom, abs2meta FROM metadata LIMIT 1')
        row = cur.fetchone()
        if row is None:
            raise ValueError(f'No metadata found in: {db_path}')

        abs2prom_all = _decompress_blob(row[0])
        abs2meta = _decompress_blob(row[1])
        prom_out = abs2prom_all.get('output', {})

        cur.execute(
            'SELECT outputs FROM problem_cases ORDER BY counter DESC LIMIT 1'
        )
        result = cur.fetchone()
        if result is None:
            raise ValueError(f'No solved cases found in: {db_path}')

        outputs = json.loads(result[0]) if result[0] else {}
    finally:
        conn.close()

    prom_to_value: dict[str, float] = {}
    for abs_name, raw_val in outputs.items():
        prom = prom_out.get(abs_name)
        if prom is None:
            continue
        units = abs2meta.get(abs_name, {}).get('units', '')
        if units != 'lbm':
            continue
        if prom not in prom_to_value:
            prom_to_value[prom] = _scalar(raw_val)

    gross_mass = prom_to_value.get(_GROSS_MASS_PROM)
    if gross_mass is None:
        raise ValueError(
            f"Could not find '{_GROSS_MASS_PROM}' in outputs of: {db_path}"
        )

    entries: list[tuple[str, str, float]] = []
    for prom, label, category in _VARIABLE_CATALOG:
        mass = prom_to_value.get(prom)
        if mass is not None and mass > 0.0:
            entries.append((label, category, mass))

    return {'gross_mass': gross_mass, 'entries': entries}


def load_trajectory_data(db_path: str) -> dict:
    """Load altitude vs time timeseries per phase from *db_path*.

    Returns
    -------
        dict mapping phase_name → {'time_min': ndarray, 'altitude_ft': ndarray}.
        Empty dict if no timeseries data is found.
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute('SELECT abs2meta FROM metadata LIMIT 1')
        row = cur.fetchone()
        abs2meta = _decompress_blob(row[0]) if row else {}

        outputs: dict = {}
        for table in ('problem_cases', 'driver_iterations'):
            try:
                cur.execute(
                    f'SELECT outputs FROM {table} ORDER BY counter DESC LIMIT 1'
                )
                res = cur.fetchone()
                if res and res[0]:
                    candidate = json.loads(res[0])
                    if any('timeseries' in k for k in candidate):
                        outputs = candidate
                        break
            except sqlite3.OperationalError:
                continue
    finally:
        conn.close()

    if not outputs:
        return {}

    phase_keys: dict[str, dict[str, str]] = {}
    for abs_name in outputs:
        m = _TRAJ_VAR_RE.match(abs_name)
        if not m:
            continue
        phase, var = m.group(1), m.group(2).lower()
        if phase not in phase_keys:
            phase_keys[phase] = {}
        if 'altitude' in var and 'alt' not in phase_keys[phase]:
            phase_keys[phase]['alt'] = abs_name
        elif var == 'time' and 'time' not in phase_keys[phase]:
            phase_keys[phase]['time'] = abs_name
        elif var == 'time_phase' and 'time' not in phase_keys[phase]:
            phase_keys[phase]['time'] = abs_name

    result: dict = {}
    for phase, keys in phase_keys.items():
        if 'alt' not in keys or 'time' not in keys:
            continue
        alt_raw = np.asarray(outputs[keys['alt']], dtype=float).flatten()
        time_raw = np.asarray(outputs[keys['time']], dtype=float).flatten()
        if alt_raw.size == 0 or time_raw.size == 0:
            continue

        alt_units = abs2meta.get(keys['alt'], {}).get('units', 'ft')
        time_units = abs2meta.get(keys['time'], {}).get('units', 's')

        alt_ft = alt_raw * _ALT_TO_FT.get(alt_units, 1.0)
        time_min = time_raw * _TIME_TO_MIN.get(time_units, 1 / 60)

        result[phase] = {'time_min': time_min, 'altitude_ft': alt_ft}

    return result


# ---------------------------------------------------------------------------
# Chart data helpers
# ---------------------------------------------------------------------------


def _label_for_path(path: str) -> str:
    """Return a short display label derived from the database path."""
    p = Path(path)
    folder = p.parent.name
    label = folder if folder not in ('.', '') else p.stem
    if label.endswith('_out'):
        label = label[:-4]
    return label


def _assign_colors(entries: list[tuple[str, str, float]]) -> list[tuple]:
    """Return a colour per entry based on within-category mass rank."""
    cat_sorted: dict[str, list[float]] = {}
    for _, cat, mass in entries:
        cat_sorted.setdefault(cat, []).append(mass)
    for cat in cat_sorted:
        cat_sorted[cat].sort(reverse=True)

    colors = []
    for _, cat, mass in entries:
        ranked = cat_sorted[cat]
        n = len(ranked)
        rank = ranked.index(mass)
        blend = 0.0 if n == 1 else rank / (n - 1) * _MAX_BLEND
        base = _CATEGORY_COLOR.get(cat, '#808080')
        colors.append(_apply_blend(base, blend))
    return colors


def _sorted_entries(entries: list[tuple[str, str, float]]) -> list[tuple[str, str, float]]:
    return sorted(entries, key=lambda x: x[2], reverse=True)


def _apply_blend(hex_color: str, blend: float) -> tuple:
    """Blend *hex_color* toward white by *blend* ∈ [0, 1] (0 = base color)."""
    r = int(hex_color[1:3], 16) / 255
    g = int(hex_color[3:5], 16) / 255
    b = int(hex_color[5:7], 16) / 255
    return (r + (1 - r) * blend, g + (1 - g) * blend, b + (1 - b) * blend)


def _sorted_phases(traj_primary: dict, traj_compare: dict | None) -> list[str]:
    """Return all phase names sorted by start time."""
    all_phases = set(traj_primary) | (set(traj_compare) if traj_compare else set())
    return sorted(
        all_phases,
        key=lambda p: float(
            (traj_primary.get(p) or (traj_compare or {}).get(p, {}))
            .get('time_min', [np.inf])[0]
        ),
    )


# ---------------------------------------------------------------------------
# Mass-fraction single-database plot
# ---------------------------------------------------------------------------


def _plot_single(
    ax: plt.Axes,
    mass_data: dict,
    gross: float,
    *,
    bar_scale: float = 1.0,
    spacing_scale: float = 1.0,
    label_font_scale: float = 1.0,
    legend_font_scale: float = 1.0,
):
    entries = _sorted_entries(mass_data['entries'])
    colors = _assign_colors(entries)

    labels = [e[0] for e in entries]
    fractions = [e[2] / gross for e in entries]
    masses = [e[2] for e in entries]
    categories = [e[1] for e in entries]

    n = len(labels)
    spacing = _BAR_SPACING * spacing_scale
    bar_height = 0.58 * bar_scale
    y_pos = np.arange(n) * spacing

    bars = ax.barh(
        y_pos, fractions, height=bar_height,
        color=colors, edgecolor='white', linewidth=0.7,
    )
    ax.invert_yaxis()

    annot_fs = 8.0 * label_font_scale
    for bar, frac, mass in zip(bars, fractions, masses):
        ax.text(
            bar.get_width() + 0.0015,
            bar.get_y() + bar.get_height() / 2,
            f'{frac * 100:.1f}%  ({mass:,.0f} lbm)',
            va='center', ha='left', fontsize=annot_fs, color='#2a2a2a',
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.0 * label_font_scale)
    ax.tick_params(axis='y', length=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.55)

    max_frac = max(fractions) if fractions else 0.0
    ax.set_xlim(0, max_frac * 1.32)

    seen: set[str] = set()
    handles = []
    for cat in categories:
        if cat not in seen:
            handles.append(
                mpatches.Patch(facecolor=_CATEGORY_COLOR[cat], label=cat, edgecolor='white')
            )
            seen.add(cat)
    ax.legend(handles=handles, loc='lower right',
              fontsize=8.5 * legend_font_scale, framealpha=0.85, edgecolor='#cccccc')

    return max_frac


# ---------------------------------------------------------------------------
# Mass-fraction two-database comparison plot
# ---------------------------------------------------------------------------


def _plot_comparison(
    ax: plt.Axes,
    data_a: dict,
    label_a: str,
    data_b: dict,
    label_b: str,
    *,
    bar_scale: float = 1.0,
    spacing_scale: float = 1.0,
    pair_spacing_scale: float = 1.0,
    label_font_scale: float = 1.0,
    legend_font_scale: float = 1.0,
):
    gross_a = data_a['gross_mass']
    gross_b = data_b['gross_mass']

    # Use the larger gross mass as the common denominator for consistent horizontal scaling
    shared_gross_mass = max(gross_a, gross_b)

    catalog_order = [label for _, label, _ in _VARIABLE_CATALOG]
    entry_map_a = {lbl: (cat, mass) for lbl, cat, mass in data_a['entries']}
    entry_map_b = {lbl: (cat, mass) for lbl, cat, mass in data_b['entries']}

    all_labels = [lbl for lbl in catalog_order if lbl in entry_map_a or lbl in entry_map_b]

    def sort_key(lbl):
        mass_a = entry_map_a.get(lbl, (None, 0.0))[1]
        mass_b = entry_map_b.get(lbl, (None, 0.0))[1]
        return max(mass_a, mass_b)

    all_labels = sorted(all_labels, key=sort_key, reverse=True)

    n = len(all_labels)
    offset = 0.22 * pair_spacing_scale
    bar_height = 0.38 * bar_scale
    spacing = _PAIR_BAR_SPACING * spacing_scale
    y_pos = np.arange(n) * spacing

    unified_entries = []
    for lbl in all_labels:
        cat = (entry_map_a.get(lbl) or entry_map_b.get(lbl))[0]
        mass_a = entry_map_a.get(lbl, (None, 0.0))[1]
        mass_b = entry_map_b.get(lbl, (None, 0.0))[1]
        unified_entries.append((lbl, cat, max(mass_a, mass_b)))

    base_colors = _assign_colors(unified_entries)

    colors_a, colors_b = [], []
    fracs_a, fracs_b = [], []
    masses_a, masses_b = [], []
    categories = []

    for (lbl, cat, _), base_color in zip(unified_entries, base_colors):
        categories.append(cat)
        colors_a.append(base_color)
        r, g, b = base_color
        colors_b.append((r + (1 - r) * 0.30, g + (1 - g) * 0.30, b + (1 - b) * 0.30))
        mass_a = entry_map_a.get(lbl, (None, 0.0))[1]
        mass_b = entry_map_b.get(lbl, (None, 0.0))[1]
        # Use shared gross mass for consistent horizontal scale
        fracs_a.append(mass_a / shared_gross_mass)
        fracs_b.append(mass_b / shared_gross_mass)
        masses_a.append(mass_a)
        masses_b.append(mass_b)

    bars_a = ax.barh(
        y_pos + offset, fracs_a, height=bar_height,
        color=colors_a, edgecolor='white', linewidth=0.6,
    )
    bars_b = ax.barh(
        y_pos - offset, fracs_b, height=bar_height,
        color=colors_b, edgecolor='white', linewidth=0.6, hatch='//', alpha=0.95,
    )
    ax.invert_yaxis()

    annot_fs = 7.2 * label_font_scale
    for bar, frac, mass in zip(bars_a, fracs_a, masses_a):
        if mass > 0:
            ax.text(
                bar.get_width() + 0.0012, bar.get_y() + bar.get_height() / 2,
                f'{frac * 100:.1f}%  ({mass:,.0f})',
                va='center', ha='left', fontsize=annot_fs, color='#222222',
            )
    for bar, frac, mass in zip(bars_b, fracs_b, masses_b):
        if mass > 0:
            ax.text(
                bar.get_width() + 0.0012, bar.get_y() + bar.get_height() / 2,
                f'{frac * 100:.1f}%  ({mass:,.0f})',
                va='center', ha='left', fontsize=annot_fs, color='#444444',
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_labels, fontsize=8.5 * label_font_scale)
    ax.tick_params(axis='y', length=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.55)

    max_frac = max(max(fracs_a), max(fracs_b)) if fracs_a else 0.0
    ax.set_xlim(0, max_frac * 1.38)

    cat_handles = []
    seen: set[str] = set()
    for cat in categories:
        if cat not in seen:
            cat_handles.append(
                mpatches.Patch(facecolor=_CATEGORY_COLOR[cat], label=cat, edgecolor='white')
            )
            seen.add(cat)

    db_handles = [
        mpatches.Patch(facecolor='#aaaaaa', label=label_a, edgecolor='white'),
        mpatches.Patch(facecolor='#cccccc', label=label_b, edgecolor='white', hatch='//'),
    ]
    ax.legend(
        handles=cat_handles + [mpatches.Patch(visible=False)] + db_handles,
        loc='lower right', fontsize=8.0 * legend_font_scale,
        framealpha=0.88, edgecolor='#cccccc',
    )

    return max_frac


# ---------------------------------------------------------------------------
# Altitude vs time — one subplot per phase, stacked vertically
# ---------------------------------------------------------------------------


def _plot_altitude_panels(
    axes: list[plt.Axes],
    phases: list[str],
    traj_primary: dict,
    traj_compare: dict | None = None,
    *,
    label_primary: str = 'A',
    label_compare: str = 'B',
    label_font_scale: float = 1.0,
    legend_font_scale: float = 1.0,
):
    """Plot one altitude-vs-time subplot per trajectory phase, stacked vertically."""
    if not phases:
        ax = axes[0]
        ax.text(0.5, 0.5, 'No trajectory data found',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, color='#888888')
        ax.set_xlabel('Time (min)', fontsize=9 * label_font_scale)
        ax.set_ylabel('Altitude (ft)', fontsize=9 * label_font_scale)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return

    color_map = {p: _PHASE_PALETTE[i % len(_PHASE_PALETTE)] for i, p in enumerate(phases)}
    fs_label = 8 * label_font_scale
    fs_legend = 8 * legend_font_scale

    for i, (ax, phase) in enumerate(zip(axes, phases)):
        color = color_map[phase]
        has_primary = phase in traj_primary
        has_compare = bool(traj_compare and phase in traj_compare)

        if has_primary:
            d = traj_primary[phase]
            ax.plot(d['time_min'], d['altitude_ft'],
                    color=color, linewidth=1.8,
                    label=label_primary if traj_compare else None)
        if has_compare:
            d = traj_compare[phase]
            ax.plot(d['time_min'], d['altitude_ft'],
                    color=color, linewidth=1.8, linestyle='--',
                    label=label_compare)

        ax.set_title(phase, fontsize=fs_label, pad=3, loc='left',
                     color=color, fontweight='bold')
        ax.set_ylabel('Altitude (ft)', fontsize=fs_label)
        ax.set_xlabel('Time (min)', fontsize=fs_label)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        ax.tick_params(labelsize=fs_label * 0.9)
        ax.grid(linestyle='--', linewidth=0.5, alpha=0.55)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add legend for comparison plots
        if traj_compare:
            ax.legend(fontsize=fs_legend, framealpha=0.85,
                      edgecolor='#cccccc', loc='best')


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def plot_dashboard(
    db_paths: list[str],
    *,
    show_trajectory: bool = False,
    bar_scale: float = 1.0,
    spacing_scale: float = 1.0,
    pair_spacing_scale: float = 1.0,
    title_font_scale: float = 1.0,
    label_font_scale: float = 1.0,
    legend_font_scale: float = 1.0,
):
    """Load data and display the mass fraction dashboard.

    Args:
        db_paths: one or two paths to SQLite case databases.
        show_trajectory: if True, show trajectory phase altitude plots on the right.
        bar_scale: multiplier for bar height (default 1.0).
        spacing_scale: multiplier for vertical spacing between mass items.
        pair_spacing_scale: multiplier for spacing between the two solution
            bars of the same mass item in comparison mode (default 1.0).
        title_font_scale: multiplier for chart title and axis label font sizes.
        label_font_scale: multiplier for tick labels and bar annotation fonts.
        legend_font_scale: multiplier for legend font size.
    """
    shared_kw = dict(
        bar_scale=bar_scale,
        spacing_scale=spacing_scale,
        label_font_scale=label_font_scale,
        legend_font_scale=legend_font_scale,
    )

    if len(db_paths) == 1:
        data = load_mass_data(db_paths[0])
        traj = load_trajectory_data(db_paths[0]) if show_trajectory else None
        gross = data['gross_mass']
        n = len(data['entries'])
        phases = _sorted_phases(traj, None) if traj else []
        n_alt = max(len(phases), 1)

        if show_trajectory:
            fig_width = 22
            width_ratios = [1.5, 1]
            fig_height = max(6, n * 0.50 * _BAR_SPACING * spacing_scale + 2.2)
        else:
            fig_width = 12
            width_ratios = [1]
            fig_height = max(6, n * 0.50 * _BAR_SPACING * spacing_scale + 2.2)

        fig = plt.figure(figsize=(fig_width, fig_height), layout='constrained')
        fig.patch.set_facecolor('#f5f7fa')
        gs_main = fig.add_gridspec(1, 1 + int(show_trajectory), width_ratios=width_ratios)
        ax_mass = fig.add_subplot(gs_main[0])
        ax_mass.set_facecolor('#f5f7fa')

        if show_trajectory:
            gs_alt = GridSpecFromSubplotSpec(n_alt, 1, subplot_spec=gs_main[1], hspace=0.65)
            ax_alts = [fig.add_subplot(gs_alt[i]) for i in range(n_alt)]
            for ax in ax_alts:
                ax.set_facecolor('#f5f7fa')

        _plot_single(ax_mass, data, gross, **shared_kw)
        ax_mass.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax_mass.set_xlabel(
            'Fraction of Gross Mass', fontsize=10 * title_font_scale, labelpad=6
        )
        ax_mass.set_title('Mass Fractions', fontsize=10 * title_font_scale, pad=8)

        if show_trajectory and traj:
            _plot_altitude_panels(
                ax_alts, phases, traj,
                label_font_scale=label_font_scale,
                legend_font_scale=legend_font_scale,
            )

        db_name = Path(db_paths[0]).name
        fig.suptitle(
            f'Mass Fractions — {db_name}\nGross Mass: {gross:,.0f} lbm',
            fontsize=12 * title_font_scale, fontweight='bold', y=0.99,
        )

    else:
        data_a = load_mass_data(db_paths[0])
        data_b = load_mass_data(db_paths[1])
        traj_a = load_trajectory_data(db_paths[0]) if show_trajectory else None
        traj_b = load_trajectory_data(db_paths[1]) if show_trajectory else None
        label_a = _label_for_path(db_paths[0])
        label_b = _label_for_path(db_paths[1])

        n_items = len({lbl for lbl, _, _ in data_a['entries'] + data_b['entries']})
        phases = _sorted_phases(traj_a, traj_b) if traj_a or traj_b else []
        n_alt = max(len(phases), 1)

        if show_trajectory:
            fig_width = 24
            width_ratios = [1.5, 1]
            fig_height = max(7, n_items * 0.62 * _PAIR_BAR_SPACING * spacing_scale + 2.5)
        else:
            fig_width = 13
            width_ratios = [1]
            fig_height = max(7, n_items * 0.62 * _PAIR_BAR_SPACING * spacing_scale + 2.5)

        fig = plt.figure(figsize=(fig_width, fig_height))
        fig.patch.set_facecolor('#f5f7fa')
        gs_main = fig.add_gridspec(1, 1 + int(show_trajectory), width_ratios=width_ratios)
        ax_mass = fig.add_subplot(gs_main[0])
        ax_mass.set_facecolor('#f5f7fa')

        if show_trajectory and (traj_a or traj_b):
            gs_alt = GridSpecFromSubplotSpec(n_alt, 1, subplot_spec=gs_main[1], hspace=0.65)
            ax_alts = [fig.add_subplot(gs_alt[i]) for i in range(n_alt)]
            for ax in ax_alts:
                ax.set_facecolor('#f5f7fa')

        _plot_comparison(
            ax_mass, data_a, label_a, data_b, label_b,
            **shared_kw, pair_spacing_scale=pair_spacing_scale,
        )
        ax_mass.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax_mass.set_xlabel(
            'Fraction of Gross Mass (shared scale)', fontsize=10 * title_font_scale, labelpad=6
        )
        ax_mass.set_title('Mass Fractions', fontsize=10 * title_font_scale, pad=8)

        if show_trajectory and (traj_a or traj_b):
            _plot_altitude_panels(
                ax_alts, phases, traj_a, traj_b,
                label_primary=label_a, label_compare=label_b,
                label_font_scale=label_font_scale,
                legend_font_scale=legend_font_scale,
            )

        gross_a = data_a['gross_mass']
        gross_b = data_b['gross_mass']
        fig.suptitle(
            f'Mass Fraction Comparison\n'
            f'{label_a}: {gross_a:,.0f} lbm     {label_b}: {gross_b:,.0f} lbm',
            fontsize=12 * title_font_scale, fontweight='bold', y=0.99,
        )

    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """Entry point for the mass fraction dashboard CLI."""
    parser = argparse.ArgumentParser(
        description=(
            'Mass fraction dashboard for Aviary / OpenMDAO SQLite databases. '
            'Pass one database for a single chart or two for a comparison. '
            'Use --show-trajectory to display trajectory phase altitude plots.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'database', nargs='+', metavar='DB',
        help='Path(s) to SQLite database file(s) (1 or 2)',
    )
    parser.add_argument(
        '--show-trajectory', action='store_true',
        help='Show trajectory phase altitude plots on the right side',
    )
    parser.add_argument(
        '--bar-scale', type=float, default=1.0, metavar='S',
        help='Scale multiplier for bar height',
    )
    parser.add_argument(
        '--spacing-scale', type=float, default=1.0, metavar='S',
        help='Scale multiplier for vertical spacing between bars',
    )
    parser.add_argument(
        '--title-font-scale', type=float, default=1.0, metavar='S',
        help='Scale multiplier for chart title and axis label font size',
    )
    parser.add_argument(
        '--pair-spacing-scale', type=float, default=1.0, metavar='S',
        help='Scale multiplier for spacing between the two solution bars '
        'of the same mass item (comparison mode only)',
    )
    parser.add_argument(
        '--label-font-scale', type=float, default=1.0, metavar='S',
        help='Scale multiplier for tick labels and bar annotations',
    )
    parser.add_argument(
        '--legend-font-scale', type=float, default=1.0, metavar='S',
        help='Scale multiplier for legend font size',
    )
    args = parser.parse_args()
    if len(args.database) > 2:
        parser.error('At most two database paths are supported.')
    plot_dashboard(
        args.database,
        show_trajectory=args.show_trajectory,
        bar_scale=args.bar_scale,
        spacing_scale=args.spacing_scale,
        pair_spacing_scale=args.pair_spacing_scale,
        title_font_scale=args.title_font_scale,
        label_font_scale=args.label_font_scale,
        legend_font_scale=args.legend_font_scale,
    )


if __name__ == '__main__':
    main()
