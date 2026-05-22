"""Mass fraction dashboard for Aviary / OpenMDAO SQLite case databases.

Reads the last solved case and renders a horizontal bar chart of mass
fractions (each component mass divided by aircraft gross mass). Bars are
sorted largest-to-smallest from top to bottom. Each category of component
is given a distinct colour.

One database → single chart.
Two databases → grouped comparison chart with one pair of bars per item.

Usage:
    python mass_fraction_dashboard.py path/to/problem_history.db
    python mass_fraction_dashboard.py path/to/db_a.db path/to/db_b.db
"""

import argparse
import json
import sqlite3
import zlib
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


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
    # --- Fuel (individual mission segments, not the block_fuel aggregate) ---
    ('mission:fuel',
     'Mission Fuel Burned', _CAT_FUEL),
    ('mission:taxi:fuel_taxi_out',
     'Taxi Out Fuel', _CAT_FUEL),
    ('mission:takeoff:fuel',
     'Takeoff Fuel', _CAT_FUEL),
    ('mission:taxi:fuel_taxi_in',
     'Taxi In Fuel', _CAT_FUEL),
    ('mission:reserve_fuel',
     'Reserve Fuel', _CAT_FUEL),
    ('mission:reserve_fuel_additional',
     'Additional Reserve', _CAT_FUEL),
    ('aircraft:fuel:unusable_fuel_mass',
     'Unusable Fuel', _CAT_FUEL),
    # --- Payload ---
    ('aircraft:crew_and_payload:passenger_mass_total',
     'Passengers', _CAT_PAYLOAD),
    ('aircraft:crew_and_payload:baggage_mass',
     'Baggage', _CAT_PAYLOAD),
    ('aircraft:crew_and_payload:cargo_mass',
     'Cargo', _CAT_PAYLOAD),
    ('aircraft:crew_and_payload:misc_cargo',
     'Misc Cargo', _CAT_PAYLOAD),
    ('aircraft:crew_and_payload:wing_cargo',
     'Wing Cargo', _CAT_PAYLOAD),
    # --- Crew & Service ---
    ('aircraft:crew_and_payload:flight_crew_mass',
     'Flight Crew', _CAT_CREW),
    ('aircraft:crew_and_payload:cabin_crew_mass',
     'Cabin Crew', _CAT_CREW),
    ('aircraft:crew_and_payload:passenger_service_mass',
     'Passenger Service', _CAT_CREW),
    ('aircraft:crew_and_payload:cargo_container_mass',
     'Cargo Containers', _CAT_CREW),
    # --- Structure ---
    ('aircraft:fuselage:mass',
     'Fuselage', _CAT_STRUCTURE),
    ('aircraft:wing:mass',
     'Wing', _CAT_STRUCTURE),
    ('aircraft:landing_gear:main_gear_mass',
     'Main Landing Gear', _CAT_STRUCTURE),
    ('aircraft:landing_gear:nose_gear_mass',
     'Nose Landing Gear', _CAT_STRUCTURE),
    ('aircraft:horizontal_tail:mass',
     'Horizontal Tail', _CAT_STRUCTURE),
    ('aircraft:vertical_tail:mass',
     'Vertical Tail', _CAT_STRUCTURE),
    ('aircraft:canard:mass',
     'Canard', _CAT_STRUCTURE),
    ('aircraft:fins:mass',
     'Fins', _CAT_STRUCTURE),
    ('aircraft:nacelle:mass',
     'Nacelle', _CAT_STRUCTURE),
    ('aircraft:paint:mass',
     'Paint', _CAT_STRUCTURE),
    # --- Propulsion ---
    ('aircraft:propulsion:total_engine_mass',
     'Engines', _CAT_PROPULSION),
    ('aircraft:fuel:fuel_system_mass',
     'Fuel System', _CAT_PROPULSION),
    ('aircraft:propulsion:total_thrust_reversers_mass',
     'Thrust Reversers', _CAT_PROPULSION),
    ('aircraft:propulsion:total_engine_controls_mass',
     'Engine Controls', _CAT_PROPULSION),
    ('aircraft:propulsion:total_starter_mass',
     'Engine Starters', _CAT_PROPULSION),
    ('aircraft:battery:mass',
     'Battery', _CAT_PROPULSION),
    # --- Systems ---
    ('aircraft:furnishings:mass',
     'Furnishings', _CAT_SYSTEMS),
    ('aircraft:wing:surface_control_mass',
     'Surface Controls', _CAT_SYSTEMS),
    ('aircraft:electrical:mass',
     'Electrical', _CAT_SYSTEMS),
    ('aircraft:avionics:mass',
     'Avionics', _CAT_SYSTEMS),
    ('aircraft:air_conditioning:mass',
     'Air Conditioning', _CAT_SYSTEMS),
    ('aircraft:apu:mass',
     'APU', _CAT_SYSTEMS),
    ('aircraft:instruments:mass',
     'Instruments', _CAT_SYSTEMS),
    ('aircraft:hydraulics:mass',
     'Hydraulics', _CAT_SYSTEMS),
    ('aircraft:anti_icing:mass',
     'Anti-Icing', _CAT_SYSTEMS),
    # --- Margin ---
    ('aircraft:design:empty_mass_margin',
     'Empty Mass Margin', _CAT_MARGIN),
]

_GROSS_MASS_PROM = 'aircraft:design:gross_mass'

_CATEGORY_COLOR: dict[str, str] = {
    _CAT_FUEL:      '#E8871A',
    _CAT_PAYLOAD:   '#2E9E4F',
    _CAT_CREW:      '#1A9E8E',
    _CAT_STRUCTURE: '#2060C8',
    _CAT_PROPULSION:'#C82020',
    _CAT_SYSTEMS:   '#8030B0',
    _CAT_MARGIN:    '#6E8090',
}


def _shade(hex_color: str, step: int, total: int) -> tuple:
    """Lighten *hex_color* for within-category shading (step 0 = darkest)."""
    r = int(hex_color[1:3], 16) / 255
    g = int(hex_color[3:5], 16) / 255
    b = int(hex_color[5:7], 16) / 255
    blend = 0.0 if total <= 1 else step / (total - 1) * 0.40
    return (r + (1 - r) * blend, g + (1 - g) * blend, b + (1 - b) * blend)


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

    Returns:
        A dict with:
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


# ---------------------------------------------------------------------------
# Chart data helpers
# ---------------------------------------------------------------------------

def _label_for_path(path: str) -> str:
    """Return a short display label derived from the database path."""
    p = Path(path)
    folder = p.parent.name
    return folder if folder not in ('.', '') else p.stem


def _assign_colors(
    entries: list[tuple[str, str, float]],
) -> list[tuple]:
    """Return a colour per entry using within-category shading."""
    cat_counts: dict[str, int] = {}
    for _, cat, _ in entries:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    cat_step: dict[str, int] = {}
    colors = []
    for _, cat, _ in entries:
        step = cat_step.get(cat, 0)
        base = _CATEGORY_COLOR.get(cat, '#808080')
        colors.append(_shade(base, step, cat_counts[cat]))
        cat_step[cat] = step + 1
    return colors


def _sorted_entries(
    entries: list[tuple[str, str, float]],
) -> list[tuple[str, str, float]]:
    return sorted(entries, key=lambda x: x[2], reverse=True)


# ---------------------------------------------------------------------------
# Single-database plot
# ---------------------------------------------------------------------------

def _plot_single(ax: plt.Axes, mass_data: dict, gross: float):
    entries = _sorted_entries(mass_data['entries'])
    colors = _assign_colors(entries)

    labels = [e[0] for e in entries]
    fractions = [e[2] / gross for e in entries]
    masses = [e[2] for e in entries]
    categories = [e[1] for e in entries]

    n = len(labels)
    y_pos = np.arange(n)
    bar_height = 0.58

    bars = ax.barh(
        y_pos, fractions,
        height=bar_height,
        color=colors,
        edgecolor='white',
        linewidth=0.7,
    )
    ax.invert_yaxis()

    for bar, frac, mass in zip(bars, fractions, masses):
        ax.text(
            bar.get_width() + 0.0015,
            bar.get_y() + bar.get_height() / 2,
            f'{frac * 100:.1f}%  ({mass:,.0f} lbm)',
            va='center', ha='left', fontsize=8.0, color='#2a2a2a',
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.0)
    ax.tick_params(axis='y', length=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.55)

    max_frac = max(fractions) if fractions else 0.0
    ax.set_xlim(0, max_frac * 1.32)

    # Category legend
    seen: set[str] = set()
    handles = []
    for cat in categories:
        if cat not in seen:
            handles.append(mpatches.Patch(
                facecolor=_CATEGORY_COLOR[cat], label=cat, edgecolor='white',
            ))
            seen.add(cat)
    ax.legend(handles=handles, loc='lower right', fontsize=8.5,
              framealpha=0.85, edgecolor='#cccccc')

    return max_frac


# ---------------------------------------------------------------------------
# Two-database comparison plot
# ---------------------------------------------------------------------------

def _plot_comparison(
    ax: plt.Axes,
    data_a: dict, label_a: str,
    data_b: dict, label_b: str,
):
    gross_a = data_a['gross_mass']
    gross_b = data_b['gross_mass']

    # Union of all labels across both DBs, keyed by label → (category, mass_a, mass_b)
    catalog_order = [label for _, label, _ in _VARIABLE_CATALOG]
    entry_map_a = {lbl: (cat, mass) for lbl, cat, mass in data_a['entries']}
    entry_map_b = {lbl: (cat, mass) for lbl, cat, mass in data_b['entries']}

    all_labels = [
        lbl for lbl in catalog_order
        if lbl in entry_map_a or lbl in entry_map_b
    ]

    # Sort by the larger of the two values
    def sort_key(lbl):
        mass_a = entry_map_a.get(lbl, (None, 0.0))[1]
        mass_b = entry_map_b.get(lbl, (None, 0.0))[1]
        return max(mass_a, mass_b)

    all_labels = sorted(all_labels, key=sort_key, reverse=True)

    n = len(all_labels)
    offset = 0.22
    bar_height = 0.38
    y_pos = np.arange(n)

    colors_a: list[tuple] = []
    colors_b: list[tuple] = []
    fracs_a: list[float] = []
    fracs_b: list[float] = []
    masses_a: list[float] = []
    masses_b: list[float] = []
    categories: list[str] = []

    cat_counts: dict[str, int] = {}
    for lbl in all_labels:
        cat = (entry_map_a.get(lbl) or entry_map_b.get(lbl))[0]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    cat_step: dict[str, int] = {}
    for lbl in all_labels:
        cat = (entry_map_a.get(lbl) or entry_map_b.get(lbl))[0]
        categories.append(cat)
        step = cat_step.get(cat, 0)
        base = _CATEGORY_COLOR.get(cat, '#808080')
        color = _shade(base, step, cat_counts[cat])
        colors_a.append(color)
        # DB-B: lighten further for distinction
        r, g, b = color
        colors_b.append((
            r + (1 - r) * 0.30,
            g + (1 - g) * 0.30,
            b + (1 - b) * 0.30,
        ))
        cat_step[cat] = step + 1

        mass_a = entry_map_a.get(lbl, (None, 0.0))[1]
        mass_b = entry_map_b.get(lbl, (None, 0.0))[1]
        fracs_a.append(mass_a / gross_a)
        fracs_b.append(mass_b / gross_b)
        masses_a.append(mass_a)
        masses_b.append(mass_b)

    # Draw bars: DB-A above centre, DB-B below
    bars_a = ax.barh(
        y_pos + offset, fracs_a, height=bar_height,
        color=colors_a, edgecolor='white', linewidth=0.6,
    )
    bars_b = ax.barh(
        y_pos - offset, fracs_b, height=bar_height,
        color=colors_b, edgecolor='white', linewidth=0.6,
        hatch='//', alpha=0.95,
    )
    ax.invert_yaxis()

    # Annotations
    for bar, frac, mass in zip(bars_a, fracs_a, masses_a):
        if mass > 0:
            ax.text(
                bar.get_width() + 0.0012,
                bar.get_y() + bar.get_height() / 2,
                f'{frac * 100:.1f}%  ({mass:,.0f})',
                va='center', ha='left', fontsize=7.2, color='#222222',
            )
    for bar, frac, mass in zip(bars_b, fracs_b, masses_b):
        if mass > 0:
            ax.text(
                bar.get_width() + 0.0012,
                bar.get_y() + bar.get_height() / 2,
                f'{frac * 100:.1f}%  ({mass:,.0f})',
                va='center', ha='left', fontsize=7.2, color='#444444',
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_labels, fontsize=8.5)
    ax.tick_params(axis='y', length=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.55)

    max_frac = max(max(fracs_a), max(fracs_b)) if fracs_a else 0.0
    ax.set_xlim(0, max_frac * 1.38)

    # Legend: categories + DB identifiers
    cat_handles = []
    seen: set[str] = set()
    for cat in categories:
        if cat not in seen:
            cat_handles.append(mpatches.Patch(
                facecolor=_CATEGORY_COLOR[cat], label=cat, edgecolor='white',
            ))
            seen.add(cat)

    db_handles = [
        mpatches.Patch(facecolor='#aaaaaa', label=label_a, edgecolor='white'),
        mpatches.Patch(facecolor='#cccccc', label=label_b,
                       edgecolor='white', hatch='//'),
    ]
    ax.legend(
        handles=cat_handles + [mpatches.Patch(visible=False)] + db_handles,
        loc='lower right', fontsize=8.0,
        framealpha=0.88, edgecolor='#cccccc',
    )

    return max_frac


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plot_dashboard(db_paths: list[str]):
    """Load data and display the mass fraction dashboard.

    Args:
        db_paths: one or two paths to SQLite case databases.
    """
    if len(db_paths) == 1:
        data = load_mass_data(db_paths[0])
        gross = data['gross_mass']
        n = len(data['entries'])
        fig_height = max(6, n * 0.50 + 2.2)
        fig, ax = plt.subplots(figsize=(12, fig_height))
        fig.patch.set_facecolor('#f5f7fa')
        ax.set_facecolor('#f5f7fa')
        _plot_single(ax, data, gross)
        ax.xaxis.set_major_formatter(
            mticker.PercentFormatter(xmax=1.0, decimals=0)
        )
        ax.set_xlabel('Fraction of Gross Mass', fontsize=10, labelpad=6)
        db_name = Path(db_paths[0]).name
        fig.suptitle(
            f'Mass Fractions — {db_name}\n'
            f'Gross Mass: {gross:,.0f} lbm',
            fontsize=12, fontweight='bold', y=0.99,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))

    else:
        data_a = load_mass_data(db_paths[0])
        data_b = load_mass_data(db_paths[1])
        label_a = _label_for_path(db_paths[0])
        label_b = _label_for_path(db_paths[1])

        n_items = len({
            lbl
            for lbl, _, _ in data_a['entries'] + data_b['entries']
        })
        fig_height = max(7, n_items * 0.62 + 2.5)
        fig, ax = plt.subplots(figsize=(13, fig_height))
        fig.patch.set_facecolor('#f5f7fa')
        ax.set_facecolor('#f5f7fa')
        _plot_comparison(ax, data_a, label_a, data_b, label_b)
        ax.xaxis.set_major_formatter(
            mticker.PercentFormatter(xmax=1.0, decimals=0)
        )
        ax.set_xlabel('Fraction of Gross Mass', fontsize=10, labelpad=6)
        gross_a = data_a['gross_mass']
        gross_b = data_b['gross_mass']
        fig.suptitle(
            f'Mass Fraction Comparison\n'
            f'{label_a}: {gross_a:,.0f} lbm     '
            f'{label_b}: {gross_b:,.0f} lbm',
            fontsize=12, fontweight='bold', y=0.99,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))

    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """Entry point for the mass fraction dashboard CLI."""
    parser = argparse.ArgumentParser(
        description=(
            'Mass fraction dashboard for Aviary / OpenMDAO SQLite databases. '
            'Pass one database for a single chart or two for a comparison.'
        )
    )
    parser.add_argument(
        'database',
        nargs='+',
        metavar='DB',
        help='Path(s) to SQLite database file(s) (1 or 2)',
    )
    args = parser.parse_args()
    if len(args.database) > 2:
        parser.error('At most two database paths are supported.')
    plot_dashboard(args.database)


if __name__ == '__main__':
    main()
