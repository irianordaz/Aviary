"""Mass fraction dashboard for an Aviary / OpenMDAO SQLite case database.

Reads the last solved case and renders a horizontal bar chart of mass
fractions (each component mass divided by aircraft gross mass). Bars are
sorted largest-to-smallest from top to bottom. Each category of component
is given a distinct colour.

Usage:
    python mass_fraction_dashboard.py path/to/problem_history.db
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
# Curated variable list: (promoted_name, display_label, category)
# Zero-valued entries are dropped automatically.
# ---------------------------------------------------------------------------

_CATEGORY_FUEL = 'Fuel'
_CATEGORY_PAYLOAD = 'Payload'
_CATEGORY_CREW = 'Crew & Service'
_CATEGORY_STRUCTURE = 'Structure'
_CATEGORY_PROPULSION = 'Propulsion'
_CATEGORY_SYSTEMS = 'Systems'
_CATEGORY_MARGIN = 'Margin'

_VARIABLE_CATALOG: list[tuple[str, str, str]] = [
    # Fuel
    ('mission:block_fuel',
     'Block Fuel', _CATEGORY_FUEL),
    # Payload
    ('aircraft:crew_and_payload:passenger_mass_total',
     'Passengers', _CATEGORY_PAYLOAD),
    ('aircraft:crew_and_payload:baggage_mass',
     'Baggage', _CATEGORY_PAYLOAD),
    ('aircraft:crew_and_payload:cargo_mass',
     'Cargo', _CATEGORY_PAYLOAD),
    ('aircraft:crew_and_payload:misc_cargo',
     'Misc Cargo', _CATEGORY_PAYLOAD),
    ('aircraft:crew_and_payload:wing_cargo',
     'Wing Cargo', _CATEGORY_PAYLOAD),
    # Crew & Service
    ('aircraft:crew_and_payload:flight_crew_mass',
     'Flight Crew', _CATEGORY_CREW),
    ('aircraft:crew_and_payload:cabin_crew_mass',
     'Cabin Crew', _CATEGORY_CREW),
    ('aircraft:crew_and_payload:passenger_service_mass',
     'Passenger Service', _CATEGORY_CREW),
    ('aircraft:crew_and_payload:cargo_container_mass',
     'Cargo Containers', _CATEGORY_CREW),
    # Structure
    ('aircraft:fuselage:mass',
     'Fuselage', _CATEGORY_STRUCTURE),
    ('aircraft:wing:mass',
     'Wing', _CATEGORY_STRUCTURE),
    ('aircraft:landing_gear:main_gear_mass',
     'Main Landing Gear', _CATEGORY_STRUCTURE),
    ('aircraft:landing_gear:nose_gear_mass',
     'Nose Landing Gear', _CATEGORY_STRUCTURE),
    ('aircraft:horizontal_tail:mass',
     'Horizontal Tail', _CATEGORY_STRUCTURE),
    ('aircraft:vertical_tail:mass',
     'Vertical Tail', _CATEGORY_STRUCTURE),
    ('aircraft:canard:mass',
     'Canard', _CATEGORY_STRUCTURE),
    ('aircraft:fins:mass',
     'Fins', _CATEGORY_STRUCTURE),
    ('aircraft:nacelle:mass',
     'Nacelle', _CATEGORY_STRUCTURE),
    ('aircraft:paint:mass',
     'Paint', _CATEGORY_STRUCTURE),
    # Propulsion — individual sub-components
    ('aircraft:propulsion:total_engine_mass',
     'Engines', _CATEGORY_PROPULSION),
    ('aircraft:fuel:fuel_system_mass',
     'Fuel System', _CATEGORY_PROPULSION),
    ('aircraft:propulsion:total_thrust_reversers_mass',
     'Thrust Reversers', _CATEGORY_PROPULSION),
    ('aircraft:propulsion:total_engine_controls_mass',
     'Engine Controls', _CATEGORY_PROPULSION),
    ('aircraft:propulsion:total_starter_mass',
     'Engine Starters', _CATEGORY_PROPULSION),
    ('aircraft:battery:mass',
     'Battery', _CATEGORY_PROPULSION),
    # Systems
    ('aircraft:furnishings:mass',
     'Furnishings', _CATEGORY_SYSTEMS),
    ('aircraft:wing:surface_control_mass',
     'Surface Controls', _CATEGORY_SYSTEMS),
    ('aircraft:electrical:mass',
     'Electrical', _CATEGORY_SYSTEMS),
    ('aircraft:avionics:mass',
     'Avionics', _CATEGORY_SYSTEMS),
    ('aircraft:air_conditioning:mass',
     'Air Conditioning', _CATEGORY_SYSTEMS),
    ('aircraft:apu:mass',
     'APU', _CATEGORY_SYSTEMS),
    ('aircraft:instruments:mass',
     'Instruments', _CATEGORY_SYSTEMS),
    ('aircraft:hydraulics:mass',
     'Hydraulics', _CATEGORY_SYSTEMS),
    ('aircraft:anti_icing:mass',
     'Anti-Icing', _CATEGORY_SYSTEMS),
    # Margin
    ('aircraft:design:empty_mass_margin',
     'Empty Mass Margin', _CATEGORY_MARGIN),
]

_GROSS_MASS_PROM = 'aircraft:design:gross_mass'

# Base colour (hex) for each category
_CATEGORY_BASE_COLOR: dict[str, str] = {
    _CATEGORY_FUEL:      '#E8871A',
    _CATEGORY_PAYLOAD:   '#2E9E4F',
    _CATEGORY_CREW:      '#1A9E8E',
    _CATEGORY_STRUCTURE: '#2060C8',
    _CATEGORY_PROPULSION:'#C82020',
    _CATEGORY_SYSTEMS:   '#8030B0',
    _CATEGORY_MARGIN:    '#6E8090',
}

# Number of shading steps available per category
_SHADE_STEPS = 6


def _shade(hex_color: str, step: int, total: int) -> tuple:
    """Return a lightened variant of *hex_color* for within-category shading.

    step 0 → darkest (base colour), step total-1 → lightest.
    """
    r = int(hex_color[1:3], 16) / 255
    g = int(hex_color[3:5], 16) / 255
    b = int(hex_color[5:7], 16) / 255
    # Blend toward white proportionally
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
    """Read mass variables from the last solved case in the database.

    Returns:
        A dict with keys:
          - 'gross_mass': float in lbm
          - 'entries': list of (label, category, mass_lbm) for all
            non-zero catalogued variables found in the database
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute('SELECT abs2prom, abs2meta FROM metadata LIMIT 1')
        row = cur.fetchone()
        if row is None:
            raise ValueError('No metadata found in database.')

        abs2prom_all = _decompress_blob(row[0])
        abs2meta = _decompress_blob(row[1])
        prom_out = abs2prom_all.get('output', {})

        cur.execute(
            'SELECT outputs FROM problem_cases ORDER BY counter DESC LIMIT 1'
        )
        result = cur.fetchone()
        if result is None:
            raise ValueError('No solved cases found in database.')

        outputs = json.loads(result[0]) if result[0] else {}
    finally:
        conn.close()

    # Build a promoted → scalar-value map (outputs only, lbm units)
    prom_to_value: dict[str, float] = {}
    for abs_name, raw_val in outputs.items():
        prom = prom_out.get(abs_name)
        if prom is None:
            continue
        units = abs2meta.get(abs_name, {}).get('units', '')
        if units != 'lbm':
            continue
        # Keep the first occurrence for each promoted name
        if prom not in prom_to_value:
            prom_to_value[prom] = _scalar(raw_val)

    gross_mass = prom_to_value.get(_GROSS_MASS_PROM)
    if gross_mass is None:
        raise ValueError(
            f"Could not find '{_GROSS_MASS_PROM}' in the database outputs."
        )

    entries: list[tuple[str, str, float]] = []
    for prom, label, category in _VARIABLE_CATALOG:
        mass = prom_to_value.get(prom)
        if mass is not None and mass > 0.0:
            entries.append((label, category, mass))

    return {'gross_mass': gross_mass, 'entries': entries}


# ---------------------------------------------------------------------------
# Chart data
# ---------------------------------------------------------------------------

def build_chart_data(
    mass_data: dict,
) -> tuple[list[str], list[str], list[float], list[float], list[tuple]]:
    """Return sorted parallel lists ready for plotting.

    Returns:
        labels:    display names, largest fraction first
        categories: category name per entry
        fractions: mass / gross_mass per entry
        masses:    raw mass in lbm per entry
        colors:    RGBA tuple per entry
    """
    gross = mass_data['gross_mass']
    entries = sorted(mass_data['entries'], key=lambda x: x[2], reverse=True)

    # Count items per category (for shading)
    cat_counts: dict[str, int] = {}
    for _, cat, _ in entries:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    cat_step: dict[str, int] = {}
    labels, categories, fractions, masses, colors = [], [], [], [], []

    for label, cat, mass in entries:
        step = cat_step.get(cat, 0)
        total = cat_counts[cat]
        base = _CATEGORY_BASE_COLOR.get(cat, '#808080')
        colors.append(_shade(base, step, total))
        cat_step[cat] = step + 1

        labels.append(label)
        categories.append(cat)
        fractions.append(mass / gross)
        masses.append(mass)

    return labels, categories, fractions, masses, colors


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_dashboard(db_path: str):
    """Load data and display the mass fraction dashboard."""
    mass_data = load_mass_data(db_path)
    labels, categories, fractions, masses, colors = build_chart_data(
        mass_data
    )
    gross = mass_data['gross_mass']

    n = len(labels)
    bar_height = 0.58
    fig_height = max(6, n * 0.50 + 2.2)

    fig, ax = plt.subplots(figsize=(12, fig_height))
    fig.patch.set_facecolor('#f5f7fa')
    ax.set_facecolor('#f5f7fa')

    # Sorted descending → index 0 = largest; invert_yaxis puts 0 at top
    y_pos = np.arange(n)
    bars = ax.barh(
        y_pos,
        fractions,
        height=bar_height,
        color=colors,
        edgecolor='white',
        linewidth=0.7,
    )
    ax.invert_yaxis()

    # Inline annotations
    for bar, frac, mass in zip(bars, fractions, masses):
        x_end = bar.get_width()
        ax.text(
            x_end + 0.0015,
            bar.get_y() + bar.get_height() / 2,
            f'{frac * 100:.1f}%  ({mass:,.0f} lbm)',
            va='center',
            ha='left',
            fontsize=8.0,
            color='#2a2a2a',
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.0)
    ax.xaxis.set_major_formatter(
        mticker.PercentFormatter(xmax=1.0, decimals=0)
    )
    ax.set_xlabel('Fraction of Gross Mass', fontsize=10, labelpad=6)

    max_frac = max(fractions) if fractions else 0.0
    ax.set_xlim(0, max_frac * 1.32)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.55)

    # Category legend
    legend_handles = []
    seen_cats: set[str] = set()
    for cat, color in zip(categories, colors):
        if cat not in seen_cats:
            legend_handles.append(
                mpatches.Patch(
                    facecolor=_CATEGORY_BASE_COLOR[cat],
                    label=cat,
                    edgecolor='white',
                )
            )
            seen_cats.add(cat)
    ax.legend(
        handles=legend_handles,
        loc='lower right',
        fontsize=8.5,
        framealpha=0.85,
        edgecolor='#cccccc',
    )

    db_name = Path(db_path).name
    fig.suptitle(
        f'Mass Fractions — {db_name}\n'
        f'Gross Mass: {gross:,.0f} lbm',
        fontsize=12,
        fontweight='bold',
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Entry point for the mass fraction dashboard CLI."""
    parser = argparse.ArgumentParser(
        description=(
            'Display a mass fraction dashboard for an Aviary / OpenMDAO '
            'SQLite case database.'
        )
    )
    parser.add_argument(
        'database',
        help='Path to the SQLite database file (problem_history.db)',
    )
    args = parser.parse_args()
    plot_dashboard(args.database)


if __name__ == '__main__':
    main()
