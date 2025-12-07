#!/usr/bin/env python3
"""
Characteristic Distribution Heatmap for CMV Taxonomy

This script generates visualizations demonstrating that the characteristic structure
reflects empirical observation: most current agentic AI systems cluster at C_.2
(intermediate) levels, with advanced characteristics (C_.4) remaining relatively rare.

This distribution validates that the taxonomy captures the actual state of the field
rather than aspirational ideals.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# CMV Matrix Data from 6. CMV_Framework_IntegrationNDeliverables.md
# Format: Object ID, Name, D1-D9 characteristic levels (1-4)
CMV_DATA = {
    'O1':  {'name': 'ReAct Agent',              'D1': 2, 'D2': 2, 'D3': 1, 'D4': 1, 'D5': 3, 'D6': 2, 'D7': 2, 'D8': 2, 'D9': 2},
    'O2':  {'name': 'AutoGPT',                  'D1': 3, 'D2': 3, 'D3': 2, 'D4': 1, 'D5': 4, 'D6': 3, 'D7': 2, 'D8': 1, 'D9': 1},
    'O3':  {'name': 'BabyAGI',                  'D1': 2, 'D2': 2, 'D3': 1, 'D4': 1, 'D5': 4, 'D6': 2, 'D7': 2, 'D8': 1, 'D9': 1},
    'O4':  {'name': 'Generative Agents',        'D1': 4, 'D2': 1, 'D3': 4, 'D4': 1, 'D5': 3, 'D6': 3, 'D7': 3, 'D8': 2, 'D9': 2},
    'O5':  {'name': 'AgentBench',               'D1': 2, 'D2': 3, 'D3': 1, 'D4': 2, 'D5': 3, 'D6': 2, 'D7': 1, 'D8': 2, 'D9': 2},
    'O6':  {'name': 'Role-based Cooperation',   'D1': 2, 'D2': 2, 'D3': 1, 'D4': 2, 'D5': 3, 'D6': 2, 'D7': 4, 'D8': 2, 'D9': 2},
    'O7':  {'name': 'Debate-based Cooperation', 'D1': 4, 'D2': 1, 'D3': 1, 'D4': 3, 'D5': 3, 'D6': 3, 'D7': 3, 'D8': 2, 'D9': 2},
    'O8':  {'name': 'Passive Goal Creator',     'D1': 1, 'D2': 1, 'D3': 1, 'D4': 1, 'D5': 1, 'D6': 1, 'D7': 2, 'D8': 2, 'D9': 2},
    'O9':  {'name': 'Proactive Goal Creator',   'D1': 2, 'D2': 2, 'D3': 2, 'D4': 1, 'D5': 4, 'D6': 2, 'D7': 2, 'D8': 1, 'D9': 1},
    'O10': {'name': 'AIOps Agents',             'D1': 2, 'D2': 3, 'D3': 2, 'D4': 1, 'D5': 3, 'D6': 2, 'D7': 3, 'D8': 3, 'D9': 3},
    'O11': {'name': 'SW Engineering Agents',    'D1': 2, 'D2': 3, 'D3': 2, 'D4': 1, 'D5': 3, 'D6': 2, 'D7': 2, 'D8': 2, 'D9': 2},
    'O12': {'name': 'ERP-Integrated Agents',    'D1': 2, 'D2': 4, 'D3': 3, 'D4': 1, 'D5': 2, 'D6': 2, 'D7': 3, 'D8': 3, 'D9': 3},
    'O13': {'name': 'Human-Agent Teams',        'D1': 3, 'D2': 2, 'D3': 2, 'D4': 4, 'D5': 3, 'D6': 3, 'D7': 4, 'D8': 3, 'D9': 3},
    'O14': {'name': 'Deep Research Agents',     'D1': 3, 'D2': 3, 'D3': 2, 'D4': 1, 'D5': 4, 'D6': 2, 'D7': 1, 'D8': 2, 'D9': 2},
    'O15': {'name': 'Agent Evaluator',          'D1': 2, 'D2': 1, 'D3': 1, 'D4': 1, 'D5': 1, 'D6': 1, 'D7': 1, 'D8': 2, 'D9': 2},
    'O16': {'name': 'MRKL Architecture',        'D1': 2, 'D2': 3, 'D3': 1, 'D4': 1, 'D5': 3, 'D6': 2, 'D7': 2, 'D8': 2, 'D9': 2},
    'O17': {'name': 'Long-Horizon Simulation',  'D1': 2, 'D2': 2, 'D3': 3, 'D4': 1, 'D5': 3, 'D6': 2, 'D7': 2, 'D8': 2, 'D9': 3},
    'O18': {'name': 'Enterprise RBAC Agents',   'D1': 2, 'D2': 4, 'D3': 2, 'D4': 1, 'D5': 2, 'D6': 2, 'D7': 2, 'D8': 3, 'D9': 3},
    'O19': {'name': 'Safeguarded Models',       'D1': 3, 'D2': 3, 'D3': 2, 'D4': 1, 'D5': 2, 'D6': 3, 'D7': 3, 'D8': 4, 'D9': 4},
    'O20': {'name': 'Agent Workflow Systems',   'D1': 2, 'D2': 3, 'D3': 1, 'D4': 2, 'D5': 2, 'D6': 1, 'D7': 2, 'D8': 3, 'D9': 3},
    'O21': {'name': 'End-to-End Agent Models',  'D1': 3, 'D2': 3, 'D3': 2, 'D4': 1, 'D5': 4, 'D6': 4, 'D7': 2, 'D8': 1, 'D9': 3},
    'O22': {'name': 'Cross-Reflection Pattern', 'D1': 4, 'D2': 1, 'D3': 1, 'D4': 3, 'D5': 3, 'D6': 3, 'D7': 3, 'D8': 2, 'D9': 2},
    'O23': {'name': 'Tool/Agent Registry',      'D1': 1, 'D2': 3, 'D3': 1, 'D4': 2, 'D5': 1, 'D6': 1, 'D7': 1, 'D8': 2, 'D9': 2},
    'O24': {'name': 'Memory-Enhanced Agent',    'D1': 3, 'D2': 2, 'D3': 4, 'D4': 1, 'D5': 3, 'D6': 4, 'D7': 2, 'D8': 2, 'D9': 2},
    'O25': {'name': 'Industry 4.0 AgentAI',     'D1': 2, 'D2': 4, 'D3': 2, 'D4': 4, 'D5': 3, 'D6': 3, 'D7': 3, 'D8': 4, 'D9': 4},
}

DIMENSIONS = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9']
DIMENSION_NAMES = {
    'D1': 'Reasoning',
    'D2': 'Tool Integration',
    'D3': 'Memory',
    'D4': 'Coordination',
    'D5': 'Autonomy',
    'D6': 'Adaptation',
    'D7': 'Human Collab',
    'D8': 'Compliance',
    'D9': 'Safety'
}

CHAR_LABELS = {1: 'C_.1\n(Basic)', 2: 'C_.2\n(Intermediate)', 3: 'C_.3\n(Advanced)', 4: 'C_.4\n(Sophisticated)'}


def create_dataframe():
    """Create pandas DataFrame from CMV data."""
    rows = []
    for obj_id, data in CMV_DATA.items():
        row = {'Object': obj_id, 'Name': data['name']}
        for dim in DIMENSIONS:
            row[dim] = data[dim]
        rows.append(row)
    return pd.DataFrame(rows)


def calculate_statistics(df):
    """Calculate distribution statistics."""
    all_values = []
    for dim in DIMENSIONS:
        all_values.extend(df[dim].tolist())

    counter = Counter(all_values)
    total = len(all_values)

    stats = {
        'total': total,
        'distribution': {
            1: {'count': counter[1], 'pct': counter[1] / total * 100},
            2: {'count': counter[2], 'pct': counter[2] / total * 100},
            3: {'count': counter[3], 'pct': counter[3] / total * 100},
            4: {'count': counter[4], 'pct': counter[4] / total * 100},
        }
    }

    # Per-dimension statistics
    stats['per_dimension'] = {}
    for dim in DIMENSIONS:
        dim_counter = Counter(df[dim].tolist())
        stats['per_dimension'][dim] = {
            1: dim_counter.get(1, 0),
            2: dim_counter.get(2, 0),
            3: dim_counter.get(3, 0),
            4: dim_counter.get(4, 0),
        }

    return stats


def plot_main_heatmap(df, stats, output_path):
    """Generate the main heatmap visualization with multiple panels."""
    fig = plt.figure(figsize=(16, 14))

    # Create grid for subplots
    gs = fig.add_gridspec(3, 2, height_ratios=[2.5, 1, 1], width_ratios=[3, 1],
                          hspace=0.3, wspace=0.2)

    # ============================================
    # Panel 1: Main Heatmap (Objects x Dimensions)
    # ============================================
    ax1 = fig.add_subplot(gs[0, 0])

    # Prepare heatmap data
    heatmap_data = df[DIMENSIONS].values

    # Custom colormap: light to dark blue
    cmap = sns.color_palette("YlOrRd", as_cmap=True)

    sns.heatmap(heatmap_data,
                ax=ax1,
                cmap=cmap,
                vmin=1, vmax=4,
                annot=True,
                fmt='d',
                cbar_kws={'label': 'Characteristic Level', 'ticks': [1, 2, 3, 4]},
                xticklabels=[DIMENSION_NAMES[d] for d in DIMENSIONS],
                yticklabels=[f"{row['Object']}: {row['Name'][:15]}" for _, row in df.iterrows()],
                linewidths=0.5,
                linecolor='white')

    ax1.set_title('Characteristic Distribution Across 25 Reference Objects\n(CMV Taxonomy Validation)',
                  fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlabel('Capability Dimension', fontsize=11)
    ax1.set_ylabel('Reference Object', fontsize=11)
    ax1.tick_params(axis='x', rotation=45)

    # ============================================
    # Panel 2: Overall Distribution Bar Chart
    # ============================================
    ax2 = fig.add_subplot(gs[0, 1])

    levels = [1, 2, 3, 4]
    counts = [stats['distribution'][l]['count'] for l in levels]
    pcts = [stats['distribution'][l]['pct'] for l in levels]
    colors = plt.cm.YlOrRd([0.2, 0.4, 0.6, 0.8])

    bars = ax2.barh(['C_.1\nBasic', 'C_.2\nIntermediate', 'C_.3\nAdvanced', 'C_.4\nSophisticated'],
                    counts, color=colors, edgecolor='black', linewidth=0.5)

    # Add count and percentage labels
    for i, (bar, count, pct) in enumerate(zip(bars, counts, pcts)):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f'{count} ({pct:.1f}%)', va='center', fontsize=10, fontweight='bold')

    ax2.set_xlabel('Count (out of 225 cells)', fontsize=11)
    ax2.set_title('Overall Distribution\n(All Objects × All Dimensions)', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, max(counts) * 1.3)
    ax2.axvline(x=225/4, color='gray', linestyle='--', alpha=0.5, label='Uniform distribution')

    # ============================================
    # Panel 3: Per-Dimension Stacked Bar Chart
    # ============================================
    ax3 = fig.add_subplot(gs[1, :])

    dim_labels = [DIMENSION_NAMES[d] for d in DIMENSIONS]
    x = np.arange(len(DIMENSIONS))
    width = 0.6

    bottom = np.zeros(len(DIMENSIONS))
    for level in [1, 2, 3, 4]:
        values = [stats['per_dimension'][d][level] for d in DIMENSIONS]
        ax3.bar(x, values, width, bottom=bottom, label=f'C_.{level}',
                color=plt.cm.YlOrRd((level-1)/4 + 0.1), edgecolor='white', linewidth=0.5)
        bottom += values

    ax3.set_xlabel('Capability Dimension', fontsize=11)
    ax3.set_ylabel('Count (25 objects)', fontsize=11)
    ax3.set_title('Characteristic Distribution Per Dimension', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(dim_labels, rotation=45, ha='right')
    ax3.legend(title='Level', loc='upper right', ncol=4)
    ax3.set_ylim(0, 27)

    # ============================================
    # Panel 4: Summary Statistics Box
    # ============================================
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    # Calculate key insights
    c2_pct = stats['distribution'][2]['pct']
    c1_c2_pct = stats['distribution'][1]['pct'] + stats['distribution'][2]['pct']
    c4_pct = stats['distribution'][4]['pct']

    summary_text = f"""
    KEY FINDINGS: Characteristic Structure Reflects Empirical Reality
    ══════════════════════════════════════════════════════════════════════════════════════════

    Total Classifications: {stats['total']} cells (25 objects × 9 dimensions)

    DISTRIBUTION ANALYSIS:
    ├─ C_.1 (Basic):         {stats['distribution'][1]['count']:3d} cells ({stats['distribution'][1]['pct']:5.1f}%)  │  Foundational capabilities
    ├─ C_.2 (Intermediate):  {stats['distribution'][2]['count']:3d} cells ({stats['distribution'][2]['pct']:5.1f}%)  │  ◀ MOST COMMON - Current industry standard
    ├─ C_.3 (Advanced):      {stats['distribution'][3]['count']:3d} cells ({stats['distribution'][3]['pct']:5.1f}%)  │  Emerging capabilities
    └─ C_.4 (Sophisticated): {stats['distribution'][4]['count']:3d} cells ({stats['distribution'][4]['pct']:5.1f}%)  │  ◀ RARE - Frontier capabilities

    VALIDATION INSIGHT:
    • {c1_c2_pct:.1f}% of systems operate at basic-to-intermediate levels (C_.1 + C_.2)
    • Only {c4_pct:.1f}% achieve sophisticated levels (C_.4)
    • This distribution validates that the taxonomy captures actual field state, not aspirational ideals
    """

    ax4.text(0.02, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def print_statistics(stats):
    """Print detailed statistics to console."""
    print("\n" + "="*70)
    print("CMV TAXONOMY: CHARACTERISTIC DISTRIBUTION ANALYSIS")
    print("="*70)

    print(f"\nTotal cells analyzed: {stats['total']} (25 objects × 9 dimensions)")
    print("\n" + "-"*50)
    print("OVERALL DISTRIBUTION:")
    print("-"*50)

    for level in [1, 2, 3, 4]:
        d = stats['distribution'][level]
        bar = "█" * int(d['pct'] / 2)
        print(f"  C_.{level}: {d['count']:3d} cells ({d['pct']:5.1f}%) {bar}")

    print("\n" + "-"*50)
    print("PER-DIMENSION DISTRIBUTION:")
    print("-"*50)
    print(f"{'Dimension':<20} {'C_.1':>6} {'C_.2':>6} {'C_.3':>6} {'C_.4':>6} {'Mode':>8}")
    print("-"*50)

    for dim in DIMENSIONS:
        d = stats['per_dimension'][dim]
        mode = max(d, key=d.get)
        print(f"{DIMENSION_NAMES[dim]:<20} {d[1]:>6} {d[2]:>6} {d[3]:>6} {d[4]:>6} {'C_.'+str(mode):>8}")

    print("\n" + "-"*50)
    print("KEY INSIGHTS:")
    print("-"*50)

    c2_count = stats['distribution'][2]['count']
    c4_count = stats['distribution'][4]['count']
    c1_c2_pct = stats['distribution'][1]['pct'] + stats['distribution'][2]['pct']

    print(f"  • C_.2 (Intermediate) is MOST COMMON with {c2_count} occurrences ({stats['distribution'][2]['pct']:.1f}%)")
    print(f"  • C_.4 (Sophisticated) is RARE with only {c4_count} occurrences ({stats['distribution'][4]['pct']:.1f}%)")
    print(f"  • {c1_c2_pct:.1f}% of all classifications are at basic-to-intermediate levels")
    print(f"\n  CONCLUSION: The taxonomy captures the ACTUAL state of the field,")
    print(f"              not aspirational ideals. Most systems cluster at intermediate")
    print(f"              capability levels with advanced characteristics remaining rare.")
    print("="*70 + "\n")


def main():
    """Main function to generate heatmap and statistics."""
    print("Generating CMV Taxonomy Characteristic Distribution Heatmap...")

    # Create DataFrame
    df = create_dataframe()

    # Calculate statistics
    stats = calculate_statistics(df)

    # Print statistics to console
    print_statistics(stats)

    # Generate visualization
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'characteristic_heatmap.png')

    plot_main_heatmap(df, stats, output_path)
    print(f"Heatmap saved to: {output_path}")

    return df, stats


if __name__ == "__main__":
    main()
