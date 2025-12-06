#!/usr/bin/env python3
"""
taxonomy_quality_metrics.py
Calculates quality metrics for CMV Taxonomy
Author: Generated for CMV Taxonomy PAKDD 2026 submission
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr
from itertools import combinations

# Classification matrix (from paper Table app:classification)
# Format: [D1, D2, D3, D4, D5, D6, D7, D8, D9]
DATA = {
    'O1':  [2, 2, 1, 1, 3, 2, 2, 2, 2],  # ReAct Agent
    'O2':  [3, 3, 2, 1, 4, 3, 2, 1, 1],  # AutoGPT
    'O3':  [2, 2, 1, 1, 4, 2, 2, 1, 1],  # BabyAGI
    'O4':  [4, 1, 4, 1, 3, 3, 3, 2, 2],  # Generative Agents
    'O5':  [2, 3, 1, 2, 3, 2, 1, 2, 2],  # AgentBench
    'O6':  [2, 2, 1, 2, 3, 2, 4, 2, 2],  # Role-Based Pattern
    'O7':  [4, 1, 1, 3, 3, 3, 3, 2, 2],  # Debate-Based Pattern
    'O8':  [1, 1, 1, 1, 1, 1, 2, 2, 2],  # Prompt Chain
    'O9':  [2, 2, 2, 1, 4, 2, 2, 1, 1],  # Goal Creator
    'O10': [2, 3, 2, 1, 3, 2, 3, 3, 3],  # AIOps Agents
    'O11': [2, 3, 2, 1, 3, 2, 2, 2, 2],  # SW Engineering Agent
    'O12': [2, 4, 3, 1, 2, 2, 3, 3, 3],  # ERP-Integrated
    'O13': [3, 2, 2, 4, 3, 3, 4, 3, 3],  # Human-Agent Teams
    'O14': [3, 3, 2, 1, 4, 2, 1, 2, 2],  # Deep Research Agent
    'O15': [2, 1, 1, 1, 1, 1, 1, 2, 2],  # Evaluator Agent
    'O16': [2, 3, 1, 1, 3, 2, 2, 2, 2],  # MRKL Agent
    'O17': [2, 2, 3, 1, 3, 2, 2, 2, 3],  # Long-Horizon Simulation
    'O18': [2, 4, 2, 1, 2, 2, 2, 3, 3],  # RBAC-Compliant Agent
    'O19': [3, 3, 2, 1, 2, 3, 3, 4, 4],  # Safeguarded Models
    'O20': [2, 3, 1, 2, 2, 1, 2, 3, 3],  # Agent Workflows
    'O21': [3, 3, 2, 1, 4, 4, 2, 1, 3],  # End-to-End Agent
    'O22': [4, 1, 1, 3, 3, 3, 3, 2, 2],  # Cross-Reflection Pattern
    'O23': [1, 3, 1, 2, 1, 1, 1, 2, 2],  # Tool Registry
    'O24': [3, 2, 4, 1, 3, 4, 2, 2, 2],  # Memory-Enhanced Agent
    'O25': [2, 4, 2, 4, 3, 3, 3, 4, 4],  # Industry 4.0 Agent
}

DIMS = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9']
DIM_NAMES = {
    'D1': 'Reasoning',
    'D2': 'Tool Integration',
    'D3': 'Memory',
    'D4': 'Coordination',
    'D5': 'Autonomy',
    'D6': 'Adaptation',
    'D7': 'Human Collaboration',
    'D8': 'Compliance',
    'D9': 'Safety'
}


def create_dataframe():
    """Create DataFrame from classification data."""
    df = pd.DataFrame(DATA).T
    df.columns = DIMS
    return df


def calculate_uniqueness(df):
    """Calculate profile uniqueness."""
    unique_profiles = df.drop_duplicates().shape[0]
    total = df.shape[0]
    return unique_profiles, total, unique_profiles / total * 100


def calculate_hamming_distances(df):
    """Calculate pairwise Hamming distances."""
    objects = list(df.index)
    min_dist = float('inf')
    min_pair = None
    all_distances = []

    for i, j in combinations(range(len(objects)), 2):
        dist = sum(df.iloc[i] != df.iloc[j])
        all_distances.append(dist)
        if dist < min_dist:
            min_dist = dist
            min_pair = (objects[i], objects[j])

    avg_dist = np.mean(all_distances)
    return min_dist, min_pair, avg_dist


def calculate_entropy(counts):
    """Calculate Shannon entropy."""
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Remove zeros
    return -np.sum(probs * np.log2(probs))


def calculate_dimension_entropy(df):
    """Calculate entropy for each dimension."""
    results = {}
    max_entropy = np.log2(4)  # 4 characteristics

    for dim in DIMS:
        counts = df[dim].value_counts()
        h = calculate_entropy(counts.values)
        results[dim] = {
            'entropy': h,
            'max_entropy': max_entropy,
            'normalized': h / max_entropy,
            'distribution': counts.sort_index().to_dict()
        }
    return results


def calculate_cramers_v(df, dim1, dim2):
    """Calculate Cramer's V for two dimensions."""
    contingency = pd.crosstab(df[dim1], df[dim2])
    chi2, p, dof, expected = chi2_contingency(contingency)
    n = len(df)
    min_dim = min(contingency.shape) - 1
    v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
    return v, chi2, p


def calculate_all_cramers_v(df):
    """Calculate Cramer's V for all dimension pairs."""
    results = {}
    for dim1, dim2 in combinations(DIMS, 2):
        v, chi2, p = calculate_cramers_v(df, dim1, dim2)
        results[(dim1, dim2)] = {'V': v, 'chi2': chi2, 'p': p}
    return results


def calculate_spearman_correlations(df):
    """Calculate Spearman correlations for all dimension pairs."""
    results = {}
    for dim1, dim2 in combinations(DIMS, 2):
        rho, p = spearmanr(df[dim1], df[dim2])
        results[(dim1, dim2)] = {'rho': rho, 'p': p}
    return results


def interpret_cramers_v(v):
    """Interpret Cramer's V value."""
    if v > 0.5:
        return "Strong"
    elif v > 0.3:
        return "Moderate"
    elif v > 0.1:
        return "Weak"
    else:
        return "Negligible"


def interpret_entropy(normalized):
    """Interpret normalized entropy."""
    if normalized > 0.85:
        return "Balanced"
    elif normalized > 0.7:
        return "Moderate"
    else:
        return "Skewed"


def generate_entropy_table(entropy_results):
    """Generate plain text table for entropy results."""
    lines = []
    lines.append("Distribution Entropy by Dimension")
    lines.append("-" * 60)
    lines.append(f"{'Dim':<4} {'Name':<20} {'H (bits)':>10} {'H/Hmax':>8} {'Interpretation':<12}")
    lines.append("-" * 60)

    for dim in DIMS:
        h = entropy_results[dim]['entropy']
        norm = entropy_results[dim]['normalized']
        interp = interpret_entropy(norm)
        name = DIM_NAMES[dim]
        lines.append(f"{dim:<4} {name:<20} {h:>10.3f} {norm:>8.2f} {interp:<12}")

    lines.append("-" * 60)
    return "\n".join(lines)


def generate_cramers_table(cramers_results):
    """Generate plain text table for Cramer's V results."""
    lines = []
    lines.append("Dimension Independence Analysis (Cramer's V)")
    lines.append("-" * 60)
    lines.append(f"{'Pair':<10} {'V':>8} {'chi2':>10} {'p-value':>10} {'Association':<12}")
    lines.append("-" * 60)

    # Sort by V value descending and take top 15
    sorted_pairs = sorted(cramers_results.items(), key=lambda x: -x[1]['V'])[:15]

    for (d1, d2), vals in sorted_pairs:
        v = vals['V']
        chi2 = vals['chi2']
        p = vals['p']
        assoc = interpret_cramers_v(v)
        sig = "*" if p < 0.05 else ""
        lines.append(f"{d1}--{d2:<6} {v:>7.3f}{sig:<1} {chi2:>10.2f} {p:>10.3f} {assoc:<12}")

    lines.append("-" * 60)
    lines.append("* p < 0.05")
    return "\n".join(lines)


def generate_spearman_table(spearman_results):
    """Generate plain text table for key Spearman correlations."""
    lines = []
    lines.append("Autonomy-Governance Correlation (Spearman's rho)")
    lines.append("-" * 75)
    lines.append(f"{'Dim 1':<20} {'Dim 2':<20} {'rho':>8} {'p-value':>10} {'Interpretation':<18}")
    lines.append("-" * 75)

    # Focus on key pairs: D5 vs D8, D5 vs D9, D8 vs D9
    key_pairs = [('D5', 'D8'), ('D5', 'D9'), ('D8', 'D9')]

    for pair in key_pairs:
        if pair in spearman_results:
            vals = spearman_results[pair]
        else:
            vals = spearman_results[(pair[1], pair[0])]

        rho = vals['rho']
        p = vals['p']

        if abs(rho) > 0.5:
            interp = "Strong " + ("positive" if rho > 0 else "negative")
        elif abs(rho) > 0.3:
            interp = "Moderate " + ("positive" if rho > 0 else "negative")
        elif abs(rho) > 0.1:
            interp = "Weak " + ("positive" if rho > 0 else "negative")
        else:
            interp = "Negligible"

        sig = "*" if p < 0.05 else ""
        d1_name = DIM_NAMES[pair[0]]
        d2_name = DIM_NAMES[pair[1]]
        lines.append(f"{pair[0]} ({d1_name})"[:20].ljust(20) + f" {pair[1]} ({d2_name})"[:20].ljust(20) + f" {rho:>7.3f}{sig:<1} {p:>10.3f} {interp:<18}")

    lines.append("-" * 75)
    lines.append("* p < 0.05")
    return "\n".join(lines)


def generate_distribution_table(entropy_results):
    """Generate plain text table showing characteristic distribution."""
    lines = []
    lines.append("Characteristic Distribution by Dimension (n=25)")
    lines.append("-" * 50)
    lines.append(f"{'Dimension':<30} {'C.1':>4} {'C.2':>4} {'C.3':>4} {'C.4':>4}")
    lines.append("-" * 50)

    for dim in DIMS:
        dist = entropy_results[dim]['distribution']
        c1 = dist.get(1, 0)
        c2 = dist.get(2, 0)
        c3 = dist.get(3, 0)
        c4 = dist.get(4, 0)
        lines.append(f"{dim}: {DIM_NAMES[dim]:<26} {c1:>4} {c2:>4} {c3:>4} {c4:>4}")

    lines.append("-" * 50)
    return "\n".join(lines)


def main():
    df = create_dataframe()

    print("=" * 70)
    print("CMV TAXONOMY QUALITY METRICS")
    print("=" * 70)

    # 1. Uniqueness Analysis
    print("\n" + "=" * 70)
    print("1. PROFILE UNIQUENESS / DISCRIMINATION")
    print("=" * 70)
    unique, total, pct = calculate_uniqueness(df)
    print(f"   Unique profiles: {unique}/{total} = {pct:.1f}%")

    min_dist, min_pair, avg_dist = calculate_hamming_distances(df)
    print(f"   Minimum Hamming distance: {min_dist} (between {min_pair[0]} and {min_pair[1]})")
    print(f"   Average Hamming distance: {avg_dist:.2f}")

    # Show the two most similar profiles
    print(f"\n   Most similar pair ({min_pair[0]} vs {min_pair[1]}):")
    print(f"   {min_pair[0]}: {list(df.loc[min_pair[0]])}")
    print(f"   {min_pair[1]}: {list(df.loc[min_pair[1]])}")

    # 2. Entropy Analysis
    print("\n" + "=" * 70)
    print("2. DISTRIBUTION ENTROPY (H_max = 2.0 bits)")
    print("=" * 70)
    entropy = calculate_dimension_entropy(df)
    for dim in DIMS:
        h = entropy[dim]['entropy']
        norm = entropy[dim]['normalized']
        dist = entropy[dim]['distribution']
        interp = interpret_entropy(norm)
        print(f"   {dim} ({DIM_NAMES[dim]:18s}): H={h:.3f} ({norm*100:.1f}%) - {interp}")
        print(f"      Distribution: C.1={dist.get(1,0)}, C.2={dist.get(2,0)}, C.3={dist.get(3,0)}, C.4={dist.get(4,0)}")

    # 3. Cramer's V Analysis
    print("\n" + "=" * 70)
    print("3. DIMENSION INDEPENDENCE (Cramer's V)")
    print("=" * 70)
    cramers = calculate_all_cramers_v(df)
    sorted_pairs = sorted(cramers.items(), key=lambda x: -x[1]['V'])

    print("   Top 10 dimension pairs by association strength:")
    for (d1, d2), vals in sorted_pairs[:10]:
        v = vals['V']
        p = vals['p']
        interp = interpret_cramers_v(v)
        sig = "*" if p < 0.05 else ""
        print(f"   {d1}-{d2}: V={v:.3f}{sig}, p={p:.3f} ({interp})")

    # 4. Spearman Correlation Analysis
    print("\n" + "=" * 70)
    print("4. SPEARMAN CORRELATIONS")
    print("=" * 70)
    spearman = calculate_spearman_correlations(df)

    print("   Key pairs (Autonomy-Governance relationship):")
    key_pairs = [('D5', 'D8'), ('D5', 'D9'), ('D8', 'D9')]
    for pair in key_pairs:
        if pair in spearman:
            vals = spearman[pair]
        else:
            vals = spearman[(pair[1], pair[0])]
        rho = vals['rho']
        p = vals['p']
        sig = "*" if p < 0.05 else ""
        print(f"   {pair[0]} ({DIM_NAMES[pair[0]]}) vs {pair[1]} ({DIM_NAMES[pair[1]]}): rho={rho:.3f}{sig}, p={p:.3f}")

    print("\n   All significant correlations (p < 0.05):")
    sig_corrs = [(k, v) for k, v in spearman.items() if v['p'] < 0.05]
    sig_corrs.sort(key=lambda x: -abs(x[1]['rho']))
    for (d1, d2), vals in sig_corrs:
        print(f"   {d1}-{d2}: rho={vals['rho']:.3f}, p={vals['p']:.3f}")

    # Generate table output
    print("\n" + "=" * 70)
    print("TABLE OUTPUT")
    print("=" * 70)

    print("\n--- Distribution Table ---")
    print(generate_distribution_table(entropy))

    print("\n--- Entropy Table ---")
    print(generate_entropy_table(entropy))

    print("\n--- Cramer's V Table ---")
    print(generate_cramers_table(cramers))

    print("\n--- Spearman Table ---")
    print(generate_spearman_table(spearman))

    # Summary for paper text
    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER TEXT")
    print("=" * 70)
    print(f"""
Key findings for Section 5.4:
- Profile uniqueness: {unique}/{total} ({pct:.0f}%) unique profiles
- Minimum Hamming distance: {min_dist} dimensions (between {min_pair[0]} and {min_pair[1]})
- Lowest entropy: D4 (Coordination) with H={entropy['D4']['entropy']:.3f} bits ({entropy['D4']['normalized']*100:.0f}%)
- Highest entropy: D5 (Autonomy) with H={entropy['D5']['entropy']:.3f} bits ({entropy['D5']['normalized']*100:.0f}%)
- D8-D9 Cramer's V: {cramers[('D8','D9')]['V']:.3f} (p={cramers[('D8','D9')]['p']:.3f})
- D5-D8 Spearman: rho={spearman[('D5','D8')]['rho']:.3f} (p={spearman[('D5','D8')]['p']:.3f})
- D5-D9 Spearman: rho={spearman[('D5','D9')]['rho']:.3f} (p={spearman[('D5','D9')]['p']:.3f})
""")


if __name__ == "__main__":
    main()
