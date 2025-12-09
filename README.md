# CMV Taxonomy for Agentic AI Systems

A comprehensive research framework for classifying and evaluating agentic AI systems using a **Capability-Maturity-Value (CMV) Taxonomy**.

## Overview

This project develops a taxonomy that addresses a critical gap in existing frameworks by explicitly integrating:
- **Capability** - What AI systems can technically do
- **Maturity** - Implementation maturity levels
- **Value** - Organizational value realization

The taxonomy enables researchers to systematically classify agentic AI systems, supports practitioners in making informed design decisions, and equips enterprise leaders with structured approaches for portfolio assessment.

## Project Structure

```
ERP1/
├── ERP1_Artifact/           # Research documents and data
│   ├── 1. Foundation–*.md   # Phase 1: Scope and meta-characteristics
│   ├── 2. *.md              # Phase 2: Capability extraction
│   ├── 3.1-3.2*.md          # Phase 3: Taxonomy construction
│   ├── 4. *.md              # Phase 4: Taxonomy refinement
│   ├── 5. *.md              # Phase 5: Maturity level definitions
│   ├── 6. *.md              # Phase 6: CMV framework integration
│   └── *.jsonl              # Supporting data files
│
├── scripts/
│   ├── taxonomy_quality_metrics.py   # Quality validation script
│   ├── characteristic_heatmap.py     # Characteristic distribution visualization
│   └── characteristic_heatmap.png    # Generated heatmap output
│
└── AgenticAI_Capability_Maturity_Value (CMV)_Taxonomy_V0.6.pdf  # Research paper
```

## Taxonomy Dimensions

The taxonomy consists of 9 capability dimensions:

| Dimension | Description |
|-----------|-------------|
| D1: Reasoning Sophistication | Internal cognitive processes |
| D2: Tool Integration Depth | Ability to discover, select, and invoke tools |
| D3: Memory Persistence | Temporal scope of information retention |
| D4: Multi-Agent Coordination | Collaboration mechanisms |
| D5: Autonomy Level | Degree of self-direction |
| D6: Adaptation Capability | Responsiveness to feedback and change |
| D7: Human-Agent Collaboration | Integration with human teams |
| D8: Compliance & Governance | Adherence to constraints |
| D9: Safety Assurance | Hazard mitigation and safeguards |

## Reference Objects

The taxonomy is validated against 25 reference objects including:
- Concrete implementations (ReAct Agent, AutoGPT, BabyAGI, CAMEL, MetaGPT)
- Architectural patterns (role-based cooperation, debate-based systems)
- Application domains (AIOps, software engineering, ERP systems)

## Maturity Levels

Five maturity levels based on CMM methodology:
1. **Initial** - Ad-hoc, unpredictable
2. **Developing** - Basic processes established
3. **Defined** - Standardized processes
4. **Managed** - Measured and controlled
5. **Optimizing** - Continuous improvement

## Value Categories

Seven value categories identified:
- Productivity Gain (40%)
- Decision Quality Enhancement (24%)
- Risk Mitigation (16%)
- Revenue Enhancement (8%)
- Innovation Enablement (8%)
- Safety Assurance (4%)

## Installation

### Prerequisites
- Python 3.7+
- pip or conda

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd ERP1

# Install dependencies
pip install numpy pandas scipy matplotlib seaborn
```

## Usage

### Running Quality Metrics Analysis

```bash
python scripts/taxonomy_quality_metrics.py
```

This generates:
- Profile uniqueness analysis (100% unique across 25 objects)
- Distribution entropy metrics
- Dimension independence analysis (Cramer's V)
- Spearman correlation analysis
- Summary statistics

### Generating Characteristic Distribution Heatmap

```bash
python scripts/characteristic_heatmap.py
```

This generates a visualization showing characteristic distribution across all 25 reference objects:
- Heatmap of 25 objects × 9 dimensions
- Overall distribution analysis
- Per-dimension characteristic breakdown
- Output saved to `scripts/characteristic_heatmap.png`

**Key Finding:** The distribution validates that the taxonomy captures the actual state of the field:

| Level | Description | Count | Percentage |
|-------|-------------|-------|------------|
| C_.1 | Basic | 53 | 23.6% |
| C_.2 | Intermediate | 90 | **40.0%** |
| C_.3 | Advanced | 59 | 26.2% |
| C_.4 | Sophisticated | 23 | **10.2%** |

Most systems cluster at intermediate levels (C_.2), with advanced characteristics (C_.4) remaining relatively rare.

## Research Phases

1. **Foundation** - Define scope, meta-characteristics, and 25 reference objects
2. **Capability Extraction** - Extract capability families from corpus analysis
3. **Taxonomy Construction** - Iterative development of 9 dimensions
4. **Taxonomy Refinement** - Final dimension and characteristic refinement
5. **Maturity Levels** - Define 5-level maturity model per dimension
6. **CMV Integration** - Integrate all components into final framework

## Data Files

JSONL files in `ERP1_Artifact/` contain:
- Reference objects table
- Value linkage specifications
- Dimension definitions
- Characteristics table
- Object classification matrix
- Maturity specifications
