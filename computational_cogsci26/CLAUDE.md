# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CogSci 2026 paper: "How the Teaching Style and Interpretation Type of State Interventions Shape Multi-Agent Coordination." This repository contains simulation data, analysis scripts, plotting scripts, and the LaTeX paper manuscript.

## Running Scripts

All Python scripts run from the repository root (they use relative paths to data directories):

```bash
python statistical_analysis.py        # Full statistical analysis (ANOVA, t-tests, effect sizes)
python 2agent_plot_creation.py         # Generate 2-agent figures
python 3agent_plot_creation.py         # Generate 3-agent figures
```

Dependencies: pandas, numpy, scipy, matplotlib, statsmodels, seaborn

## Architecture

### Experimental Design

- **Teaching Styles** (mode 1-4): Undoing, Correcting, Exploration-encouraging Correcting, Restart
- **Interpretation Types** (type 0-5): Suggestion, Reset, Interrupt, Transition, Disrupt, Impede
- **Intervention Rates**: 0, 0.25, 0.5, 0.75, 1.0
- **Scenarios**: 2-agent (world3_size_12) and 3-agent (world3_size_12_3agent)
- **50 runs per condition**

### Data Layout

- `world3_size_12/` — 2-agent experiment results
- `world3_size_12_3agent/` — 3-agent experiment results
- Subdirectory naming: `rate_0_mode_{1-4}` (baseline) or `type_{0-5}_rate_{0.25-1.0}_mode_{1-4}` (intervention)
- Per-run files: `run_{N}.csv` (agent Q-values/rewards per step), `visit_stats_{N}.csv` (grid visit counts per agent)

### CSV Schema

**run_N.csv**: `agentid, step, ExpectedQvalue, CumulativeReward, center1_q, center2_q`

**visit_stats_N.csv**: `grid_x, grid_y, agent_0, agent_1[, agent_2]` (visit counts per cell)

### Key Metrics

- **Policy Divergence**: |agent0_preference - agent1_preference| where preference = center1_q - center2_q (higher = more specialized)
- **System Equilibrium**: |CumulativeReward_agent0 - CumulativeReward_agent1| (lower = more balanced)
- **Region Attachment** (3-agent): visit counts to each agent's primary region

### Paper

- `cogsci_computing_exp1_in_supp.tex` — main LaTeX manuscript (CogSci proceedings format)
- `figs/` — figures referenced in the paper
- `cogsci2026_computation.pdf` — compiled paper

## Plotting Conventions

Plot scripts assign agents to regions based on visit_stats (agent with more visits to patch1 = "group 1"). The `_fixed` and `_improved` variants are iterative refinements of the same plots. Key metrics plotted over training steps: `reward_diff` (cumulative reward gap) and `q_diff_diff` (Q-value preference divergence between agents).
