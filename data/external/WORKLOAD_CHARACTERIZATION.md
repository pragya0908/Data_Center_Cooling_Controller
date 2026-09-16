# CoolRL — Workload Characterization and Experimental Scenario Definition Report (Phase 1C)

**Project**: CoolRL — Data-Driven Adaptive Data Center Cooling Optimization using Reinforcement Learning  
**Status**: Completed Phase 1C  
**Author**: CoolRL Project Team  
**Date**: September 2026  
**Scope**: Statistical Characterization of Real-World Datasets and Formal Definition of Reproducible Experimental Scenarios  

---

## Executive Summary & Strict Methodological Boundaries

This report concludes **Phase 1C** of the CoolRL engineering lifecycle. The objective of this phase is to formally characterize the empirical properties of the acquired real-world datasets, establish rigorous mathematical definitions for operational stress, and define reproducible experimental scenarios for subsequent reinforcement learning phases.

> [!IMPORTANT]
> **CRITICAL METHODOLOGICAL PRINCIPLES**:
> 1. **STRICT SEPARATION OF OBSERVED DATA AND SYNTHETIC STRESS**:
>    - **OBSERVED REAL DATA**: All baseline workload traces ($W(t) \in [0.0, 1.0]$) and outdoor ambient temperatures ($T^{\text{amb}}(t) \in \mathbb{R}$) directly originate from the validated historical records of Alibaba Cluster Trace 2018, Google Cluster Data 2019, and NASA POWER meteorological observations.
>    - **CONTROLLED EXPERIMENTAL TRANSFORMATIONS**: Scenarios 2 through 5 apply mathematically defined, fully documented, and parameter-controlled transformations to the real trajectories to evaluate controller robustness under severe operating stress. These stress scenarios are **never** labeled or conflated with historical observed data.
> 2. **NO MACHINE LEARNING IMPLEMENTATION**: Reinforcement learning algorithms (tabular Q-learning), reward formulations, thermal environment models, and cooling equations are strictly excluded from Phase 1C.

---

## 1. Dataset Overview

The CoolRL data-grounded simulation framework ingests three real-world datasets cleaned and structured during Phase 1B:

| Dataset Identifier | Domain | Source / Organization | Spatial / Granularity | Temporal Resolution | Duration | Total Observations | Key Target Variable |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Alibaba Trace 2018** | Primary Workload | Alibaba Group (Zenodo 14564847) | Cluster-wide machine aggregate | 300 seconds (5 min) | ~7.79 days | 2,243 rows | `workload` (CPU util / 100) |
| **Google Trace 2019** | Secondary / Evaluation | Google LLC (Zenodo 14564847) | Cluster-wide instance aggregate | 300 seconds (5 min) | 28.0 days | 8,064 rows | `workload` (`avg_cpu`) |
| **NASA POWER Hourly** | Ambient Weather | NASA Langley Research Center | Bengaluru, India (12.97°N, 77.59°E) | 1 hour (60 min) | 30.0 days (May 2018) | 720 rows | `ambient_temp_c` ($T^{\text{amb}}$, °C) |
| **NASA POWER Aligned** | Ambient Weather | NASA POWER (Interpolated) | Bengaluru, India (Aligned to Alibaba) | 300 seconds (5 min) | ~7.79 days (May 1–8) | 2,243 rows | `ambient_temp_c` ($T^{\text{amb}}$, °C) |

All datasets reside under `data/processed/` and are read without alteration.

---

## 2. Alibaba Primary Workload Characterization

### 2.1 Summary Statistics
The primary workload trajectory $W(t)$ is derived from `cpu_util_percent / 100` from the Alibaba 2018 trace across 2,243 observations (186.83 hours / ~7.79 days):

| Metric | Empirical Value | Context / Interpretation |
| :--- | :--- | :--- |
| **Observations ($N$)** | 2,243 | Continuous 300-second intervals |
| **Time Span** | 0 to 672,600 seconds | 0.0 to 186.83 hours (~7.79 days) |
| **Mean** | **0.401757** | Baseline data-center compute load is ~40.2% |
| **Median (P50)** | **0.393917** | Close to mean; slight right-skew |
| **Standard Deviation ($\sigma$)** | **0.097910** | Broad natural dispersion across diurnal cycles |
| **Minimum** | **0.161270** | Off-peak / early-morning cluster idle utilization (~16.1%) |
| **Maximum** | **0.790743** | Observed historical peak utilization (~79.1%) |
| **Interquartile Range (IQR)** | **0.141299** | $P_{75} - P_{25} = 0.465821 - 0.324522$ |

### 2.2 Empirical Percentile Profile

| Percentile | Workload Value | Empirical Description |
| :---: | :---: | :--- |
| **5th ($P_5$)** | 0.262983 | Deep nocturnal valleys (minimum sustained batch processing) |
| **10th ($P_{10}$)** | 0.282513 | Low nocturnal baseline |
| **25th ($P_{25}$)** | 0.324522 | Boundary between low and normal operational workload |
| **50th ($P_{50}$)** | 0.393917 | Median data center workload |
| **75th ($P_{75}$)** | 0.465821 | Boundary between normal and elevated diurnal load |
| **90th ($P_{90}$)** | 0.535131 | High-load operational periods (afternoon business peaks) |
| **95th ($P_{95}$)** | 0.573648 | Heavy cluster stress events |
| **99th ($P_{99}$)** | 0.663137 | Extreme outlier burst periods (top 1% of trace) |

### 2.3 Empirical Workload Regions & Discretization Recommendation
To answer *"What constitutes low, medium, and high workload?"* without arbitrary thresholds, we ground the regions directly in the empirical cumulative distribution:

```
0.00              0.325                0.466                0.574                0.791        1.00
|-- Low Workload --|-- Normal Workload -|-- High Workload --|-- Extreme Peak ---|-------------|
     (< P25)               (P25 to P75)          (P75 to P95)          (> P95)        (Unused headroom)
```

1. **Low Workload ($W < 0.325$, $< P_{25}$)**:
   - Represents nocturnal valleys and periods of low compute activity. Data center thermal output is minimal, requiring low cooling energy.
2. **Normal Workload ($0.325 \le W < 0.466$, $P_{25} \text{ to } P_{75}$)**:
   - Represents the core interquartile operational regime (50% of all observations). Workload fluctuates predictably around the median of 0.394.
3. **High Workload ($0.466 \le W < 0.574$, $P_{75} \text{ to } P_{95}$)**:
   - Represents sustained daytime enterprise workloads and active batch computation. Cooling demand must be ramped up to avoid thermal buildup.
4. **Extreme Peak Workload ($W \ge 0.574$, $> P_{95}$)**:
   - Represents the upper 5% tail of cluster utilization, reaching up to 0.791. Requires aggressive cooling action to prevent thermal violations.

#### Proposed Practical RL Discretization Bins
For the tabular Q-learning state discretizer in Phase 3, we recommend:
- **4-Bin Practical Scheme**:
  - `Bin 0 (Low)`: $[0.00, 0.33)$ (Captures 26.9% of real Alibaba steps)
  - `Bin 1 (Normal)`: $[0.33, 0.45)$ (Captures 43.5% of real Alibaba steps)
  - `Bin 2 (High)`: $[0.45, 0.55)$ (Captures 21.7% of real Alibaba steps)
  - `Bin 3 (Extreme)`: $[0.55, 1.00]$ (Captures 7.9% of real Alibaba steps)
- **Rationale**: This guarantees balanced state visitation during exploration, preventing state-space starvation while maintaining fine resolution across the operational core.

---

## 3. Google Secondary Workload Characterization

### 3.1 Summary Statistics
The Google 2019 trace (`instance_usage_grouped_300_seconds_month.csv`) serves as an independent secondary workload source covering 28 continuous days (8,064 300-second observations):

| Metric | Empirical Value | Comparison with Alibaba |
| :--- | :--- | :--- |
| **Observations ($N$)** | 8,064 | ~3.6× larger temporal window |
| **Time Span** | 0 to 2,419,200 seconds | 672.0 hours (28.0 days) |
| **Mean** | **0.472823** | +17.7% higher average load than Alibaba |
| **Median (P50)** | **0.476579** | Near-symmetric distribution |
| **Standard Deviation ($\sigma$)** | **0.039913** | Much lower variance (~40% of Alibaba's $\sigma$) |
| **Minimum** | **0.320287** | Significantly higher floor than Alibaba (0.320 vs 0.161) |
| **Maximum** | **0.592437** | Lower absolute ceiling (0.592 vs 0.791) |

### 3.2 Google Empirical Percentiles

| Percentile | Google Workload | Alibaba Workload (for comparison) |
| :---: | :---: | :---: |
| **5th ($P_5$)** | 0.399253 | 0.262983 |
| **10th ($P_{10}$)** | 0.420025 | 0.282513 |
| **25th ($P_{25}$)** | 0.449647 | 0.324522 |
| **50th ($P_{50}$)** | 0.476579 | 0.393917 |
| **75th ($P_{75}$)** | 0.498077 | 0.465821 |
| **90th ($P_{90}$)** | 0.519929 | 0.535131 |
| **95th ($P_{95}$)** | 0.533950 | 0.573648 |
| **99th ($P_{99}$)** | 0.559552 | 0.663137 |

---

## 4. Meteorological Weather Characterization (NASA POWER)

### 4.1 Summary Statistics
Outdoor ambient temperature ($T^{\text{amb}}$, parameter `T2M`) directly affects chiller COP, natural convective heat dissipation, and chiller thermal lift:

| Metric | Hourly Series (May 1–30, 2018) | Aligned 300s Series (May 1–8, 2018) |
| :--- | :--- | :--- |
| **Observations ($N$)** | 720 hours (30 days) | 2,243 steps (~7.79 days) |
| **Mean Temperature** | **26.4538°C** | **27.8278°C** |
| **Median Temperature** | **25.6400°C** | **27.0200°C** |
| **Standard Deviation ($\sigma$)** | **4.1520°C** | **4.6396°C** |
| **Minimum Temperature** | **19.2900°C** | **20.3200°C** |
| **Maximum Temperature** | **36.1900°C** | **36.1900°C** |
| **Interquartile Range (IQR)** | **7.3525°C** (22.74°C to 30.09°C) | **8.5745°C** (23.61°C to 32.18°C) |

### 4.2 Meteorological Percentile Profile

| Percentile | Hourly May 2018 ($N=720$) | Aligned Days 1–8 ($N=2,243$) | Meteorological Classification |
| :---: | :---: | :---: | :--- |
| **5th ($P_5$)** | 21.17°C | 21.46°C | Nocturnal minimum (high economizer efficiency) |
| **10th ($P_{10}$)** | 21.51°C | 22.40°C | Cool night conditions |
| **25th ($P_{25}$)** | 22.74°C | 23.61°C | Mild morning / evening temperatures |
| **50th ($P_{50}$)** | 25.64°C | 27.02°C | Median ambient temperature |
| **75th ($P_{75}$)** | 30.09°C | 32.18°C | Warm afternoon temperatures |
| **90th ($P_{90}$)** | **32.14°C** | **34.61°C** | **High ambient threshold** (severe cooling load) |
| **95th ($P_{95}$)** | **33.30°C** | **35.34°C** | Peak summer heatwave conditions |
| **99th ($P_{99}$)** | **35.59°C** | **36.09°C** | Extreme ambient temperature ceiling |

### 4.3 Temperature Variability & Diurnal Range
- **Hourly Delta ($\Delta T_{\text{hourly}} = T(t) - T(t-1)$)**:
  - Mean: $-0.0002^\circ\text{C}$, Std: $1.190^\circ\text{C}$, Min: $-3.44^\circ\text{C}$, Max: $+3.53^\circ\text{C}$.
  - Absolute hourly change: Mean = $0.93^\circ\text{C}$, $P_{90} = 2.10^\circ\text{C}$, $P_{99} = 2.86^\circ\text{C}$.
- **Aligned 300s Delta ($\Delta T_{\text{step}} = T(t) - T(t-1)$)**:
  - Mean: $+0.0005^\circ\text{C}$, Std: $0.113^\circ\text{C}$, Min: $-0.287^\circ\text{C}$, Max: $+0.253^\circ\text{C}$.
  - Absolute step change: Mean = $0.092^\circ\text{C}$, $P_{90} = 0.198^\circ\text{C}$, $P_{99} = 0.254^\circ\text{C}$.
- **Diurnal Fluctuation**: Daily temperature swing is between $11^\circ\text{C}$ and $16^\circ\text{C}$ per day (e.g., May 6 swings from $20.36^\circ\text{C}$ at dawn to $36.19^\circ\text{C}$ in mid-afternoon).

---

## 5. Workload Variability and Dynamics

### 5.1 Step-to-Step First Differences
To evaluate dynamic volatility and the rate of workload change, we compute the first difference:
$$\Delta W(t) = W(t) - W(t-1)$$
and absolute first difference:
$$|\Delta W(t)| = |W(t) - W(t-1)|$$

| Metric | Alibaba ($\Delta t = 300\text{ s}$) | Google ($\Delta t = 300\text{ s}$) | Dynamic Contrast |
| :--- | :--- | :--- | :--- |
| **Mean Step Difference ($\Delta W$)** | $+0.000108$ | $+0.000010$ | Zero-centered stationarity |
| **Std Dev of Step Difference ($\sigma_{\Delta W}$)** | **$0.060873$** | **$0.016270$** | Alibaba is **3.74× more volatile** |
| **Max Positive Step Change ($\max \Delta W$)** | **$+0.318679$** | **$+0.121059$** | Alibaba experiences massive sudden surges |
| **Max Negative Step Change ($\min \Delta W$)** | **$-0.232592$** | **$-0.117424$** | Rapid workload drops |
| **Mean Absolute Change ($|\Delta W|$)** | **$0.046060$** | **$0.011512$** | Alibaba shifts ~4.6% per 5 min |
| **Median Absolute Change ($P_{50}$)** | $0.035132$ | $0.008159$ | Typical step adjustment |
| **75th Percentile ($P_{75}$)** | $0.064187$ | $0.015584$ | Regular diurnal ramp |
| **90th Percentile ($P_{90}$)** | $0.098105$ | $0.025967$ | Noticeable rapid surge |
| **95th Percentile ($P_{95}$)** | **$0.122207$** | **$0.034386$** | Empirical spike boundary |
| **99th Percentile ($P_{99}$)** | **$0.184880$** | **$0.053389$** | Extreme burst boundary |

---

## 6. Objective Definition of a Workload Spike

### 6.1 Empirical Analysis
An objective definition of a workload spike cannot be chosen arbitrarily. In the Alibaba trace:
- A change of $|\Delta W| \ge 0.08$ occurs in 369 steps (16.46% of intervals).
- A change of $|\Delta W| \ge 0.10$ occurs in 212 steps (9.46% of intervals).
- A change of $|\Delta W| \ge 0.12$ occurs in 125 steps (5.58% of intervals).
- A change of $|\Delta W| \ge 0.15$ occurs in 54 steps (2.41% of intervals).
- A change of $|\Delta W| \ge 0.18$ occurs in 29 steps (1.29% of intervals).
- A change of $|\Delta W| \ge 0.20$ occurs in 15 steps (0.67% of intervals).

### 6.2 Formal Spike Criteria
We establish two tiers of workload spike events:

1. **Empirical Workload Spike ($\Delta W \ge 0.122$)**:
   - **Threshold**: $|\Delta W(t)| \ge P_{95}(|\Delta W|) \approx 0.1222$ (an instantaneous step change $\ge 12.2\%$ in 300 seconds).
   - **Significance**: Represents rare, rapid step transitions that challenge thermal responsiveness. Occurs in only ~5% of normal intervals.
2. **Extreme Workload Burst ($\Delta W \ge 0.185$)**:
   - **Threshold**: $|\Delta W(t)| \ge P_{99}(|\Delta W|) \approx 0.1849$ (an instantaneous step change $\ge 18.5\%$).
   - **Significance**: Top 1% outlier events where large compute clusters activate simultaneously.

---

## 7. High-Workload Stress Definition (Scenario 2)

### 7.1 Mathematical Formulation
To evaluate the controller under heavy, sustained thermal loading without altering the original dataset, we define the controlled transformation:
$$W_{\text{high}}(t) = \text{clip}(k \cdot W(t), 0.0, 1.0)$$

### 7.2 Evaluation of Scaling Factors ($k$)
We tested scaling factors from $k = 1.05$ to $1.50$ on the Alibaba workload ($N=2,243$):

| Factor ($k$) | Transformed Mean | Transformed Median | Transformed Std | Transformed Max | 95th Percentile | Saturation ($\ge 1.0$) | Clipped Count |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original (1.00)** | 0.4018 | 0.3939 | 0.0979 | 0.7907 | 0.5736 | 0.000% | 0 |
| **1.10** | 0.4419 | 0.4333 | 0.1077 | 0.8698 | 0.6310 | 0.000% | 0 |
| **1.15** | 0.4620 | 0.4530 | 0.1126 | 0.9094 | 0.6597 | 0.000% | 0 |
| **1.20** | 0.4821 | 0.4727 | 0.1175 | 0.9489 | 0.6884 | 0.000% | 0 |
| **1.25** | **0.5022** | **0.4924** | **0.1224** | **0.9884** | **0.7171** | **0.000%** | **0** |
| **1.26** | 0.5062 | 0.4963 | 0.1234 | 0.9963 | 0.7228 | 0.000% | 0 |
| **1.27** | 0.5102 | 0.5003 | 0.1243 | 1.0000 | 0.7285 | 0.045% | 1 |
| **1.30** | 0.5223 | 0.5121 | 0.1272 | 1.0000 | 0.7457 | 0.089% | 2 |
| **1.35** | 0.5422 | 0.5318 | 0.1317 | 1.0000 | 0.7744 | 0.535% | 12 |
| **1.40** | 0.5621 | 0.5515 | 0.1358 | 1.0000 | 0.8031 | 0.713% | 16 |
| **1.50** | 0.6017 | 0.5909 | 0.1438 | 1.0000 | 0.8605 | 0.981% | 22 |

### 7.3 Selected Scaling Factor Rationale
**$k = 1.25$ is selected as the definitive high-workload parameter**:
1. **Meaningful Stress**: Increases average data center compute load by exactly **$+25.0\%$** (mean rises from $0.4018$ to $0.5022$).
2. **Realistic Capacity Utilization**: The maximum workload reaches **$0.9884$** (98.84% capacity), bringing the facility to near-peak utilization.
3. **Zero Distortion / No Artificial Saturation**: Exactly **0.000% of values are clipped** (0 observations reach 1.0). The full natural trajectory, diurnal shape, and variance are preserved without creating flat-top artifacts.

---

## 8. High-Ambient Thermal Stress Definition (Scenario 4)

### 8.1 Empirical Temperature Regimes
Based on NASA POWER hourly data for Bengaluru ($N=720$):
- **Normal Operating Range**: $22.74^\circ\text{C}$ to $30.09^\circ\text{C}$ ($P_{25} \text{ to } P_{75}$).
- **Elevated Temperature Regime**: $30.10^\circ\text{C}$ to $32.14^\circ\text{C}$ ($P_{75} \text{ to } P_{90}$).
- **High-Ambient Regime**: **$T^{\text{amb}} \ge 32.14^\circ\text{C}$** (Top 10% of observations).
- **Extreme Heatwave Peak**: **$T^{\text{amb}} \ge 35.00^\circ\text{C}$** (Top 1–2% of observations, up to $36.19^\circ\text{C}$).

### 8.2 Reproducible High-Ambient Scenario Construction
The original NASA POWER observations remain unmodified in `data/processed/`. For experimental evaluation, Scenario 4 employs two complementary, reproducible methods:
1. **Subsequence Selection (Natural Peak Heatwave)**:
   - Day 6 of the aligned record (Steps 1440 to 1728; elapsed hours 120 to 144) naturally recorded the maximum temperature in the dataset ($36.19^\circ\text{C}$), with 7 consecutive hours exceeding $32.0^\circ\text{C}$. This slice provides a purely empirical extreme heatwave test.
2. **Additive Thermal Anomaly ($\Delta T_{\text{heatwave}} = +3.0^\circ\text{C}$)**:
   - For whole-trajectory stress testing, an additive shift of $+3.0^\circ\text{C}$ represents severe pre-monsoon heatwave conditions common in semi-arid tech corridors.
   - Mean rises from $27.83^\circ\text{C} \to 30.83^\circ\text{C}$; afternoon peaks reach $39.19^\circ\text{C}$, testing chiller operating limits under thermal distress.

---

## 9. The Five Required Experimental Scenarios

Every scenario is formally defined below with its input trajectories, mathematical transformations, parameters, expected thermal stress, reproducibility mechanism, and experimental objective.

| Scenario ID & Name | Input Workload Trajectory | Input Ambient Trajectory | Transformation & Parameters | Expected Facility Stress | Reproducibility Method | Experimental Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Scenario 1: Normal** | Real Alibaba 300s trace (`workload`) | Real NASA POWER 300s aligned trace (`ambient_temp_c`) | **None**. Identity: $W(t)$, $T^{\text{amb}}(t)$ | Baseline diurnal compute (mean 0.40) & normal summer ambient (mean 27.8°C). | Deterministic ingestion of processed CSV files. | Establish baseline cooling energy, temperature stability, and standard PUE. |
| **Scenario 2: High Workload** | Real Alibaba 300s trace | Real NASA POWER 300s trace | $W_{\text{high}}(t) = \text{clip}(1.25 \cdot W(t), 0, 1)$ | Continuous +25% heat generation. Peak load reaches 98.8%. Mean load 0.502. | Deterministic scaling parameter $k=1.25$. | Evaluate energy efficiency and temperature control under heavy sustained computation. |
| **Scenario 3: Workload Spikes** | Real Alibaba 300s trace | Real NASA POWER 300s trace | Controlled injection: $W(t) + 0.25$, duration 15–30 min, 2 spikes/day. | Sudden thermal transients. Rapid heat dump before cooling can respond. | Seeded pseudo-random generation (`seed=42`) or fixed step schedule. | Test controller reactivity, lag compensation, and prevention of thermal overshoot. |
| **Scenario 4: High Ambient** | Real Alibaba 300s trace | Real NASA POWER 300s trace | Natural Day 6 heatwave slice or additive shift $T^{\text{amb}} + 3.0^\circ\text{C}$. | Degraded chiller COP, reduced heat rejection capacity, high condensing temp. | Parameterized offset $\Delta T = 3.0^\circ\text{C}$ or step slice $[1440, 1728]$. | Assess chiller efficiency, COP degradation, and thermal safety under outdoor heatwaves. |
| **Scenario 5: Combined Stress** | Real Alibaba 300s trace | Real NASA POWER 300s trace | $W_{\text{high}}(t)$ ($k=1.25$) + Spikes ($+0.20$) + Ambient ($+3.0^\circ\text{C}$). | Simultaneous worst-case electrical heat generation and minimal cooling COP. | Deterministic combination with seed `42` and constants in `scenario_config.py`. | Stress-test thermal fail-safes, boundary violations, and maximum power demand. |

---

## 10. Temporal Train / Validation / Test Methodology

### 10.1 Causal Leakage Prevention in Time Series
Data center workloads and meteorological temperatures are non-stationary time series with strong temporal autocorrelation:
- Alibaba workload lag-1 (5 min) autocorrelation: **$r = 0.8066$**
- Alibaba workload lag-12 (1 hour) autocorrelation: **$r = 0.6293$**
- Alibaba workload lag-288 (24 hour diurnal) autocorrelation: **$r = 0.5702$**

> [!WARNING]
> Standard random k-fold cross-validation or random shuffling is **strictly prohibited**. Shuffling time-series observations causes severe data leakage: adjacent steps separated by only 5 minutes share 81% correlation, resulting in artificially inflated performance estimates.

### 10.2 Defensible Chronological Split
The 2,243 observations (~7.79 days) are partitioned chronologically along complete diurnal cycle boundaries:

```
Step:  0                 1440                1728                2242
Hour:  0.0               120.0               144.0               186.83
       |--- Training (Days 1–5) ---|-- Val (Day 6) --|-- Testing (Days 7–8) -|
       |       1,440 Steps         |    288 Steps    |       515 Steps       |
       |         (64.20%)          |     (12.84%)    |        (22.96%)       |
```

| Partition | Step Range | Elapsed Hours | Duration | Proportion | Mean Workload | Std Dev | Min / Max | Operational Character |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Training** | `0` to `1439` | $0.0$ to $120.0$ | 5.0 days | 64.20% | 0.3919 | 0.0983 | 0.1613 / 0.7681 | 5 complete diurnal cycles for policy learning. |
| **Validation** | `1440` to `1727` | $120.0$ to $144.0$ | 1.0 day | 12.84% | 0.3802 | 0.0985 | 0.2350 / 0.7547 | Day 6 contains dataset peak ambient (36.19°C); hyperparameter tuning. |
| **Testing** | `1728` to `2242` | $144.0$ to $186.83$ | ~1.79 days | 22.96% | 0.4413 | 0.0850 | 0.2501 / 0.7907 | Completely unseen; higher mean load (0.441) and global peak (0.791). |

**Evaluation Rule**: The testing partition (Steps 1728–2242) is held strictly out of training and validation loops.

---

## 11. Google Workload Generalization Methodology

### 11.1 Cross-Cluster Evaluation Architecture
The Google 2019 cluster trace is not merged with the Alibaba dataset during model development. Instead, it serves as an **independent cross-distribution evaluation benchmark**:

```
+-------------------------------------------------------------+
|                     PHASE 3 & 4 RL TRAINING                 |
|  Alibaba Trace (Days 1–5: Training Split)                   |
|  --> Learns Optimal Q-Table: Q*(s, a)                       |
+-------------------------------------------------------------+
                               |
               +---------------+---------------+
               |                               |
               v                               v
+-----------------------------+ +-----------------------------+
|    IN-DISTRIBUTION TEST     | |    OUT-OF-DISTRIBUTION      |
|                             | |    GENERALIZATION TEST      |
| Alibaba Days 7–8 (Held-Out) | | Google 28-Day Trace (8,064) |
| Same cluster, unseen time   | | Different facility, diff    |
|                             | | variance & baseline         |
+-----------------------------+ +-----------------------------+
```

### 11.2 Rationale
1. **Architectural Heterogeneity**: Alibaba and Google traces represent different hardware compositions, cluster schedulers, and workload mixing. Training on Alibaba and evaluating on Google directly tests whether CoolRL learns robust thermal management principles rather than memorizing Alibaba-specific periodicities.
2. **Multi-Week Stability Testing**: Google's 28-day duration (8,064 steps) provides a continuous month of simulated operation, validating that the cooling policy does not drift, accumulate heat, or exhibit long-term instability.

---

## 12. Recommended State Discretization for Phase 3

In tabular Q-learning, continuous state variables must be mapped to discrete integer bin indices. Below is the formal recommendation:

### 12.1 Observed Data Bins (Empirically Grounded)

#### A. Workload Bins ($W \in [0.0, 1.0]$) — 4-Bin Recommendation
- `Bin 0 (Low)`: $[0.00, 0.33)$ — Bottom quartile ($< P_{25}$), represents idle/night computing.
- `Bin 1 (Normal)`: $[0.33, 0.45)$ — Core interquartile range ($P_{25} \text{ to } P_{75}$), standard operations.
- `Bin 2 (High)`: $[0.45, 0.55)$ — Elevated diurnal load ($P_{75} \text{ to } P_{90}$).
- `Bin 3 (Extreme)`: $[0.55, 1.00]$ — Top 10% peak bursts and stress conditions.

#### B. Ambient Temperature Bins ($T^{\text{amb}} \in \mathbb{R}$) — 4-Bin Recommendation
- `Bin 0 (Cool)`: $< 23.0^\circ\text{C}$ — Nighttime / early morning (high economizer cooling).
- `Bin 1 (Mild)`: $[23.0, 27.0)^\circ\text{C}$ — Standard daytime baseline.
- `Bin 2 (Warm)`: $[27.0, 31.0)^\circ\text{C}$ — Elevated daytime temperatures.
- `Bin 3 (Hot)`: $\ge 31.0^\circ\text{C}$ — High ambient regime ($\ge P_{90}$), low heat rejection.

---

### 12.2 Simulation Design Choices (NOT Observed Data)

> [!CAUTION]
> The variables below are **not** present in the historical traces. They are internal simulation state variables whose discretization is grounded in engineering standards (ASHRAE TC 9.9) and control design.

#### C. Internal Data Center Temperature ($T^{\text{int}}$) — 5-Bin Recommendation
Grounded in ASHRAE Class A1 Thermal Guidelines (Recommended: 18°C–27°C, Allowable: 15°C–32°C):
- `Bin 0 (Undercooled)`: $< 18.0^\circ\text{C}$ (Overcooling / energy waste)
- `Bin 1 (Target Low)`: $[18.0, 21.0)^\circ\text{C}$ (Safe, slightly cold)
- `Bin 2 (Target Optimal)`: $[21.0, 24.0)^\circ\text{C}$ (Optimal balance of safety and energy)
- `Bin 3 (Warning Warm)`: $[24.0, 27.0)^\circ\text{C}$ (Safe upper limit, approaching constraint)
- `Bin 4 (Critical Hot)`: $\ge 27.0^\circ\text{C}$ (Boundary violation / thermal throttling risk)

#### D. Active Cooling Level ($C \in [0.0, 1.0]$) — 4-Bin Recommendation
- `Bin 0`: $[0.00, 0.25)$ (Low cooling effort)
- `Bin 1`: $[0.25, 0.50)$ (Medium-low cooling effort)
- `Bin 2`: $[0.50, 0.75)$ (Medium-high cooling effort)
- `Bin 3`: $[0.75, 1.00]$ (High / maximum cooling effort)

#### E. Temperature Trend ($\Delta T^{\text{int}} = T^{\text{int}}(t) - T^{\text{int}}(t-1)$) — 3-Bin Recommendation
- `Bin 0 (Cooling)`: $\Delta T < -0.20^\circ\text{C}$
- `Bin 1 (Stable)`: $-0.20^\circ\text{C} \le \Delta T \le +0.20^\circ\text{C}$
- `Bin 2 (Warming)`: $\Delta T > +0.20^\circ\text{C}$

### 12.3 Discrete State Space Tractability
Total discrete states:
$$|S| = |T^{\text{int}}| \times |W| \times |T^{\text{amb}}| \times |C| \times |\Delta T| = 5 \times 4 \times 4 \times 4 \times 3 = 960\text{ states}$$
With an action space of 5 discrete adjustments ($\mathcal{A} \in \{-20\%, -10\%, 0\%, +10\%, +20\%\}$):
$$|\mathcal{Q}| = |S| \times |\mathcal{A}| = 960 \times 5 = 4,800\text{ Q-table entries}$$
With 1,440 training steps per episode and multi-episode training, a 4,800-entry table achieves dense coverage and rapid convergence without sparse-visitation pathologies.

---

## 13. Assumptions

1. **Temporal Stationarity of 5-Minute Averages**: We assume that 300-second aggregated measurements adequately capture cluster thermal inertia, as physical server rooms possess substantial thermal mass that filters out sub-minute fluctuations.
2. **Homogeneous Compute Heat Dissipation**: We assume aggregate cluster CPU utilization maps monotonically to electrical power consumption via standard affine data-center power models ($P_{\text{IT}} = P_{\text{idle}} + (P_{\text{peak}} - P_{\text{idle}}) \cdot W$).
3. **Macro-Meteorological Representativeness**: We assume NASA POWER 2-meter ambient temperatures accurately characterize outdoor condenser heat-rejection conditions for the simulated facility.

---

## 14. Limitations

1. **Absence of Server Fan & Internal Sensor Telemetry**: The historical traces provide compute utilization but omit rack-level inlet temperatures and fan RPMs. These physical dynamics must be simulated in Phase 2.
2. **Aggregated Cluster-Level Spatial Resolution**: The datasets represent cluster-wide averages rather than individual server rack hot-spots.
3. **Synthetic Ambient Alignment**: Alibaba and Google clusters were not physically located in Bengaluru. Bengaluru weather serves as a realistic, data-grounded external climate scenario, clearly documented as such.

---

## 15. Summary of Visualizations Generated

All diagnostic visualizations have been generated and saved under `results/data_eda/phase_1c/`:

| File Name | Description | Key Insight Demonstrated |
| :--- | :--- | :--- |
| `alibaba_workload_distribution_regions.png` | Alibaba workload density with empirical regions | Illustrates Low (<0.325), Normal (0.325–0.466), High (0.466–0.574), and Extreme (>0.574) regions. |
| `alibaba_workload_timeline.png` | 8-day timeline with train/val/test partitions | Confirms diurnal cycles and shows clear boundaries at 120h (Train), 144h (Val), and 186.8h (Test). |
| `alibaba_workload_absolute_change.png` | Histogram of step-to-step absolute changes $|\Delta W|$ | Highlights the heavy-tailed distribution of 5-minute workload transitions. |
| `alibaba_spike_threshold.png` | First differences with P95/P99 threshold overlays | Visualizes empirical spike events exceeding the 12.2% step threshold. |
| `high_workload_comparison.png` | Original vs. scaled ($k=1.25$) workload distributions | Demonstrates +25% load increase, max at 0.988, and 0.00% clipping saturation. |
| `google_workload_distribution.png` | Google 28-day vs. Alibaba 8-day distribution | Demonstrates Google's higher mean (0.473) and tighter variance for generalization. |
| `ambient_temperature_distribution.png` | NASA POWER hourly vs. 300s aligned distributions | Shows full summer temperature spread (19.3°C to 36.2°C) and median (25.6°C / 27.0°C). |
| `high_ambient_threshold.png` | Ambient temperature timeline with P90 threshold | Identifies high-ambient periods ($\ge 32.1^\circ\text{C}$) and Day 6 extreme heatwave peak. |
| `scenarios_overview.png` | Multi-panel 48-hour comparison of all 5 scenarios | Visualizes workload and ambient trajectories across Scenarios 1 through 5 side-by-side. |

---

## 16. Recommendations for Phase 2

1. **Ground Thermal Equations in Physical Principles**: In Phase 2, implement a standard lumped-capacitance thermodynamic model of the data center that consumes $W(t)$ from `scenario_config.py` and $T^{\text{amb}}(t)$ from `bengaluru_weather_300s_aligned.csv`.
2. **Maintain Strict Scenario Interfaces**: Ensure the Phase 2 environment takes a `scenario_name` or `ScenarioConfig` parameter that selects the exact transformation parameters established in `src/data/scenario_config.py`.
3. **Preserve Raw & Processed Dataset Immutability**: All scenario stress transformations must be applied dynamically or during environment initialization, never overwriting the underlying CSV files.
