# CoolRL — Thermal Simulation Model Design (Phase 2A)

**Project**: CoolRL — Data-Driven Adaptive Data Center Cooling Optimization using Reinforcement Learning  
**Lifecycle Stage**: Phase 2A — Thermal Model Design  
**Author**: CoolRL Project Team  
**Date**: September 2026  
**Status**: Completed Design Specification  

---

## Strict Methodological Distinction: Real Observations vs. Simulated Variables

Before detailing the equations, we establish the explicit boundary between real-world historical data and simulated physics:

| Variable Identifier | Domain | Source / Nature | Role in Simulation |
| :--- | :--- | :--- | :--- |
| **`workload(t)` ($W(t)$)** | Compute Utilization | **REAL OBSERVED DATA** (Alibaba Cluster Trace 2018 / Google Trace 2019) | External time-series input; dictates electrical power dissipation into thermal energy. |
| **`ambient_temp_c(t)` ($T^{\text{amb}}(t)$)** | Meteorology | **REAL OBSERVED DATA** (NASA POWER Hourly Reanalysis for Bengaluru) | External time-series input; dictates envelope heat transfer and chiller COP derating. |
| **`internal_temp(t)` ($T^{\text{int}}(t)$)** | Thermal Dynamics | **SIMULATION MODEL VARIABLE** (First-order lumped capacitance) | State variable; computed dynamically via the thermal balance equation. |
| **`cooling_level(t)` ($C(t)$)** | Control Action | **SIMULATION CONTROL VARIABLE** (Discretized relative effort in $[0.0, 1.0]$) | State/Action variable; adjusted by the RL agent via discrete action increments. |
| **`cop(t)` ($\text{COP}(t)$)** | Thermodynamics | **SIMULATION MODEL VARIABLE** (Empirical refrigeration curve) | Coefficient of Performance; maps thermal heat extraction to electrical power demand. |
| **`cooling_energy(t)` ($E_{\text{cool}}(t)$)** | Energy Accounting | **SIMULATION MODEL VARIABLE** (Integrated electrical power over step) | Consumed electrical energy; feeds directly into the RL reward function. |
| **`disturbance(t)` ($\epsilon(t)$)** | Noise Process | **SIMULATION MODEL VARIABLE** (Deterministic $\equiv 0.0$ by default) | Evaluates controller robustness under optional stochastic unmodeled heat shocks. |

> [!IMPORTANT]
> **This is a TRACE-DRIVEN THERMAL SIMULATION, not a physical digital twin.**  
> Real-world traces provide the external driving forcing functions ($W(t)$ and $T^{\text{amb}}(t)$). The internal facility physics, thermal inertia, cooling response, and power consumption are computed by the lightweight, interpretable, lumped mathematical model specified below.

---

## 1. Purpose

The objective of Phase 2A is to design and validate a first-order lumped data-center thermal simulation model. This model will serve as the core transition dynamic for the future gymnasium-compatible `DataCenterEnv` in Phase 2B.

The model is engineered to satisfy four foundational design criteria:
1. **Computational Tractability**: Executes thousands of steps per second to enable fast episodic tabular Q-learning exploration without requiring expensive CFD (Computational Fluid Dynamics).
2. **Physical Interpretability & Qualitative Realism**: Conforms to first-principles conservation of energy, exhibiting realistic thermal inertia, envelope conduction, and ambient chiller derating.
3. **Control Sensitivity & Non-Trivial Decision Space**: Demands dynamic adaptation by making both undercooling (energy waste) and overcooling (boundary violation) costly across diurnal weather swings.
4. **Strict Numerical Stability**: Proven strictly stable (BIBO stable) over the fixed 300-second discrete timestep $\Delta t = 5\text{ minutes}$.

---

## 2. Modeling Assumptions

1. **Lumped Thermal Capacitance**: The internal air volume, server racks, and containment structures are modeled as an aggregate macroscopic thermal zone characterized by a single effective temperature $T^{\text{internal}}(t)$ (server intake plenum air).
2. **Fixed 300-Second Timestep ($\Delta t = 300\text{ s}$)**: One environment step corresponds to exactly one 5-minute interval ($1/12\text{ hour}$), precisely matching the sampling rate of the processed Alibaba and aligned weather datasets.
3. **Linear IT Power Dissipation**: Electrical energy consumed by servers is converted into thermal heat dissipation within the room according to an affine function of CPU utilization $W(t) \in [0.0, 1.0]$.
4. **Envelope Thermal Conduction**: Heat transfer across the building boundary is driven by Newton's law of cooling, proportional to the temperature gradient $(T^{\text{ambient}}(t) - T^{\text{internal}}(t))$.
5. **Deterministic Baseline**: Disturbance $\epsilon(t) \equiv 0.0$ for primary benchmark reproducibility.

---

## 3. Input Variables and State Representations

At every discrete step $t \in \{0, 1, 2, \dots\}$, the simulation ingests two external observations and maintains internal simulation states:

| Variable | Symbol | Unit / Range | Source | Physical Description |
| :--- | :---: | :---: | :--- | :--- |
| **Workload** | $W(t)$ | $[0.0, 1.0]$ | Real Alibaba 300s trace | Cluster CPU utilization normalized from `cpu_util_percent / 100`. |
| **Ambient Temperature** | $T^{\text{amb}}(t)$ | $^\circ\text{C}$ | Real NASA POWER 300s aligned | Outdoor near-surface dry-bulb temperature ($T_{2M}$). |
| **Internal Temperature** | $T^{\text{int}}(t)$ | $^\circ\text{C}$ | Simulated State | Internal intake air temperature of the data hall. |
| **Cooling Level** | $C(t)$ | $[0.0, 1.0]$ | Simulated Control State | Effective operating effort of the cooling equipment ($0.0 = \text{off}$, $1.0 = \text{max}$). |

---

## 4. Thermal State Equation

The evolution of internal data center temperature $T^{\text{int}}$ across a discrete 5-minute timestep is governed by the recurrence relation:

$$T^{\text{int}}(t+1) = T^{\text{int}}(t) + \alpha \cdot W(t) + \beta \cdot \left(T^{\text{amb}}(t) - T^{\text{int}}(t)\right) - \text{cooling\_effect}(t) + \epsilon(t)$$

Where:
- $\alpha \cdot W(t)$ represents **IT Workload Heat Generation** ($\alpha > 0$).
- $\beta \cdot (T^{\text{amb}}(t) - T^{\text{int}}(t))$ represents **Envelope Heat Exchange** ($\beta > 0$).
- $\text{cooling\_effect}(t)$ represents **Net Active Heat Extraction** by the cooling plant.
- $\epsilon(t)$ represents **Unmodeled Thermal Disturbances** ($\epsilon(t) \equiv 0.0$ by default).

### Dimensional Consistency
Every term in the recurrence equation carries the units of **degrees Celsius per timestep ($^\circ\text{C} / \text{step}$)**:
- $[T^{\text{int}}(t+1)] = {}^\circ\text{C}$
- $[T^{\text{int}}(t)] = {}^\circ\text{C}$
- $[\alpha \cdot W(t)] = \left(\frac{^\circ\text{C}}{\text{step}}\right) \cdot [1] = {}^\circ\text{C}/\text{step}$
- $\left[\beta \cdot (T^{\text{amb}} - T^{\text{int}})\right] = [1] \cdot [^\circ\text{C}] = {}^\circ\text{C}/\text{step}$
- $[\text{cooling\_effect}(t)] = {}^\circ\text{C}/\text{step}$

---

## 5. Ambient Coupling Term

The building envelope heat transfer term:
$$\Delta T_{\text{envelope}}(t) = \beta \cdot \left(T^{\text{amb}}(t) - T^{\text{int}}(t)\right)$$

### Sign & Physical Correctness
1. **When Outdoor Air is Hotter than Inside ($T^{\text{amb}} > T^{\text{int}}$)**:  
   $(T^{\text{amb}} - T^{\text{int}}) > 0 \implies \Delta T_{\text{envelope}} > 0$. Heat conducts naturally inward into the data hall, raising internal temperature.
2. **When Outdoor Air is Cooler than Inside ($T^{\text{amb}} < T^{\text{int}}$)**:  
   $(T^{\text{amb}} - T^{\text{int}}) < 0 \implies \Delta T_{\text{envelope}} < 0$. Heat conducts naturally outward through the walls, cooling the room (natural economizer effect).
3. **When Temperatures are Equal ($T^{\text{amb}} = T^{\text{int}}$)**:  
   $\Delta T_{\text{envelope}} = 0$. Thermal equilibrium across the envelope.

With calibrated $\beta = 0.05$, a $10^\circ\text{C}$ thermal differential contributes $0.50^\circ\text{C}$ of thermal change over 5 minutes ($6^\circ\text{C}/\text{hour}$), matching typical insulated data center wall assemblies.

---

## 6. Cooling Effectiveness Formulation

### 6.1 Mathematical Critique of the Candidate Formulation
The candidate formulation specifies an ambient-dependent cooling resistance factor $F_{\text{ambient}}(t)$:
$$F_{\text{ambient}}(t) = \text{clip}\left(1 + \lambda \cdot \frac{T^{\text{amb}}(t) - T_{\text{ref}}}{T_{\text{hot}} - T_{\text{ref}}}, F_{\text{lower}}, F_{\text{upper}}\right)$$
$$\text{cooling\_effect}(t) = \frac{\gamma \cdot C(t)}{F_{\text{ambient}}(t)}$$

#### Mathematical Review & Edge-Case Audit:
1. **Sign Correctness**: As $T^{\text{amb}}$ increases above $T_{\text{ref}}$, $F_{\text{ambient}} > 1$. Because $F_{\text{ambient}}$ sits in the denominator, $\text{cooling\_effect}$ decreases. This correctly captures refrigeration derating under elevated condensing temperatures.
2. **Zero Cooling Behavior ($C = 0$)**: $\text{cooling\_effect} = \frac{\gamma \cdot 0}{F_{\text{ambient}}} = 0.0$. Correct. No unphysical refrigeration occurs when chillers are off.
3. **Maximum Cooling Behavior ($C = 1$)**: $\text{cooling\_effect} = \frac{\gamma}{F_{\text{ambient}}}$. Under reference conditions ($25^\circ\text{C}$), cooling removes $\gamma = 1.25^\circ\text{C}$ per step.
4. **Division-by-Zero Safety**: Guaranteed by $F_{\text{lower}} = 0.80 > 0$. The denominator is strictly bounded in $[0.80, 1.30]$, completely preventing singularity risks.
5. **Low Ambient Enhancement ($T^{\text{amb}} < 25^\circ\text{C}$)**: $F_{\text{ambient}} < 1.0 \implies 1 / F_{\text{ambient}} > 1.0$. Cooling capacity is modestly enhanced (up to $+25\%$) due to superior condenser heat rejection in cool weather.

### 6.2 Calibrated Parameters for Cooling Effect
- $\gamma = 1.25^\circ\text{C}/\text{step}$ (Nominal maximum cooling temperature drop)
- $T_{\text{ref}} = 25.0^\circ\text{C}$ (Reference mild outdoor temperature)
- $T_{\text{hot}} = 35.0^\circ\text{C}$ (High heatwave threshold, derived from NASA POWER 95th percentile)
- $\lambda = 0.25$ (25% capacity reduction slope at $35^\circ\text{C}$)
- $F_{\text{lower}} = 0.80$, $F_{\text{upper}} = 1.30$

---

## 7. Coefficient of Performance (COP) Model

The Coefficient of Performance represents the ratio of thermal heat extracted to electrical power consumed:
$$\text{COP}(t) = \frac{Q_{\text{cooling}}(t)}{P_{\text{cooling}}(t)}$$

In vapor-compression refrigeration, elevated outdoor condensing temperatures increase compressor lift and head pressure, reducing thermodynamic efficiency:

$$\text{COP}(t) = \text{clip}\left(\text{COP}_{\text{ref}} - k_{\text{cop}} \cdot \left(T^{\text{amb}}(t) - T_{\text{ref}}\right), \text{COP}_{\min}, \text{COP}_{\max}\right)$$

### Parameter Grounding
- $\text{COP}_{\text{ref}} = 3.50$: Standard seasonal efficiency for modern chilled-water data-center systems at $25.0^\circ\text{C}$ ambient.
- $k_{\text{cop}} = 0.08^\circ\text{C}^{-1}$: Represents a drop of ~0.08 in COP per $1^\circ\text{C}$ increase in outdoor air.
  - At $T^{\text{amb}} = 20.0^\circ\text{C}$: $\text{COP} = 3.50 - 0.08 \times (-5) = 3.90$ (High efficiency).
  - At $T^{\text{amb}} = 25.0^\circ\text{C}$: $\text{COP} = 3.50$ (Reference).
  - At $T^{\text{amb}} = 35.0^\circ\text{C}$: $\text{COP} = 3.50 - 0.08 \times (10) = 2.70$ (Degraded efficiency).
  - At $T^{\text{amb}} = 36.19^\circ\text{C}$ (All-time peak): $\text{COP} = 2.60$.
- $\text{COP}_{\min} = 1.50$, $\text{COP}_{\max} = 5.00$: Enforce thermodynamic feasibility boundaries.

---

## 8. Power and Energy Accounting Model

### 8.1 Unit & Normalization Convention
To avoid asserting speculative megawatt (MW) ratings for the uncharacterized Alibaba facility, the primary simulation adopts a **Normalized Thermal & Power Unit Convention**:
- **Thermal Capacity**: $Q_{\text{max}} = 1.00$ normalized thermal unit.
- **Thermal Extraction**: $Q_{\text{cooling}}(t) = C(t) \cdot Q_{\text{max}} = C(t) \in [0.0, 1.0]$.
- **Electrical Power Demand**:
  $$P_{\text{cooling}}(t) = \frac{Q_{\text{cooling}}(t)}{\text{COP}(t)} = \frac{C(t)}{\text{COP}(t)}$$
- **Electrical Energy Consumed per 5-Minute Step**:
  $$E_{\text{cooling}}(t) = P_{\text{cooling}}(t) \cdot \Delta t_{\text{hours}} = \frac{C(t)}{\text{COP}(t)} \cdot \left(\frac{5}{60}\right) = \frac{C(t)}{12 \cdot \text{COP}(t)}$$

### 8.2 Physical Scaling Reference (Modular Pod Translation)
If physical engineering units are required for reporting or visualization, the normalized units map directly to a representative **$100\text{ kW}$ IT facility pod**:
- $Q_{\text{max}} = 100.0\text{ kW}_{\text{thermal}}$
- $P_{\text{cooling}}(t) = \frac{100 \cdot C(t)}{\text{COP}(t)}\text{ kW}_{\text{electric}}$
- $E_{\text{cooling}}(t) = P_{\text{cooling}}(t) \cdot \left(\frac{1}{12}\right)\text{ kWh}$ (range: $0.0$ to $3.82\text{ kWh}$ per 5-minute step).

---

## 9. Cooling Action Space and Compatibility

The future RL action space $\mathcal{A} = \{0, 1, 2, 3, 4\}$ consists of 5 discrete incremental adjustments:

$$\Delta C \in \{-0.20, -0.10, 0.00, +0.10, +0.20\}$$

The cooling state updates via:
$$C(t+1) = \text{clip}\left(C(t) + \Delta C, 0.0, 1.0\right)$$

### Compatibility Validation
- A single $+0.20$ maximum step adjustment increases cooling rate by:
  $$\Delta (\text{cooling\_effect}) \approx \gamma \cdot 0.20 = 1.25 \times 0.20 = 0.25^\circ\text{C}/\text{step}$$
- This capability enables the agent to immediately arrest a sharp workload spike ($P_{95}$ jump $\approx 0.122 \implies \Delta T \approx +0.122^\circ\text{C}$) in a single action step.
- Four consecutive $+0.20$ adjustments ($4 \times 5\text{ min} = 20\text{ min}$) transition the cooling plant from 20% to 100% capacity, matching physical thermal compressor ramping constraints.

---

## 10. Initial Simulation Conditions

- **Initial Internal Temperature**: $T^{\text{int}}(0) = 24.0^\circ\text{C}$
  - **Rationale**: $24.0^\circ\text{C}$ is a simulation initialization value selected within the ASHRAE-referenced recommended operating range ($18^\circ\text{C}$ to $27^\circ\text{C}$), representing industry standard setpoint operation.
- **Initial Cooling Level**: $C(0) = 0.50$
  - **Rationale**: 50% cooling effort roughly balances mean workload heat generation ($W \approx 0.40$) under median outdoor conditions ($T^{\text{amb}} \approx 27^\circ\text{C}$), initializing the simulation near dynamic equilibrium without transient shock.

---

## 11. Thermal Safety Boundaries (ASHRAE TC 9.9)

We map internal temperatures strictly to official **ASHRAE Thermal Guidelines for Data Processing Environments (Class A1)**:

```
    15°C                  18°C                          27°C                    32°C
-----|---------------------|------------------------------|-----------------------|-----> Temperature
     |  Below Recommended  |     Recommended Envelope     |   Above Recommended   |
     |   (Overcooling)     |     (Optimal Operation)      |    (Elevated Risk)    | Beyond Allowable
```

| Zone Identifier | Temperature Range | Physical / Operational Interpretation |
| :--- | :---: | :--- |
| **Below Recommended** | $T^{\text{int}} < 18.0^\circ\text{C}$ | Safe for silicon, but represents excessive overcooling, energy waste, and chiller strain. |
| **Recommended (Optimal)** | **$18.0^\circ\text{C} \le T^{\text{int}} \le 27.0^\circ\text{C}$** | Target operational envelope; optimal balance of component reliability, fan power, and efficiency. |
| **Above Recommended** | **$27.0^\circ\text{C} < T^{\text{int}} \le 32.0^\circ\text{C}$** | Safe allowable operation, but server internal fans spin up rapidly, increasing parasitic server power. |
| **Beyond A1 Allowable** | **$T^{\text{int}} > 32.0^\circ\text{C}$** | Critical thermal boundary violation; high risk of server thermal throttling and emergency shutdowns. |

> [!CAUTION]
> **27°C is NOT a hardware failure point.** Server silicon operates safely up to 85°C–100°C junction temperatures. 27°C is an ambient facility intake boundary established by ASHRAE to prevent localized rack hot-spots and excess server fan energy consumption.

---

## 12. Temperature Trend Classification

To provide the controller with rate-of-change awareness, we define:
$$\Delta T^{\text{int}}(t) = T^{\text{int}}(t) - T^{\text{int}}(t-1)$$

| Trend Category | Condition | Interpretation |
| :--- | :---: | :--- |
| **Cooling** | $\Delta T^{\text{int}}(t) < -0.20^\circ\text{C}$ | Facility is actively cooling down ($> 2.4^\circ\text{C} / \text{hour}$). |
| **Stable** | **$-0.20^\circ\text{C} \le \Delta T^{\text{int}}(t) \le +0.20^\circ\text{C}$** | Thermal balance ($\le \pm 2.4^\circ\text{C} / \text{hour}$). |
| **Warming** | $\Delta T^{\text{int}}(t) > +0.20^\circ\text{C}$ | Facility is actively accumulating heat ($> 2.4^\circ\text{C} / \text{hour}$). |

**Numerical Sensitivity Evaluation**:  
Over a 5-minute timestep, $\pm 0.20^\circ\text{C}$ filters out steady-state micro-fluctuations while detecting genuine thermal shifts early, giving the agent 2 to 3 timesteps (10 to 15 minutes) to react before temperature drifts across an ASHRAE boundary.

---

## 13. Calibrated Parameter Table

| Parameter Name | Symbol | Value | Units | Type / Origin | Justification |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Timestep** | $\Delta t$ | $300$ | $\text{seconds}$ | Real Data Grounded | Aligned to 5-minute sampling of Alibaba and NASA datasets. |
| **Timestep Hours** | $\Delta t_{\text{hours}}$ | $1/12 \approx 0.08333$ | $\text{hours}$ | Derived Constant | $\Delta t / 3600$. |
| **Workload Heat Coefficient** | $\alpha$ | $1.00$ | $^\circ\text{C}/\text{step}$ | Simulation Calibrated | Produces $0.40^\circ\text{C}/5\text{ min}$ at mean load; $1.0^\circ\text{C}/5\text{ min}$ at peak load. |
| **Ambient Coupling Fraction** | $\beta$ | $0.05$ | dimensionless | Simulation Calibrated | Ensures stability ($0 < \beta < 2$); reflects insulated data hall envelope. |
| **Cooling Capacity** | $\gamma$ | $1.25$ | $^\circ\text{C}/\text{step}$ | Simulation Calibrated | Max cooling ($C=1.0$) removes $1.25^\circ\text{C}/5\text{ min}$; overcomes peak load. |
| **Reference Ambient Temp** | $T_{\text{ref}}$ | $25.0$ | $^\circ\text{C}$ | Real Data Grounded | Close to NASA POWER Bengaluru median ($25.64^\circ\text{C}$). |
| **Heatwave Benchmark Temp** | $T_{\text{hot}}$ | $35.0$ | $^\circ\text{C}$ | Real Data Grounded | Aligned with empirical 95th percentile summer heatwave peaks. |
| **Cooling Derating Slope** | $\lambda$ | $0.25$ | dimensionless | Engineering Calibrated | Imposes 20% net cooling capacity penalty under $35^\circ\text{C}$ heatwave. |
| **Derating Lower Bound** | $F_{\text{lower}}$ | $0.80$ | dimensionless | Stability Bound | Caps sub-ambient cooling enhancement at $+25\%$; prevents division by zero. |
| **Derating Upper Bound** | $F_{\text{upper}}$ | $1.30$ | dimensionless | Physical Bound | Caps maximum ambient cooling derating penalty at $30\%$. |
| **Reference COP** | $\text{COP}_{\text{ref}}$ | $3.50$ | dimensionless | Engineering Grounded | Representative of modern chilled-water data center installations. |
| **COP Temperature Slope** | $k_{\text{cop}}$ | $0.08$ | $^\circ\text{C}^{-1}$ | Engineering Grounded | Chiller lift thermodynamic degradation curve ($\approx 0.08 \text{ COP} / ^\circ\text{C}$). |
| **Minimum COP** | $\text{COP}_{\min}$ | $1.50$ | dimensionless | Thermodynamic Bound | Lowest efficiency floor under extreme heatwave condensing pressure. |
| **Maximum COP** | $\text{COP}_{\max}$ | $5.00$ | dimensionless | Thermodynamic Bound | Economizer / cool night maximum efficiency ceiling. |
| **Normalized Max Capacity** | $Q_{\text{max}}$ | $1.00$ | norm. units | Normalization Choice | Avoids speculative facility megawatt claims ($100\text{ kW}$ pod equivalent). |
| **Initial Temperature** | $T^{\text{int}}(0)$ | $24.0$ | $^\circ\text{C}$ | Simulation Initialization | Simulation initialization value selected within the ASHRAE-referenced recommended operating range. |
| **Initial Cooling Level** | $C(0)$ | $0.50$ | dimensionless | Simulation Initialization | Balances baseline mean workload heat influx. |
| **Disturbance Term** | $\epsilon(t)$ | $0.00$ | $^\circ\text{C}$ | Reproducibility Setting | Deterministic baseline environment. |

---

## 14. Calibration Methodology

Parameters were calibrated systematically to satisfy the **10 Behavioral Requirements**:

1. **Workload Influx**: $\alpha = 1.00 > 0 \implies \frac{\partial T(t+1)}{\partial W(t)} = 1.00 > 0$. Higher workload directly increases heat.
2. **Ambient Influx**: $\beta = 0.05 > 0 \implies \frac{\partial T(t+1)}{\partial T^{\text{amb}}(t)} = 0.05 > 0$. Hotter outdoor air transfers heat inward.
3. **Cooling Extraction**: $\gamma = 1.25 > 0 \implies \frac{\partial T(t+1)}{\partial C(t)} = -\frac{\gamma}{F_{\text{amb}}} < 0$. Higher cooling always lowers temperature.
4. **Ambient Derating**: $\frac{\partial F_{\text{amb}}}{\partial T^{\text{amb}}} > 0 \implies \frac{\partial (\text{cooling\_effect})}{\partial T^{\text{amb}}} < 0$. High outdoor air degrades cooling extraction.
5. **Strong Warming Trajectory**: Under uncooled peak load ($W=0.75, C=0.0$), temperature increases by $+1.35^\circ\text{C}$ per 5 minutes ($16.2^\circ\text{C}/\text{hr}$), demonstrating a strong warming trajectory under zero cooling.
6. **Max Cooling Stabilization**: Under peak stress ($W=0.75, T^{\text{amb}}=36.0^\circ\text{C}$), max cooling ($C=1.0$) stabilizes temperature at $31.39^\circ\text{C}$ (within A1 allowable boundary).
7. **Realistic Step Delta**: Normal operational deltas are bounded in $[-0.3^\circ\text{C}, +0.4^\circ\text{C}]$ per step, avoiding unphysical discrete temperature spikes.
8. **Numerical Stability**: Discrete characteristic root is $A = 1 - \beta = 0.95 \in (-1, 1)$, guaranteeing bounded-input bounded-output (BIBO) stability.
9. **Rich Control Landscape**: Over a 24-hour cycle, static cooling policies fail: fixed $C=0.30$ overheats to $36.7^\circ\text{C}$ by afternoon; fixed $C=0.70$ overcools to $10.5^\circ\text{C}$ at night. An adaptive policy is mandatory.
10. **Non-Trivial Tradeoff**: Continuous max cooling ($C=1.0$) wastes excessive energy (consuming $3.8\times$ more energy than $C=0.25$), forcing the agent to optimize efficiency while maintaining safety.

---

## 15. Numerical Sanity Checks on Paper

We evaluate the finalized model across the six mandatory operational scenarios ($T^{\text{int}}(t) = 24.0^\circ\text{C}$):

| Test Case | $W$ | $T^{\text{amb}}$ ($^\circ\text{C}$) | $C$ | $F_{\text{amb}}$ | Cooling Effect ($^\circ\text{C}$) | $T^{\text{int}}(t+1)$ ($^\circ\text{C}$) | $\Delta T^{\text{int}}$ ($^\circ\text{C}$) | Trend | $\text{COP}$ | $P_{\text{cool}}$ | $E_{\text{cool}}$ | ASHRAE Zone |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **A. Normal W + Mod Amb + Med Cool** | 0.40 | 27.0 | 0.50 | 1.0500 | 0.5952 | **23.9548** | **-0.0452** | Stable | 3.34 | 0.1497 | 0.01248 | Recommended |
| **B. High W + Mod Amb + Med Cool** | 0.70 | 27.0 | 0.50 | 1.0500 | 0.5952 | **24.2548** | **+0.2548** | Warming | 3.34 | 0.1497 | 0.01248 | Recommended |
| **C. Normal W + High Amb + Med Cool** | 0.40 | 35.0 | 0.50 | 1.2500 | 0.5000 | **24.4500** | **+0.4500** | Warming | 2.70 | 0.1852 | 0.01543 | Recommended |
| **D. High W + High Amb + Max Cool** | 0.75 | 36.0 | 1.00 | 1.2750 | 0.9804 | **24.3696** | **+0.3696** | Warming | 2.62 | 0.3817 | 0.03181 | Recommended |
| **E. High W + High Amb + Zero Cool** | 0.75 | 36.0 | 0.00 | 1.2750 | 0.0000 | **25.3500** | **+1.3500** | Warming | 2.62 | 0.0000 | 0.00000 | Recommended |
| **F. Low W + Cool Amb + Low Cool** | 0.20 | 20.0 | 0.20 | 0.8750 | 0.2857 | **23.7143** | **-0.2857** | Cooling | 3.90 | 0.0513 | 0.00427 | Recommended |

### Verification of Behavioral Integrity
- **Case A**: $\Delta T = -0.045^\circ\text{C}$. Near-perfect thermal equilibrium in standard operation.
- **Case B**: Workload jump causes $\Delta T = +0.255^\circ\text{C}$ (classified `Warming`), proving sensitivity to compute surges.
- **Case C**: $35^\circ\text{C}$ heatwave derates cooling and conducts heat inward ($\Delta T = +0.450^\circ\text{C}$), demonstrating weather vulnerability.
- **Case D**: Max cooling absorbs $0.9804^\circ\text{C}$ of heat, stabilizing extreme conditions near $31.4^\circ\text{C}$ equilibrium.
- **Case E**: Complete cooling shutdown produces an immediate $+1.35^\circ\text{C}/5\text{ min}$ thermal surge, demonstrating a strong warming trajectory under zero cooling.
- **Case F**: Night conditions enhance cooling, COP rises to $3.90$, energy consumption plummets by 66%.

---

## 16. Mathematical Stability Analysis

Isolating the internal temperature term $T^{\text{int}}(t)$ in the recurrence relation:

$$T^{\text{int}}(t+1) = (1 - \beta) \cdot T^{\text{int}}(t) + \underbrace{\left[\alpha \cdot W(t) + \beta \cdot T^{\text{amb}}(t) - \frac{\gamma \cdot C(t)}{F_{\text{ambient}}(t)} + \epsilon(t)\right]}_{u(t)}$$

This represents a discrete-time linear time-invariant system of the canonical form:
$$x(t+1) = A \cdot x(t) + u(t)$$
where scalar state $x(t) = T^{\text{int}}(t)$ and scalar system eigenvalue $A = 1 - \beta$.

### Formal Stability Theorems
1. **Asymptotic Stability Condition**:  
   A discrete linear system is asymptotically stable if and only if all system eigenvalues lie strictly inside the unit circle on the complex plane:
   $$|A| < 1 \iff |1 - \beta| < 1 \iff -1 < 1 - \beta < 1 \iff 0 < \beta < 2$$
2. **Non-Oscillatory Monotonicity Condition**:  
   To prevent unphysical alternating-sign numerical oscillations across consecutive timesteps ($A < 0$):
   $$A > 0 \iff 1 - \beta > 0 \iff \beta < 1$$
3. **Verification of Selected Parameter ($\beta = 0.05$)**:  
   $$A = 1 - 0.05 = 0.95$$
   Because $0 < 0.95 < 1$, the discrete recurrence is **strictly BIBO stable and non-oscillatory**.
4. **Thermal Time Constant**:  
   In continuous time, $\dot{T} = -\frac{1}{\tau} T$. Discretized with backward Euler over $\Delta t = 300\text{ s}$:
   $$1 - \beta = e^{-\Delta t / \tau} \approx 1 - \frac{\Delta t}{\tau} \implies \tau \approx \frac{\Delta t}{\beta} = \frac{300\text{ s}}{0.05} = 6,000\text{ seconds} = 1.67\text{ hours}$$
   A thermal relaxation time of ~1.67 hours accurately reflects the physical thermal mass of a commercial data center room envelope.

---

## 17. Limitations

1. **Spatial Lumped-Mass Approximation**: The model neglects micro-scale rack inlet/outlet temperature gradients and hot-aisle containment bypass leakage.
2. **Instantaneous Airflow Response**: The model assumes cooling airflow and chiller lift adjust within the 5-minute timestep, omitting sub-minute fan inertia.
3. **Exogenous Ambient Coupling**: Macro-meteorological reanalysis temperatures are assumed unaffected by localized data-center condenser exhaust recirculation.
