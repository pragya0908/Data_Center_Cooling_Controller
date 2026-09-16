# CoolRL Multi-Objective Reward Function Specification

**Document Version:** 1.0  
**Phase:** 2C — Formalize and Validate the Multi-Objective Reward Function  
**Component:** `src/environment/reward.py` & `src/environment/datacenter_env.py`  
**Date:** September 2026  
**Status:** Approved & Formally Validated

---

## 1. Reward Objective

The primary objective of the CoolRL multi-objective reward function $R_t$ is to provide an unambiguous, mathematically principled scalar signal that trains a reinforcement learning agent to balance four critical operational concerns in data-center HVAC operation:

1. **Thermal Safety:** Prevent internal intake temperatures from exceeding ASHRAE TC 9.9 Class A1 thermal boundaries.
2. **Energy Conservation:** Minimize chiller electrical energy consumption without compromising thermal reliability.
3. **Overcooling Avoidance:** Discourage unnecessary chilling below the ASHRAE recommended range ($T < 18^\circ\text{C}$).
4. **Actuator Stability:** Penalize excessive cooling action churn ($|\Delta C|$) to avoid high mechanical wear on variable-frequency drives (VFDs) and compressor stages.

### Thermal Safety Priority Hierarchy
The reward hierarchy strictly enforces that thermal preservation dominates energy reduction:

$$\text{Severe Overheating Penalty } (T > 32^\circ\text{C}) \gg \text{Ordinary Thermal Deviation } (27^\circ\text{C} < T \le 32^\circ\text{C}) \gg \text{Overcooling Penalty } (T < 18^\circ\text{C}) \gg \text{Energy Penalty } (E_{\text{cooling}}) > \text{Action Churn Penalty } (|\Delta C|)$$

The agent must **never** discover that saving cooling energy is advantageous when it leads to thermal boundary violations.

---

## 2. Mathematical Reward Equation

The total scalar reward $R_t$ at discrete simulation timestep $t$ is formulated as an additive multi-objective function:

$$R_t = R_{\text{safety}}(T_{t+1}) + R_{\text{severe}}(T_{t+1}) + R_{\text{overcooling}}(T_{t+1}) + R_{\text{energy}}(E_{\text{cooling}, t}) + R_{\text{action}}(\Delta C_t)$$

where:
- $T_{t+1}$ is the internal intake temperature resulting from the transition,
- $E_{\text{cooling}, t}$ is the normalized electrical cooling energy consumed during the step,
- $\Delta C_t \in \{-0.20, -0.10, 0.00, +0.10, +0.20\}$ is the discrete cooling adjustment applied.

All penalty components are non-positive ($R \le 0$), establishing an optimal theoretical ceiling of $0.0$ (achieved only if operating at zero energy, inside the recommended envelope, with zero actuator movement).

---

## 3. Thermal Safety Term

Thermal boundaries adhere to the standard **ASHRAE TC 9.9 Thermal Guidelines (Class A1 Data Centers)**:
- $T \le 27.0^\circ\text{C}$: Inside Recommended Range. Penalty is zero.
- $27.0^\circ\text{C} < T \le 32.0^\circ\text{C}$: Above Recommended Range, but within Class A1 Allowable Envelope ($15^\circ\text{C} - 32^\circ\text{C}$).
- $T > 32.0^\circ\text{C}$: Beyond Class A1 Allowable Envelope (severe overheating risk).

### Mathematical Formulation
The thermal safety penalty is partitioned into two transparent terms:

1. **Ordinary Thermal Deviation Penalty ($R_{\text{safety}}$):**
   A linear penalty on any temperature excess above the recommended threshold $T_{\text{rec, max}} = 27.0^\circ\text{C}$:

   $$R_{\text{safety}}(T) = \begin{cases} 0.0, & \text{if } T \le 27.0^\circ\text{C} \\ -w_{\text{safety}} \cdot (T - 27.0), & \text{if } T > 27.0^\circ\text{C} \end{cases}$$

   With $w_{\text{safety}} = 2.0$, a $1^\circ\text{C}$ violation above $27^\circ\text{C}$ receives a $-2.0$ penalty per 5-minute step.

2. **Severe Overheating Penalty ($R_{\text{severe}}$):**
   A quadratic penalty on any temperature excess beyond the Class A1 allowable threshold $T_{\text{allow, max}} = 32.0^\circ\text{C}$:

   $$R_{\text{severe}}(T) = \begin{cases} 0.0, & \text{if } T \le 32.0^\circ\text{C} \\ -w_{\text{severe}} \cdot (T - 32.0)^2, & \text{if } T > 32.0^\circ\text{C} \end{cases}$$

   With $w_{\text{severe}} = 5.0$, penalties scale superlinearly:
   - At $T = 33^\circ\text{C}$ ($1^\circ\text{C}$ excess): $R_{\text{severe}} = -5.0 \times 1^2 = -5.0$
   - At $T = 34^\circ\text{C}$ ($2^\circ\text{C}$ excess): $R_{\text{severe}} = -5.0 \times 2^2 = -20.0$
   - At $T = 36^\circ\text{C}$ ($4^\circ\text{C}$ excess): $R_{\text{severe}} = -5.0 \times 4^2 = -80.0$

This superlinear escalation creates an impassable reward cliff preventing the agent from allowing temperatures to enter the hazardous zone.

> **Operational Clarification:** $27^\circ\text{C}$ is an operational comfort/reliability target, **not** an instantaneous hardware failure limit. The simulation does not artificially terminate at $27^\circ\text{C}$; the agent experiences the continuous consequence of poor control through the reward gradient.

---

## 4. Energy Term

The energy penalty directly reflects electrical cooling energy calculated by the physical cooling model:

$$E_{\text{cooling}} = P_{\text{cooling}} \cdot \Delta t_{\text{hours}} = \left(\frac{C \cdot P_{\text{cooling, max}}}{\text{COP}(T_{\text{amb}})}\right) \cdot \frac{300}{3600}$$

### Mathematical Formulation

$$R_{\text{energy}} = -w_{\text{energy}} \cdot E_{\text{cooling}}$$

- **Energy Weight ($w_{\text{energy}}$):** Calibrated to $10.0$.
- **Magnitude Bounds:** At maximum cooling ($C = 1.0$) and high ambient temperature ($\text{COP} \approx 2.6$), normalized energy $E_{\text{cooling}} \approx 0.032$. Thus, the maximum single-step energy penalty is:

  $$R_{\text{energy, max}} = -10.0 \times 0.032 = -0.320$$

Because $-0.320$ is an order of magnitude smaller than the mildest safety penalty ($-2.0$ for $T = 28^\circ\text{C}$) and over 300 times smaller than a severe overheating penalty ($-98.0$ at $36^\circ\text{C}$), energy minimization can never incentivize the agent to accept overheating.

Tariff Independence: The formulation intentionally avoids volatile real-world utility tariffs or regional currency pricing ($/kWh), preserving geographic neutrality.

---

## 5. Overcooling Term

Subcooling servers below the ASHRAE recommended minimum ($T_{\text{rec, min}} = 18.0^\circ\text{C}$) increases thermal contraction stress and wastes energy without reliability benefits.

### Mathematical Formulation

$$R_{\text{overcooling}}(T) = \begin{cases} 0.0, & \text{if } T \ge 18.0^\circ\text{C} \\ -w_{\text{overcooling}} \cdot (18.0 - T), & \text{if } T < 18.0^\circ\text{C} \end{cases}$$

- **Overcooling Weight ($w_{\text{overcooling}}$):** Calibrated to $1.0$.
- Inside the recommended range ($18^\circ\text{C} \le T \le 27^\circ\text{C}$), the overcooling penalty is exactly $0.0$. Specifically, operating at $20^\circ\text{C}$ or $24^\circ\text{C}$ incurs **zero** overcooling penalty.
- At $T = 17.0^\circ\text{C}$ ($1.0^\circ\text{C}$ subcooling), $R_{\text{overcooling}} = -1.0 \times 1.0 = -1.0$.

---

## 6. Action-Churn Term

Frequent abrupt changes in cooling output accelerate mechanical wear on compressor clutches, chilled-water valve actuators, and variable-speed fan motors.

### Mathematical Formulation

$$R_{\text{action}}(\Delta C) = -w_{\text{action}} \cdot |\Delta C|$$

- **Action Churn Weight ($w_{\text{action}}$):** Calibrated to $0.50$.
- **Action Delta Mapping:**
  - Action 2 ($\Delta C = 0.00$, Maintain): $R_{\text{action}} = 0.000$
  - Actions 1 & 3 ($\Delta C = \pm 0.10$, Moderate): $R_{\text{action}} = -0.50 \times 0.10 = -0.050$
  - Actions 0 & 4 ($\Delta C = \pm 0.20$, Aggressive): $R_{\text{action}} = -0.50 \times 0.20 = -0.100$

Maintaining steady-state cooling is costless ($0.0$), while larger adjustments are penalized proportionally more than smaller adjustments.

---

## 7. Centralized Reward Configuration (`RewardConfig`)

All weights and thresholds are centralized in a frozen dataclass in `src/environment/reward.py`:

```python
@dataclass(frozen=True)
class RewardConfig:
    safety_weight: float = 2.0
    severe_overheat_weight: float = 5.0
    energy_weight: float = 10.0
    overcooling_weight: float = 1.0
    action_churn_weight: float = 0.50
    temp_recommended_min: float = 18.0  # °C
    temp_recommended_max: float = 27.0  # °C
    temp_allowable_max: float = 32.0    # °C
```

No magic numbers or reward constants are scattered across simulation files.

---

## 8. Explicit Distinction: Real Observations vs. Simulation-Derived Quantities

To maintain scientific integrity, the inputs and outputs of the simulation must be explicitly distinguished:

| Variable | Symbol | Nature | Source / Description |
| :--- | :---: | :---: | :--- |
| **Workload Trace** | $W_t$ | **Real Observed** | Measured server CPU utilization from Alibaba Cluster Trace 2018 ($[0, 1]$). |
| **Ambient Temperature** | $T_{\text{amb}, t}$ | **Real Observed** | NASA POWER hourly surface reanalysis for Bengaluru, India ($^\circ\text{C}$). |
| **Internal Intake Temperature** | $T^{\text{int}}_t$ | *Simulation-Derived* | State variable updated via first-order lumped thermodynamic equation. |
| **Cooling Level** | $C_t$ | *Simulation-Derived* | Actuator control state updated via discrete action increments ($[0, 1]$). |
| **Coefficient of Performance** | $\text{COP}_t$ | *Simulation-Derived* | Chiller efficiency calculated from empirical polynomial curve derating. |
| **Cooling Energy** | $E_{\text{cooling}, t}$ | *Simulation-Derived* | Estimated normalized electrical energy consumed during the 300 s interval. |
| **Reward Signal** | $R_t$ | *Simulation-Derived* | Synthetic reinforcement learning objective constructed for policy optimization. |

> **Critical Note:** Reward values are **not** physical measurements from physical sensors; they represent a mathematical design construct guiding algorithmic optimization.

---

## 9. Timing Convention

The timing convention for reward evaluation is strictly aligned with Markov Decision Process (MDP) theory:

$$\text{State } S_t = (T_t, W_t, T_{\text{amb}, t}, C_t, \Delta T_{t-1}) \xrightarrow{\text{Action } A_t = \Delta C_t} \text{State } S_{t+1} = (T_{t+1}, W_{t+1}, T_{\text{amb}, t+1}, C_{t+1}, \Delta T_t)$$

$$R_t = \mathcal{R}(S_{t+1}, A_t) = \mathcal{R}(T_{t+1}, E_{\text{cooling}, t}, \Delta C_t)$$

1. The cooling level is updated: $C_{t+1} = \text{clip}(C_t + \Delta C_t, 0.0, 1.0)$.
2. The thermodynamic transition calculates the post-action temperature: $T_{t+1} = f(T_t, W_t, T_{\text{amb}, t}, C_{t+1})$.
3. Energy $E_{\text{cooling}, t}$ is consumed based on $C_{t+1}$ and ambient temperature $T_{\text{amb}, t}$.
4. **Reward $R_t$ is computed using $T_{t+1}$, $E_{\text{cooling}, t}$, and $\Delta C_t$.**

This ensures that the agent immediately receives the feedback of its chosen action. Accidental evaluation on the prior temperature $T_t$ is strictly prevented.

---

## 10. Numerical Validation Matrix

The reward function was evaluated under the 11 controlled reference states specified by the Phase 2C mandate.

| State # | Scenario / Condition | $T^{\text{int}}$ (°C) | $E_{\text{cooling}}$ | $\Delta C$ | $R_{\text{safety}}$ | $R_{\text{severe}}$ | $R_{\text{overcool}}$ | $R_{\text{energy}}$ | $R_{\text{churn}}$ | $R_{\text{total}}$ | Logical Assessment |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | Safe, low cooling | 24.0 | 0.005 | 0.00 | 0.00 | 0.00 | 0.00 | -0.050 | 0.00 | **-0.050** | Excellent baseline |
| **2** | Safe, high cooling | 24.0 | 0.030 | 0.00 | 0.00 | 0.00 | 0.00 | -0.300 | 0.00 | **-0.300** | Higher energy penalty |
| **3** | Undercooling | 17.0 | 0.015 | 0.00 | 0.00 | 0.00 | -1.00 | -0.150 | 0.00 | **-1.150** | Penalized for subcooling |
| **4** | Recommended, lower | 20.0 | 0.015 | 0.00 | 0.00 | 0.00 | 0.00 | -0.150 | 0.00 | **-0.150** | Zero thermal penalty |
| **5** | Recommended, upper | 26.0 | 0.015 | 0.00 | 0.00 | 0.00 | 0.00 | -0.150 | 0.00 | **-0.150** | Zero thermal penalty |
| **6** | Mild breach | 28.0 | 0.015 | 0.00 | -2.00 | 0.00 | 0.00 | -0.150 | 0.00 | **-2.150** | Linear penalty active |
| **7** | Approaching boundary | 31.0 | 0.015 | 0.00 | -8.00 | 0.00 | 0.00 | -0.150 | 0.00 | **-8.150** | $4^\circ\text{C}$ excess above 27°C |
| **8** | At allowable boundary | 32.0 | 0.015 | 0.00 | -10.00 | 0.00 | 0.00 | -0.150 | 0.00 | **-10.150** | Class A1 upper limit |
| **9** | Severe breach | 33.0 | 0.015 | 0.00 | -12.00 | -5.00 | 0.00 | -0.150 | 0.00 | **-17.150** | Quadratic penalty kicks in |
| **10**| Severe, high cooling | 36.0 | 0.032 | 0.00 | -18.00 | -80.00 | 0.00 | -0.320 | 0.00 | **-98.320** | Catastrophic penalty |
| **11**| Severe, low cooling | 36.0 | 0.005 | 0.00 | -18.00 | -80.00 | 0.00 | -0.050 | 0.00 | **-98.050** | Catastrophic penalty |

### Key Mathematical Proofs from Validation Matrix:
1. **Thermal Safety Strictly Trumps Energy Conservation:**
   - Safe temperature with maximum cooling energy (State 2): $R = -0.300$
   - Severe overheating with minimum cooling energy (State 11): $R = -98.050$
   - Difference: $\Delta R = +97.750$ in favor of cooling.
   - The agent will **never** save 0.25 energy penalty units at the cost of incurring a 98.0 overheating penalty.
2. **Monotonicity Across Critical Regimes:**
   - As temperature increases from $26^\circ\text{C} \to 28^\circ\text{C} \to 31^\circ\text{C} \to 32^\circ\text{C} \to 33^\circ\text{C} \to 36^\circ\text{C}$, reward decreases monotonically: $-0.150 > -2.150 > -8.150 > -10.150 > -17.150 > -98.320$.
3. **Selective Overcooling Penalty:**
   - State 3 ($17^\circ\text{C}$) receives $-1.00$ overcooling penalty.
   - States 4 and 1 ($20^\circ\text{C}$ and $24^\circ\text{C}$) receive **zero** overcooling penalty.

---

## 11. Deterministic Five-Scenario Rollout Analysis

To verify numerical stability across the full 2,243 simulation steps (7.79 days), deterministic baseline rollouts were executed under fixed maintain-cooling policy ($C = 0.50$, Action 2):

| Metric | Scenario 1: Normal | Scenario 2: High Workload | Scenario 3: Workload Spikes | Scenario 4: High Ambient | Scenario 5: Combined Stress |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Total Episode Steps** | 2,243 | 2,243 | 2,243 | 2,243 | 2,243 |
| **Cumulative Reward** | -4,744.09 | -11,329.96 | -4,830.12 | -26,355.43 | -65,511.56 |
| **Mean Step Reward** | -2.1151 | -5.0513 | -2.1534 | -11.7501 | -29.2071 |
| **Min Step Reward** | -12.8045 | -49.7875 | -12.9215 | -97.4169 | -209.2693 |
| **Max Step Reward** | -0.1079 | -0.1075 | -0.1077 | -0.1146 | -0.1146 |
| **Total Cooling Energy** | 28.9340 | 28.9340 | 28.9340 | 31.2945 | 31.2945 |
| **Steps $> 27^\circ\text{C}$** | 732 (32.6%) | 910 (40.6%) | 745 (33.2%) | 1,114 (49.7%) | 1,516 (67.6%) |
| **Steps $> 32^\circ\text{C}$** | 91 (4.1%) | 307 (13.7%) | 97 (4.3%) | 576 (25.7%) | 803 (35.8%) |
| **Steps $< 18^\circ\text{C}$** | 194 (8.6%) | 41 (1.8%) | 171 (7.6%) | 0 (0.0%) | 0 (0.0%) |
| **Total Safety Penalty** | -4,225.63 | -7,314.84 | -4,339.08 | -10,744.67 | -15,978.86 |
| **Total Severe Penalty** | -37.49 | -3,683.59 | -40.88 | -15,297.82 | -49,219.75 |
| **Total Overcooling Penalty**| -191.63 | -42.20 | -160.82 | 0.00 | 0.00 |
| **Total Action Churn Penalty**| 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

### Reward Scale Observations:
- **Bounded and Well-Conditioned:** Under normal operations, step rewards fluctuate between $-0.10$ and $-12.80$, with an average around $-2.11$.
- **Graceful Stress Degradation:** Under extreme combined stress, average reward drops to $-29.21$ with minimum instantaneous reward reaching $-209.27$. These values remain well within standard floating-point precision ($[-10^3, 0]$), preventing numerical overflow or underflow in future Bellman updates.
- **Zero NaN or Infinite Values:** All rewards are finite and deterministic across all 11,215 simulated steps.

---

## 12. Model Assumptions

1. **Additive Separability:** Each objective (safety, severe overheating, overcooling, energy, churn) can be evaluated independently and combined linearly into a unified scalar reward.
2. **Symmetric Actuator Churn:** Cooling level increases and decreases of identical magnitude incur identical mechanical wear penalties ($|\Delta C|$).
3. **Instantaneous Step Consumption:** The energy penalty is charged for the energy consumed within the discrete 300 s window.
4. **Independence of External Tariffs:** The reward does not depend on fluctuating commercial electricity contracts, peak demand charges, or carbon taxes.

---

## 13. Limitations

1. **Single-Zone Lumping:** The reward relies on a lumped bulk internal temperature $T_{\text{int}}$ and does not penalize localized hot spots or vertical thermal stratification across individual server racks.
2. **Fixed Penalty Weights:** Weights are constant over time and do not adapt dynamically to grid demand-response signals or ambient weather forecasting.
3. **Absence of Humidity Accounting:** The ASHRAE Class A1 envelope also specifies allowable dew point and relative humidity ranges ($5.5^\circ\text{C}$ DP to $60\%$ RH). Humidity is not currently modeled in the first-order thermal simulator.
