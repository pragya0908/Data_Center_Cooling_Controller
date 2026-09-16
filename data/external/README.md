# External Environmental Data (`data/external/`)

This directory documents external environmental datasets used to supply exogenous meteorological conditions to the CoolRL simulation environment.

> [!IMPORTANT]
> **No weather files are downloaded in Phase 1A.** Data retrieval via the NASA POWER REST API will occur during the data ingestion phase.

---

## 1. Meteorological Source: NASA POWER Hourly Surface Meteorology

- **Portal**: [NASA POWER (Prediction of Worldwide Energy Resources)](https://power.larc.nasa.gov/)
- **NASA Open Data Catalog**: [NASA POWER Open Data Listing](https://data.nasa.gov/dataset/prediction-of-worldwide-energy-resources-power)
- **API Endpoint**: `https://power.larc.nasa.gov/api/temporal/hourly/point`
- **License**: NASA Open Data Policy (Public domain for academic, scientific, and research use)
- **Primary Meteorological Parameter**:
  - `T2M`: Temperature at 2 Meters above ground level
  - **Units**: Degrees Celsius (°C)
  - **Temporal Resolution**: Hourly observations
- **Target Location (Initial Scenario)**:
  - **City**: Bengaluru, Karnataka, India
  - **Latitude**: `12.9716`
  - **Longitude**: `77.5946`
  - **API Community Parameter**: `RE` (Renewable Energy)

---

## 2. Crucial Scientific & Methodological Distinction

```
┌──────────────────────────────────────────────────────────┐
│              COOLRL METHODOLOGICAL COUPLING              │
├────────────────────────────┬─────────────────────────────┤
│ Real-World Workload Traces │ Real-World Weather Scenario │
│ (Alibaba 2018 / Google)    │ (NASA POWER - Bengaluru)    │
└─────────────┬──────────────┴──────────────┬──────────────┘
              │                             │
              └──────────────►◄─────────────┘
                             │
                             ▼
       Data-Grounded Simulation Environment (CoolRL)
```

1. **No Physical Geographic Association**: We explicitly do **not** claim that the Alibaba cluster (Hangzhou, China) or Google cluster physically resided in Bengaluru, India.
2. **Coupled Experimental Grounding**: Bengaluru meteorology serves as an **authentic, real-world outdoor climate scenario** (exhibiting realistic diurnal temperature swings, monsoon transitions, and seasonal heat peaks) to ground the data-center heat dissipation dynamics.
3. **Formal Project Description**: This architecture is rigorously documented as:
   > *"Real-world compute workload traces combined with real-world meteorological observations to construct a data-grounded simulation environment."*

---

## 3. Data Retrieval & Ingestion Protocol (Phase 1B/1C)

- **API Query Format**:
  ```http
  GET https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=T2M&community=RE&longitude=77.5946&latitude=12.9716&start=YYYYMMDD&end=YYYYMMDD&format=JSON
  ```
- **Date Range Selection**: The specific calendar window will be selected during ingestion to match the duration of the workload evaluation episodes (e.g. multi-day continuous spans capturing diurnal day/night cycles).
- **Temporal Alignment**: Hourly $T^{\text{amb}}$ readings will be smoothly interpolated to match sub-hour (e.g., 300-second) simulation timesteps.

---

## 4. Limitations & Scope

- **Macro-scale Weather**: Atmospheric observations at 2m height reflect regional weather station conditions, not localized exhaust recirculation or thermal micro-plumes adjacent to individual chiller condensors.
- **Controlled Simulation Boundary**: Ambient weather governs the heat rejection boundary condition ($T^{\text{amb}}$), while internal rack temperatures and HVAC heat pump thermodynamics remain simulated.
