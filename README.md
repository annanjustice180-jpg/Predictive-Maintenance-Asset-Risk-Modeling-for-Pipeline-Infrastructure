# Pipeline Predictive Maintenance & Failure Forecasting


Predictive maintenance system designed to identify pipeline infrastructure failure risk and support risk-based inspection planning for energy assets.

This project demonstrates how data analytics and machine learning can be applied to pipeline asset integrity management to forecast failure risk and detect degradation trends.



## Project Overview

Pipeline failures can lead to operational disruption, environmental risk, and high maintenance costs. Traditional inspection strategies often rely on fixed schedules rather than risk-based monitoring.

This project develops a predictive maintenance framework capable of identifying pipeline assets at risk of failure and detecting degradation signals months before failure events occur.

The framework integrates:

- Time-series degradation modeling
- Rare-event machine learning
- Inspection threshold optimization
- Operational monitoring dashboards



## Key Results

### Imminent Failure Detection Model

Predicts failures within a **12-month horizon**.

Performance:

- Recall: **91.7%**
- Precision: **73.3%**
- PR-AUC: **0.82**

Operational inspection workload:

**0.57% of monitored assets**

This allows operators to detect most upcoming failures while inspecting fewer than 1% of pipeline assets.



### Early Warning Degradation Model

Identifies degradation patterns before imminent failure signals emerge.

Performance:

- PR-AUC: **0.20**
- Rare-event baseline: **≈0.0017**

Lead time:

- Average lead time: **7.6 months**
- Median lead time: **8 months**

This provides time for preventive maintenance planning.



## Dashboard

An interactive **Streamlit dashboard** was developed to operationalize model outputs.

Dashboard features include:

- Risk tier segmentation
- Inspection workload estimation
- Asset risk prioritization
- Adjustable alert thresholds
- Downloadable inspection candidate lists

Live dashboard:https://pipeline-predictive-maintenance-dashboard-bguhkqbzfpvj9ukypmsm.streamlit.app/

## Repository Structure

pipeline-predictive-maintenance/
│
├── data/
│ └── market_pipe_thickness_loss_dataset.csv
│
├── notebooks/
│ └── predictive_maintenance_model.ipynb
│
├── dashboard/
│ ├── app.py
│ └── risk_snapshot.csv
│
├── reports/
│ └── Pipeline Predictive Maintenance Case Study.md
│
├── requirements.txt
│
└── README.md


## Dataset

The dataset contains operational characteristics for **881 pipeline assets** including:

- Pipe diameter and wall thickness
- Operating pressure and temperature
- Material type and grade
- Corrosion exposure indicators
- Degradation measurements

These variables represent structural and operational pipeline conditions.



## Methodology

### Data Preparation

Validation steps included:

- Removal of physically inconsistent degradation values
- Correction of target leakage
- Feature distribution validation

Final dataset:

**881 pipeline assets**



### Time-Series Simulation

To replicate continuous monitoring environments, a time-series dataset was simulated.

Each asset was modeled across:

**60 months of operational history**

Resulting dataset:

**≈52,760 asset-month observations**



### Feature Engineering

Temporal degradation indicators were constructed including:

- Loss increment
- Rolling degradation averages
- Degradation acceleration
- Remaining structural thickness

These features enable models to detect degradation **trends rather than single observations**.


### Rare Event Modeling

Pipeline failures represent a rare event classification problem.

Failure rate:

**≈0.23%**

To address this challenge:

- Class-weighted **XGBoost models** were trained
- Asset-level train/test splits prevented temporal leakage
- Operational thresholds balanced detection accuracy and inspection workload



## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Matplotlib



## Operational Impact

This project demonstrates how predictive analytics can support pipeline asset integrity management by:

- Improving infrastructure reliability
- Reducing unnecessary inspections
- Prioritizing maintenance resources
- Enabling risk-based asset management



## Author

**Annan Yaw Enu**

Energy Data Consultant  
Energy & Sustainability Data Scientist | AI for Oil, Gas & Renewables | Forecasting | Helping Energy Companies Reduce Downtime & Optimize Operations
