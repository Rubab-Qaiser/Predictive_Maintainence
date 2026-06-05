# Predictive Maintenance ML Pipeline with MLflow Tracking

This project implements a complete end-to-end machine learning pipeline for predictive maintenance in manufacturing equipment. The system predicts equipment failure based on sensor readings (temperature, vibration, pressure, RPM, and equipment age) using multiple ML models with comprehensive experiment tracking and model registry capabilities.

## Lab Structure

This project was completed in two parts:

### Part 1: ML Pipeline & Experiment Tracking (Week 13)

**Objectives:**
- Build production ML pipeline for predictive maintenance
- Implement MLflow for experiment tracking
- Train and compare multiple models
- Track parameters, metrics, and artifacts

**Dataset:**
- 10,000 synthetic equipment sensor readings
- Features: temperature, vibration, pressure, RPM, age_days
- Target: Equipment failure (binary classification)
- Failure rate: ~15-25%

**Models Evaluated:**

| Model | ROC AUC | Accuracy | F1 Score | Precision | Recall |
|-------|---------|----------|----------|-----------|---------|
| **Random Forest** | **0.9751** | 0.9670 | 0.5976 | 0.6049 | 0.5904 |
| **XGBoost** | 0.9710 | 0.9675 | 0.6061 | 0.6098 | 0.6024 |
| Logistic Regression | 0.9234 | 0.9620 | 0.2549 | 0.6842 | 0.1566 |

**Key Findings:**
- Random Forest achieved highest ROC AUC (0.9751)
- XGBoost showed better F1 score (0.6061)
- Temperature and vibration were strongest failure predictors

### Part 2: Model Registry & Versioning (Week 14)

**Objectives:**
- Implement model registry for production
- Manage model versions and lifecycle stages
- Build production inference pipeline
- Test rollback capabilities

**Registry Workflow:**
Training → Version 1 → Staging → Testing → Production → Rollback Testing

text

**Implemented Features:**

| Feature | Status |
|---------|--------|
| Model Versioning |  Complete |
| Stage Transitions |  Complete |
| Production Inference |  Complete |
| Rollback Capability |  Complete |
| Model Documentation |  Complete |
| Metadata Tags |  Complete |

## Technical Implementation

### Environment & Tools

```python
# Core libraries
- Python 3.12
- scikit-learn 1.3+
- XGBoost 1.7+
- pandas, numpy
- MLflow (for tracking)
- DagsHub (MLflow hosting)
- joblib (model serialization)
Data Pipeline
python
# Data generation and preprocessing
np.random.seed(42)
# Generated 10,000 synthetic samples
# Applied StandardScaler for feature scaling
# Train/test split: 80/20 with stratification
Model Training Pipeline
python
# Random Forest (Best Model)
model_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
model_rf.fit(X_train_scaled, y_train)

# Performance: ROC AUC = 0.9751, Accuracy = 0.9670
Production Inference Function
python
def predict_equipment_failure(temperature, vibration, pressure, rpm, age_days):
    """Production inference using current Production model"""
    # Loads latest Production model from registry
    # Returns: will_fail, recommendation, model_version
```

## Project Structure
text
predictive-maintenance-ml-pipeline/
│
├── week13_ml_pipeline.ipynb
│   ├── Data generation & EDA
│   ├── Model training (3 models)
│   └── MLflow experiment tracking
│
├── week14_model_registry.ipynb
│   ├── Model registration
│   ├── Staging → Production workflow
│   ├── Production inference pipeline
│   └── Rollback testing
│
├── model_registry/
│   ├── PredictiveMaintenance_v1.pkl
│   └── PredictiveMaintenance_v2.pkl
│
├── README.md
└── screenshots/
    ├── mlflow_experiments.png
    ├── model_registry.png
    └── production_predictions.png
## Key Results
Model Comparison Chart
text
## ROC AUC Comparison:
- Random Forest  ████████████████████ 0.9751
- XGBoost        ███████████████████  0.9710
- Logistic Reg.  █████████████████    0.9234

## Accuracy Comparison:
- Random Forest  ███████████████████  0.9670
- XGBoost        ███████████████████  0.9675
- Logistic Reg.  ███████████████████  0.9620
## Production Test Scenarios
Scenario	Temperature	Vibration	Pressure	Age	Prediction
- Normal	70°F	0.4	95 PSI	100 days	 Normal Operation
- Medium Risk	85°F	0.6	110 PSI	200 days	 Schedule Maintenance
- High Risk	95°F	0.9	135 PSI	320 days	 Schedule Maintenance
  
## Learning Outcomes
### Week 13 Achievements:
 Generated synthetic dataset for predictive maintenance

1.  Performed exploratory data analysis with visualizations

2.  Built data preprocessing pipeline with scaling

3.  Trained 3 ML models (Logistic Regression, Random Forest, XGBoost)

4.  Implemented MLflow experiment tracking

5.  Compared model performance metrics

## Week 14 Achievements:
1.  Mastered MLflow Model Registry concepts

2.  Registered and versioned models

3.  Managed model lifecycle stages

4.  Built production inference pipeline

5.  Tested model rollback capability

6.  Implemented model documentation and tagging

## MLOps Skills Demonstrated
Skill	Level
Experiment Tracking	
Model Versioning	
Registry Management	
Production Deployment	
Rollback Strategies	
Code Organization	
Challenges & Solutions
- Challenge 1: MLflow Artifact Upload
Issue: DagsHub artifact upload failed in Kaggle environment
Solution: Implemented manual registry with joblib serialization while maintaining MLflow concepts

- Challenge 2: Model Serialization
Issue: Pickle security warnings and compatibility
Solution: Used joblib with proper verification, documented security considerations

- Challenge 3: Environment Isolation
Issue: Different Python environments and kernels
Solution: Established consistent conda environment (tfenv) with verified dependencies

## Future Improvements
Containerization: Deploy using Docker for consistent environments

CI/CD Pipeline: Automate model training and deployment

Model Monitoring: Implement drift detection and performance monitoring

A/B Testing: Add support for canary deployments

API Deployment: Create REST API for model serving (Week 15)

## How to Run This Project
Prerequisites
bash
## Install required packages
pip install scikit-learn xgboost pandas numpy matplotlib seaborn mlflow joblib
Run Lab 13 (Training & Tracking)
Open week13_ml_pipeline.ipynb

Run all cells to generate data and train models

View MLflow UI at DagsHub URL

Run Lab 14 (Registry & Production)
Open week14_model_registry.ipynb

Ensure trained models from Lab 13 are available

Follow the step-by-step registry workflow

Test production inference function

## References
MLflow Documentation

DagsHub MLflow Integration

Scikit-learn Model Persistence

XGBoost Documentation

## Acknowledgments
Lab instructions from Week 13 & 14 curriculum

DagsHub for MLflow hosting

Kaggle for notebook environment

Author: *Rubab Qaiser*
Project: Predictive Maintenance ML Pipeline
Course: Machine Learning Engineering - Weeks 13 & 14
Status:  Complete


