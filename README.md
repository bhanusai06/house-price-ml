# house-price-ml
 House price prediction using Random Forest and California Housing dataset
.

🏠 House Price Prediction using Machine Learning

An end-to-end machine learning project that predicts residential property values using the California Housing dataset and an optimized Random Forest regression model.

This project demonstrates the full machine learning lifecycle — data processing, feature engineering, model training, evaluation, persistence, and real-world prediction.

📌 Overview

Accurate house price estimation is important for:

real estate valuation

investment decisions

market analysis

financial risk assessment

This project builds a supervised regression model that learns relationships between demographic, geographic, and housing features to predict median house value.

📊 Dataset

Source: Scikit-Learn — California Housing Dataset

Each row represents a housing block in California.

Input Features

Median household income

House age

Average number of rooms

Average number of bedrooms

Population

Average occupancy

Latitude

Longitude

Engineered Features

Rooms per household

Population per household

Target Variable

Median house value (scaled in units of $100,000)

🤖 Machine Learning Model

Algorithm: Random Forest Regressor

Random Forest is an ensemble learning method that combines multiple decision trees to improve predictive stability and reduce overfitting.

Why Random Forest?

Handles nonlinear relationships

Robust to noise and outliers

Reduces variance through averaging

Works well with structured tabular data

Requires minimal feature scaling

⚙️ ML Pipeline

Load dataset

Feature engineering

Train–test split

Hyperparameter optimization (RandomizedSearchCV)

Cross-validation

Model evaluation

Model persistence using Joblib

Prediction generation

Currency conversion (USD → INR)

📈 Model Performance
Metric	Score
R² Score	~0.80
Mean Absolute Error	~0.33
Out-of-Bag Score	~0.81
Cross-Validation R²	~0.80
Interpretation

The model explains approximately 80% of the variance in house prices, indicating strong predictive capability for real-world housing data.

💰 Prediction Output

Predictions are converted into multiple financial representations:

USD

INR

Lakhs

Crores

Example Output
USD: 235344.75
INR: ₹ 19,533,614.59
IN LAKHS: 195.34 L
IN CRORES: 1.95 Cr

🧩 Project Structure
house-price-ml/
│
├── house_rf_model.py        # Model training pipeline
├── predict.py               # Inference script
├── california_house_rf.pkl  # Serialized trained model
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation

🚀 Installation & Usage
1. Clone repository
git clone https://github.com/bhanusai06/house-price-ml.git
cd house-price-ml

2. Install dependencies
pip install -r requirements.txt

3. Train model
python house_rf_model.py

4. Generate prediction
python predict.py

🎓 Skills Demonstrated

Supervised machine learning (regression)

Ensemble modeling

Feature engineering

Hyperparameter optimization

Cross-validation techniques

Model evaluation metrics

Model serialization (joblib)

End-to-end ML pipeline design

Real-world output transformation

🔍 Limitations

Dataset represents historical California housing trends

Model assumes future data distribution remains similar

External economic factors not included

🔮 Future Improvements

SHAP model explainability

Gradient Boosting / LightGBM comparison

Interactive prediction UI (Streamlit)

REST API deployment (FastAPI)

Automated model retraining pipeline

👨‍💻 Author

Bhanu Sai Veera Ashok Babu Sonti
Undergraduate — Artificial Intelligence & Data Science

GitHub: https://github.com/bhanusai06

⭐ Support

If you find this project useful, consider giving it a star
