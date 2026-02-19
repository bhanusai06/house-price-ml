# House Price Prediction ML Project

This project predicts house prices using the California Housing dataset.

## Model
- Random Forest Regression
- Hyperparameter tuning
- Cross-validation
- Feature engineering

## Features Used
- Median income
- House age
- Rooms
- Population
- Location
- Engineered features

## Output
Predicts house price in:
- USD
- INR
- Lakhs
- Crores

## Files
house_rf_model.py → training script  
predict.py → prediction script  
california_house_rf.pkl → trained model  

## How to Run

Train model:
python house_rf_model.py

Predict price:
python predict.py
