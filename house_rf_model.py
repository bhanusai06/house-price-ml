import pandas as pd
import numpy as np
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import randint

print("Loading dataset...")

# =====================================================
# 1. LOAD DATASET
# =====================================================
housing = fetch_california_housing(as_frame=True)
data = housing.frame

print("Dataset loaded. Shape:", data.shape)

X = data.drop("MedHouseVal", axis=1)
y = data["MedHouseVal"]

# =====================================================
# 2. FEATURE ENGINEERING
# =====================================================
print("Creating features...")
X["RoomsPerHousehold"] = X["AveRooms"] / X["HouseAge"]
X["PopulationPerHousehold"] = X["Population"] / X["AveOccup"]

# =====================================================
# 3. TRAIN TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# =====================================================
# 4. RANDOM FOREST MODEL
# =====================================================
rf = RandomForestRegressor(
    random_state=42,
    oob_score=True,
    bootstrap=True,
    n_jobs=-1
)

# =====================================================
# 5. HYPERPARAMETER TUNING (FASTER)
# =====================================================
print("Training model... this may take 1-2 minutes...")

param_dist = {
    "n_estimators": randint(50, 150),
    "max_depth": randint(5, 20),
    "min_samples_split": randint(2, 8),
    "min_samples_leaf": randint(1, 4),
    "max_features": ["sqrt", "log2"]
}

search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=5,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    random_state=42,
    verbose=1
)

search.fit(X_train, y_train)
model = search.best_estimator_

print("Training complete.")

# =====================================================
# 6. CROSS VALIDATION
# =====================================================
cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring="r2")

# =====================================================
# 7. EVALUATION
# =====================================================
pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)
oob = model.oob_score_

print("\n===== MODEL PERFORMANCE =====")
print("MAE:", mae)
print("R2:", r2)
print("OOB Score:", oob)
print("CV R2 Mean:", cv_scores.mean())

# =====================================================
# 8. SAVE MODEL (IMPORTANT)
# =====================================================
joblib.dump(model, "california_house_rf.pkl")
print("\nModel saved as california_house_rf.pkl")

# =====================================================
# 9. EXAMPLE PREDICTION
# =====================================================
sample = X_test.iloc[[0]]
prediction = model.predict(sample)[0]
print("Example predicted house value:", prediction)
