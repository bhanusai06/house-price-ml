import joblib
import pandas as pd

# LOAD MODEL
model = joblib.load("california_house_rf.pkl")

# SAMPLE INPUT
sample = {
    "MedInc": 5.0,
    "HouseAge": 20,
    "AveRooms": 6,
    "AveBedrms": 1,
    "Population": 1000,
    "AveOccup": 3,
    "Latitude": 34.0,
    "Longitude": -118.0,
    "RoomsPerHousehold": 6/20,
    "PopulationPerHousehold": 1000/3
}

df = pd.DataFrame([sample])

prediction = model.predict(df)[0]

# Convert
price_usd = prediction * 100000
usd_to_inr = 83
price_inr = price_usd * usd_to_inr

# Convert to lakh and crore
price_lakh = price_inr / 100000
price_crore = price_inr / 10000000

# OUTPUT
print("\n===== HOUSE PRICE PREDICTION =====")
print("USD:", round(price_usd, 2))
print("INR: ₹", format(round(price_inr, 2), ","))
print("IN LAKHS:", round(price_lakh, 2), "L")
print("IN CRORES:", round(price_crore, 2), "Cr")
