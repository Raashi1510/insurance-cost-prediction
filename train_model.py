import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle

# Load the updated dataset
df = pd.read_csv("insurance_updated_extended_fixed (1).csv")

# Encode categorical features
df['sex'] = df['sex'].map({'female': 0, 'male': 1})
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
df['alcohol_consumption'] = df['alcohol_consumption'].map({'no': 0, 'yes': 1})
df['region'] = df['region'].map({'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3})
df['diet_type'] = df['diet_type'].map({
    'Balanced': 0, 'High-Protein': 1, 'Fast-Food': 2, 'Vegan': 3
})
df['diabetes'] = df['diabetes'].map({'no': 0, 'yes': 1})
df['heart_disease'] = df['heart_disease'].map({'no': 0, 'yes': 1})
df['hypertension'] = df['hypertension'].map({'no': 0, 'yes': 1})

# Define features and target
features = [
    'age', 'sex', 'bmi', 'children', 'smoker', 'region',
    'exercise_frequency', 'diet_type', 'alcohol_consumption',
    'diabetes', 'heart_disease', 'hypertension',
    'sleep_hours', 'stress_level', 'water_intake',
    'screen_time', 'health_checkups'
]
target = 'charges'

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Model MAE: ₹{mae:.2f}")

# Save the model
with open("insurance_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as 'insurance_model.pkl'")
