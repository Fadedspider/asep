import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import xgboost as xgb
import streamlit as st

# Load and train model (your existing code)
data = pd.read_csv('crops_data_2015_2023.csv')
for col in ['Temperature', 'soil_ph', 'water_required', 'productivity', 'average_annual_rainfall', 'Humidity', 'area']:
    data[col] = data[col].fillna(data[col].mean())
numeric_cols = ['soil_ph', 'water_required', 'Temperature', 'productivity', 'average_annual_rainfall', 'Humidity', 'area']
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data['temp_rain'] = data['Temperature'] * data['average_annual_rainfall']
data['water_per_area'] = np.log1p(data['water_required'] / data['area'])
data['productivity'] = data['productivity'].clip(upper=2000)
X = data[['district', 'soil_type', 'water_required', 'Temperature', 'average_annual_rainfall', 'Humidity', 'crop', 'area', 'temp_rain', 'water_per_area']]
y = np.log1p(data['productivity'])
X_encoded = pd.get_dummies(X, columns=['district', 'soil_type', 'crop'])
scaler = StandardScaler()
scale_cols = ['water_required', 'Temperature', 'average_annual_rainfall', 'Humidity', 'area', 'temp_rain', 'water_per_area']
X_encoded[scale_cols] = scaler.fit_transform(X_encoded[scale_cols])
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
param_grid = {
    'max_depth': [5, 7],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [200, 300],
    'reg_lambda': [1.0, 10.0]
}
grid = GridSearchCV(xgb.XGBRegressor(random_state=42, early_stopping_rounds=10), param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
model = grid.best_estimator_

# Prediction function (same as above)
def predict_best_crop_and_fertilizer(district, soil_type, soil_ph, water_required, temperature, rainfall, humidity, area_acres):
    area_ha = area_acres * 0.404686
    crops = ['Bajra', 'Jowar', 'Ragi']
    predictions = {}
    
    farmer_input = pd.DataFrame({
        'district': [district] * len(crops),
        'soil_type': [soil_type.upper()] * len(crops),
        'water_required': [water_required] * len(crops),
        'Temperature': [temperature] * len(crops),
        'average_annual_rainfall': [rainfall] * len(crops),
        'Humidity': [humidity] * len(crops),
        'crop': crops,
        'area': [area_ha] * len(crops),
        'temp_rain': [temperature * rainfall] * len(crops),
        'water_per_area': [np.log1p(water_required / area_ha)] * len(crops)
    })
    
    farmer_input_encoded = pd.get_dummies(farmer_input, columns=['district', 'soil_type', 'crop'])
    missing_cols = set(X_encoded.columns) - set(farmer_input_encoded.columns)
    for col in missing_cols: farmer_input_encoded[col] = 0
    farmer_input_encoded = farmer_input_encoded[X_encoded.columns]
    farmer_input_encoded[scale_cols] = scaler.transform(farmer_input_encoded[scale_cols])
    
    for i, crop in enumerate(crops):
        pred_productivity_log = model.predict(farmer_input_encoded.iloc[[i]])[0]
        pred_productivity_log = np.clip(pred_productivity_log, 0, 7.6)
        pred_productivity = np.expm1(pred_productivity_log)
        predictions[crop] = pred_productivity
    
    district_data = data[data['district'].str.lower() == district.lower()]
    if not district_data.empty:
        district_avg = district_data.groupby('crop')['productivity'].mean().to_dict()
        for crop in predictions:
            if crop in district_avg:
                predictions[crop] = predictions[crop] * 0.2 + district_avg[crop] * 0.8
            else:
                predictions[crop] *= 0.5
    
    best_crop = max(predictions, key=predictions.get)
    total_yield = predictions[best_crop] * area_ha / 1000
    fertilizer_needs = {'Bajra': {'N': 60, 'P': 20, 'K': 40}, 'Jowar': {'N': 80, 'P': 40, 'K': 20}, 'Ragi': {'N': 50, 'P': 50, 'K': 40}}
    fertilizer_ha = fertilizer_needs[best_crop]
    
    return predictions, best_crop, total_yield, fertilizer_ha

# Streamlit frontend
st.title("Crop Productivity Prediction")
st.write("Enter farm details to predict the best crop and yield.")

district = st.text_input("District (e.g., buldhana)")
soil_type = st.text_input("Soil Type (e.g., BLACK)")
soil_ph = st.number_input("Soil pH (0-14)", min_value=0.0, max_value=14.0, value=7.0)
water_required = st.number_input("Water Required (liters)", min_value=0.0, value=3000000.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0, value=34.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=980.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=21.0)
area_acres = st.number_input("Area (acres)", min_value=0.0, value=8.0)

if st.button("Predict"):
    predictions, best_crop, total_yield, fertilizer_ha = predict_best_crop_and_fertilizer(
        district, soil_type, soil_ph, water_required, temperature, rainfall, humidity, area_acres
    )
    st.write("### Results")
    st.write(f"**Predictions (kg/ha):**")
    for crop, value in predictions.items():
        st.write(f"{crop}: {value:.2f}")
    st.write(f"**Best Crop:** {best_crop}")
    st.write(f"**Total Yield:** {total_yield:.2f} tonnes")
    st.write(f"**Fertilizer (kg/ha):** N: {fertilizer_ha['N']}, P: {fertilizer_ha['P']}, K: {fertilizer_ha['K']}")
