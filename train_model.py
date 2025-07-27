

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib


data = pd.read_csv("Energy_consumption.csv")
data

data.drop('Timestamp',axis=1,inplace=True)

data.info()

categorical_cols = ['HVACUsage', 'LightingUsage', 'DayOfWeek', 'Holiday']

# One-hot encode the categorical columns
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Drop one-hot encoded weekday columns
day_cols = [
    'DayOfWeek_Monday', 'DayOfWeek_Tuesday', 'DayOfWeek_Wednesday',
    'DayOfWeek_Thursday', 'DayOfWeek_Saturday', 'DayOfWeek_Sunday'
]
data = data.drop(columns=day_cols)



import matplotlib.pyplot as plt
import seaborn as sns

# Set up the figure
plt.figure(figsize=(15, 8))

# Create boxplots for selected numerical columns
numerical_cols = ['Temperature', 'Humidity', 'SquareFootage', 'Occupancy', 'RenewableEnergy', 'EnergyConsumption']
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=data[col])
    plt.title(col)

plt.tight_layout()
plt.show()

# Select only numeric columns
numeric_data = data.select_dtypes(include='number')

# IQR method to find outliers
Q1 = numeric_data.quantile(0.25)
Q3 = numeric_data.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = ((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR)))

# Print number of outliers per column
outliers.sum()

# Extract scalar bounds specifically for EnergyConsumption
energy_Q1 = Q1['EnergyConsumption']
energy_Q3 = Q3['EnergyConsumption']
energy_IQR = IQR['EnergyConsumption']

energy_lower = energy_Q1 - 1.5 * energy_IQR
energy_upper = energy_Q3 + 1.5 * energy_IQR

# Now apply the filtering
data = data[~((data['EnergyConsumption'] < energy_lower) | (data['EnergyConsumption'] > energy_upper))]

# Feature and target selection
#X = data[['Temperature', 'Humidity', 'Occupancy', 'RenewableEnergy']]
#y = data['EnergyConsumption']

data['HVAC_Temp_Interaction'] = data['HVACUsage_On'].astype(int) * data['Temperature']
data['PeoplePerSqFt'] = data['Occupancy'] / data['SquareFootage']
# Final feature selection after cleanup and feature engineering
feature_cols = [
    'Temperature', 'Humidity', 'SquareFootage', 'Occupancy',
    'RenewableEnergy', 'HVACUsage_On', 'LightingUsage_On',
    'Holiday_Yes', 'HVAC_Temp_Interaction', 'PeoplePerSqFt'
]
X = data[feature_cols]
y = data['EnergyConsumption']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train.columns.tolist())

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")
print(f"Model Accuracy (R² Score): {r2 * 100:.2f}%")

# Save model and scaler
joblib.dump(model, "linear_regression_model.pkl")
joblib.dump(scaler, "scaler.pkl")

