import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Initialize model and label encoder
model = DecisionTreeClassifier()
le = LabelEncoder()

# Load the data
data = pd.read_csv(r'C:\\Users\\Kantipur\\Desktop\\New folder\\matplotlib_practicw\\Weather_Data.csv')

# Encode target labels
data['Weather'] = le.fit_transform(data['Weather'])

# Drop non-numeric column
x = data.drop(['Weather', 'Date/Time'], axis=1)
y = data['Weather']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the model
model.fit(x_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# ===== User Input for Prediction =====
Temp_C = float(input("Enter Temperature (C): "))
Dew_Point_Temp_C = float(input("Enter Dew Point Temp (C): "))
Hum_ = float(input("Enter Humidity (%): "))
Wind = float(input("Enter Wind Speed (km/h): "))
Visibility_km = float(input("Enter Visibility (km): "))
Press_kPa = float(input("Enter Pressure (kPa): "))

# Prepare input and make prediction
user_input = [[Temp_C, Dew_Point_Temp_C, Hum_, Wind, Visibility_km, Press_kPa]]
prediction = model.predict(user_input)

# Decode and print predicted weather
weather_label = le.inverse_transform(prediction)
print("Predicted Weather:", weather_label[0])
