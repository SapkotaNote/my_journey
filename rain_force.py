import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from time import time
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

start_time = time()

# Initialize LabelEncoder
le = LabelEncoder()

# Load data
data = pd.read_csv(r'C:\\Users\\Kantipur\\Desktop\\New folder\\matplotlib_practicw\\Weather_Data.csv')

# Encode the Weather column
data['Weather'] = le.fit_transform(data['Weather'])

# Define features and target
x = data.drop(['Date/Time', 'Weather'], axis=1)
y = data['Weather']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier(n_estimators=500, random_state=42)
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# ---------- User Input Prediction ----------
Temp_C = float(input("Temperature (°C): "))
Dew_Point_Temp_C = float(input("Dew Point Temp (°C): "))
Rel_Hum__C = float(input("Relative Humidity (%): "))
Wind_Speed_ = float(input("Wind Speed (km/h): "))
Visibility_km = float(input("Visibility (km): "))
Press_kPa = float(input("Pressure (kPa): "))

user_input = [[Temp_C, Dew_Point_Temp_C, Rel_Hum__C, Wind_Speed_, Visibility_km, Press_kPa]]
prediction = model.predict(user_input)

# Convert encoded prediction back to label
weather_label = le.inverse_transform(prediction)
print(f"The predicted weather: {weather_label[0]}")

# Print elapsed time
print("Time taken:", time() - start_time, "seconds")
