import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import flask
from flask import Flask, request, jsonify, render_template

# Step 1: Simulating Data for Safety Parameters
def generate_data(samples=1000):
    np.random.seed(42)
    temperature = np.random.uniform(20, 100, samples)  # in Celsius
    noise_level = np.random.uniform(30, 120, samples)  # in dB
    proximity = np.random.uniform(0, 2, samples)       # in meters

    # Safety thresholds
    temp_threshold = 50
    noise_threshold = 85
    proximity_threshold = 0.5

    # Label: 0 = Safe, 1 = Unsafe
    labels = (
        (temperature > temp_threshold) |
        (noise_level > noise_threshold) |
        (proximity < proximity_threshold)
    ).astype(int)

    data = pd.DataFrame({
        'Temperature': temperature,
        'Noise_Level': noise_level,
        'Proximity': proximity,
        'Label': labels
    })
    return data

# Generate and split data
data = generate_data()
X = data[['Temperature', 'Noise_Level', 'Proximity']]
y = data['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train a Classification Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Flask App Setup
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    temp = float(data['temperature'])
    noise = float(data['noise_level'])
    prox = float(data['proximity'])

    input_data = np.array([[temp, noise, prox]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        result = "ALERT! Unsafe conditions detected."
    else:
        result = "Safe conditions."

    return render_template('index2.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)