from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    loaded_model = joblib.load('trained_model.pkl')
    longitude = float(request.form['longitude'])
    latitude = float(request.form['latitude'])
    housing_median_age = float(request.form['housing_median_age'])
    total_rooms = float(request.form['total_rooms'])
    total_bedrooms = float(request.form['total_bedrooms'])
    population = float(request.form['population'])
    households = float(request.form['households'])
    median_income = float(request.form['median_income'])
    input_data = np.array([[longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income]])

    prediction = loaded_model.predict(input_data)

    return f'Predicted Median House Value: {prediction}'

if __name__ == '__main__':
    app.run(debug=True)
