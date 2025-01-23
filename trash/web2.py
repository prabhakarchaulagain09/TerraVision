from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Loading the model
model_filename = 'crop_recommendation_model.pkl'
try:
    model = joblib.load(model_filename)
except FileNotFoundError:
    raise FileNotFoundError(f"The model file '{model_filename}' does not exist.")
except Exception as e:
    raise Exception(f"Error loading the model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data
        nitrogen = request.form.get('nitrogen')
        phosphorous = request.form.get('phosphorous')
        potassium = request.form.get('potassium')
        temperature = request.form.get('temperature')
        humidity = request.form.get('humidity')
        ph = request.form.get('ph')
        rainfall = request.form.get('rainfall')

        # Convert form data to float
        nitrogen = float(nitrogen)
        phosphorous = float(phosphorous)
        potassium = float(potassium)
        temperature = float(temperature)
        humidity = float(humidity)
        ph = float(ph)
        rainfall = float(rainfall)

        # Prepare data for prediction
        features = np.array([[nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall]])

        # Make a prediction
        prediction = model.predict(features)
        crop = prediction[0]

        # Respond with the prediction
        return jsonify({'Recommended Crop': crop})
    except Exception as e:
        # Handle errors and bad requests
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)