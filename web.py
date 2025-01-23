import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify, redirect, url_for
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# Load the crop names from the CSV file
data = pd.read_csv('Crop_recommendation.csv')
crop_names = data['label'].unique()

# Initialize cart
cart = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        nitrogen = float(request.form['nitrogen'])
        phosphorus = float(request.form['phosphorus'])
        potassium = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Prepare the input array
        mydata = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]

        mydata = np.array(mydata).reshape(1, -1)

        # Make a prediction
        prediction = model.predict(mydata)
        crop_index = prediction[0]  # Predicted crop index
        crop_name = crop_names[crop_index]  # Get the crop name
        
        return render_template('result.html', crop=crop_name)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/seedbank')
def seedbank():
    seeds = crop_names[:22]  # Display 22 seeds
    return render_template('seedbank.html', seeds=seeds)

@app.route('/add_to_cart', methods=['POST'])
def add_to_cart():
    seed = request.form['seed']
    cart.append(seed)
    return redirect(url_for('seedbank'))

@app.route('/remove_from_cart', methods=['POST'])
def remove_from_cart():
    seed = request.form['seed']
    if seed in cart:
        cart.remove(seed)
    return redirect(url_for('seedbank'))

@app.route('/buy_now', methods=['POST'])
def buy_now():
    seed = request.form['seed']
    # Implement the buy logic here
    return redirect(url_for('seedbank'))

# Run the app
if __name__ == '__main__':
    app.run(debug=True)