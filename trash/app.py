from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import joblib
import pandas as pd

data = pd.read_csv('Crop_recommendation.csv')

y = data['label']
y_encoded = pd.factorize(y)[0]

print(y_encoded)

app = Flask(__name__)

# Load the trained model from .pkl file
#with open('model.pkl', 'rb') as file:
#    model = pickle.load(file)

rf_model = joblib.load('model.pkl')

@app.route('/')
def home():
    """home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Take input from the form, predict using the model, and return the result."""
    try:
        # Get data from the form
        nitrogen = float(request.form['nitrogen'])
        phosphorus = float(request.form['phosphorus'])
        potassium = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])


         # Prepare the features for the model
        mydata = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]

        #pd.dataframe
        mydata = np.array(mydata).reshape(1, -1)

        result = rf_model.predict(mydata)
        print(result)

        crops = pd.factorize('lable')[0]
        crop_name = crops[result[0]]
        print(crop_name)    
    


        # prediction = rf_model.predict(features)
        # predicted_crop = prediction[0]

        # print(predicted_crop)

        crops = pd.factorize(mydata[0])[0]
        print(crops)

#        predicted_crop = str(predicted_crop)
        

        # Predict the most suitable crop
#        prediction = rf_model.predict(mydata)
#        predicted_crop = prediction[0]  # Assuming the model predicts a single output

        # Return the result
        return jsonify({'Predicted Crop': crops})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
