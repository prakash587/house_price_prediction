from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import waitress

app = Flask(__name__)
data = pd.read_csv('Cleanded_data.csv')
pipe = pickle.load(open('RidgeModel.pkl', 'rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    print(location, bhk, bath, sqft)
    input = pd.DataFrame([[location, bhk, bath, sqft]], columns=['location', 'bhk', 'bath', 'total_sqft'])
    prediction = pipe.predict(input)[0] * 1e5
    return str(np.round(prediction, 2))

if __name__ == '__main__':
    # Use waitress for production
    waitress.serve(app, host='0.0.0.0', port=5001)
