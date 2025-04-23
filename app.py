from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

# Load the data
data = pd.read_csv('sales_data.csv')
X = data[['Marketing Spend']]
y = data['Sales']

# Train model
model = LinearRegression()
model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    marketing_spend = float(request.form['marketing_spend'])
    prediction = model.predict(np.array([[marketing_spend]]))
    return render_template('index.html', prediction=round(prediction[0], 2))

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=True, host="0.0.0.0", port=port)

