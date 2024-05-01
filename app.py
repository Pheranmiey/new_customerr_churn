from flask import Flask, request, render_template, redirect, url_for
import pickle
import numpy as np
import logging

logging.basicConfig(filename="churn.log", level=logging.INFO)

app = Flask(__name__)

scaler = pickle.load(open("/config/workspace/model/minim_max.pkl", "rb"))
model = pickle.load(open("/config/workspace/model/XGBclass_model.pkl", "rb"))
label_encoders = pickle.load(open("/config/workspace/model/label_encoders.pkl", "rb"))

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict_datapoint', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Extracting data from form
            Tenure = float(request.form.get('Tenure'))
            CityTier = float(request.form.get('CityTier'))
            WarehouseToHome = float(request.form.get('WarehouseToHome'))
            HourSpendOnApp = float(request.form.get('HourSpendOnApp'))
            NumberOfDeviceRegistered = float(request.form.get('NumberOfDeviceRegistered'))
            SatisfactionScore = float(request.form.get('SatisfactionScore'))
            NumberOfAddress = float(request.form.get('NumberOfAddress'))
            Complain = float(request.form.get('Complain'))
            OrderAmountHikeFromlastYear = float(request.form.get('OrderAmountHikeFromlastYear'))
            CouponUsed = float(request.form.get('CouponUsed'))
            OrderCount = float(request.form.get('OrderCount'))
            DaySinceLastOrder = float(request.form.get('DaySinceLastOrder'))
            CashbackAmount = float(request.form.get('CashbackAmount'))

            # Encoding object columns
            PreferredLoginDevice = request.form.get('PreferredLoginDevice')
            PreferredPaymentMode = request.form.get('PreferredPaymentMode')
            Gender = request.form.get('Gender')
            PreferedOrderCat = request.form.get('PreferedOrderCat')
            MaritalStatus = request.form.get('MaritalStatus')

            # Handling cases where object values might be None
            PreferredLoginDevice_ = len(PreferredLoginDevice) if PreferredLoginDevice else 0
            PreferredPaymentMode_ = len(PreferredPaymentMode) if PreferredPaymentMode else 0
            Gender_ = len(Gender) if Gender else 0
            PreferedOrderCat_ = len(PreferedOrderCat) if PreferedOrderCat else 0
            MaritalStatus_ = len(MaritalStatus) if MaritalStatus else 0

            # Scaling the data
            new_data = scaler.transform([[Tenure, PreferredLoginDevice_, CityTier, WarehouseToHome, PreferredPaymentMode_, Gender_, HourSpendOnApp, NumberOfDeviceRegistered, PreferedOrderCat_, SatisfactionScore, MaritalStatus_, NumberOfAddress, Complain, OrderAmountHikeFromlastYear, CouponUsed, OrderCount, DaySinceLastOrder, CashbackAmount]])

            # Make prediction using all features
            predict = model.predict(new_data)

            if predict[0] == 1:
                result = 'Stop! Churn detected!'
            else:
                result = 'Hooray! No churn celebration!'
            
            # Redirect to single_prediction page with the result
            return redirect(url_for('single_prediction', result=result))

        except Exception as e:
            logging.error(str(e))
            return str(e)

    elif request.method == 'GET':
        # Handle GET requests here if needed
        return render_template('home.html')

    else:
        # Handle other request methods if needed
        return "Method not allowed", 405  # Return 405 Method Not Allowed status

@app.route('/single_prediction/<result>')
def single_prediction(result):
    return render_template('single_prediction.html', result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0")
