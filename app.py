# -*- coding: utf-8 -*-
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import numpy as np
import pandas as pd
import pickle
import joblib
from flask import Flask, request, render_template
from flask_cors import cross_origin

# Load ML model
model_heart = pickle.load(open('heartmodel.pkl', 'rb')) 
model_diabetes = pickle.load(open('diabetesmodel.pkl', 'rb'))
model_stroke = pickle.load(open("strokemodel.pkl", 'rb'))
model_breastcancer = pickle.load(open('breastcancermodel.pkl', 'rb'))
scaler1 = pickle.load(open('scaler1.pkl', 'rb'))
model_cancer = pickle.load(open('cancermodel.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
# Create application
app = Flask(__name__)
# Bind home function to URL
@app.route('/')
@cross_origin()
def home():
    return render_template('home.html')

@app.route('/diagnosis')
@cross_origin()
def diagnosis():
    return render_template('diagnosis.html')

@app.route('/about')
@cross_origin()
def about():
    return render_template('about.html')

@app.route('/consultation')
@cross_origin()
def consultation():
    return render_template('consultation.html')

@app.route('/heartdisease')
@cross_origin()
def heart():
    return render_template('heartdisease.html')

# Bind predict function to URL
@app.route('/predictheartdisease', methods =["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        # Put all form entries values in a list 
        features = [float(i) for i in request.form.values()]
        # Convert features to array
        array_features = [np.array(features)]
        # Predict features
        prediction = model_heart.predict(array_features)
        
        output = prediction
        
        # Check the output values and retrive the result with html tag based on the value
        if output == 1:
            return render_template('lowrisk.html')
        else:
            return render_template('highrisk.html')
    return render_template('heartdisease.html')
@app.route('/diabetes')
def diabetes():
	return render_template('diabetes.html')

@app.route('/predictdiabetes', methods =["GET", "POST"])
def predictdiabetes():
    if request.method == 'POST':
        preg = request.form['pregnancies']
        glucose = request.form['glucose']
        bp = request.form['bloodpressure']
        st = request.form['skinthickness']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        dpf = request.form['dpf']
        age = request.form['age']
        
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        dbprediction = model_diabetes.predict(data)
        
        if dbprediction == 1:
            return render_template('lowrisk.html')
        else:
            return render_template('highrisk.html')
    return render_template('diabetes.html')
 
@app.route('/stroke')
def stroke():
    return render_template('stroke.html')

@app.route('/predictstroke', methods =["GET", "POST"])
def predictstroke():
    gender=int(request.form['gender'])
    age=int(request.form['age'])
    hypertension=int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    ever_married = int(request.form['ever_married'])
    work_type = int(request.form['work_type'])
    Residence_type = int(request.form['Residence_type'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])
    smoking_status = int(request.form['smoking_status'])

    x=np.array([gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,
                avg_glucose_level,bmi,smoking_status]).reshape(1,-1)


    x=scaler1.transform(x)

    
    dt=joblib.load('dt.sav')

    Y_pred=dt.predict(x)

    # for No Stroke Risk
    if Y_pred==0:
        return render_template('lowrisk.html')
    else:
        return render_template('highrisk.html')
@app.route('/liverdisease')
def liver():
    return render_template("liverdisease.html")   
def ValuePred(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==7):
        loaded_model = joblib.load('liver_model.pkl')
        result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/predictliver', methods=["GET", "POST"])
def predictliver():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        if len(to_predict_list) == 7:
            result = ValuePred(to_predict_list, 7)

    if int(result) == 1:
        return render_template('highrisk.html')
    else:
        return render_template('lowrisk.html')

@app.route("/kidneydisease")
def kidney():
    return render_template("kidneydisease.html")

def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)
    if size == 7:
        loaded_model = joblib.load('kidney_model.pkl')
        result = loaded_model.predict(to_predict)
    return result[0]
@app.route("/predictkidney",  methods=['GET', 'POST'])
def predictkidney():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        if len(to_predict_list) == 7:
            result = ValuePredictor(to_predict_list, 7)
    if(int(result) == 1):
        return render_template("highrisk.html")
    else:
        return render_template("lowrisk.html")
    
@app.route("/breastcancer")
def breastcancercancer():
    return render_template("breastcancer.html")  

@app.route('/predictbreastcancer',methods=['POST'])
def predictcancer():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    final_features = scaler.transform(final_features)    
    prediction = model_cancer.predict(final_features)
    y_probabilities_test = model_cancer.predict_proba(final_features)
    y_prob_success = y_probabilities_test[:, 1]
    print("final features",final_features)
    print("prediction:",prediction)
    output = round(prediction[0], 2)
    y_prob=round(y_prob_success[0], 3)
    print(output)

    if output == 0:
        return render_template('lowrisk.html')
    else:
         return render_template('highrisk.html')

    



if __name__ == '__main__':
#Run the application
    app.run()
    
    