from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

## import ridge regresor model and standard scaler pickle 
ridge_model = pickle.load(open('model/ridge.pkl','rb'))
Standard_Scaler = pickle.load(open('model/scaler.pkl','rb'))

## Route for home page 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature=float(request.form.get("Temperature")) ## name se match kre 
        RH=float(request.form.get("RH"))
        Ws=float(request.form.get("Ws"))
        Rain=float(request.form.get("Rain"))
        FFMC=float(request.form.get("FFMC"))
        DMC=float(request.form.get("DMC"))
        ISI=float(request.form.get("ISI"))
        Classes=float(request.form.get("Classes"))
        Region=float(request.form.get("Region"))

        new_data_scaled = Standard_Scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge_model.predict(new_data_scaled)  ## list formate ma aape 

        return render_template("home.html",result=result[0])  ## pass the result into home.html

    else:
        return render_template("home.html")

if __name__=="__main__":
    app.run(host="0.0.0.0")
