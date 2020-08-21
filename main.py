import pandas as pd
import numpy as np
import joblib

from fastapi import FastAPI
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

#LOAD MODEL
model = joblib.load('./model/model_rf.joblib')

@app.get("/predict/")
def predict(cnae_0:int, cnae_1:int, cnae_2:int, cnae_3:int, score:float, pout_s12:float, pout_c12:float, pin_a12:float):
    """query with features values to get prediction
    
    the order should be:
    cnae_0 - cnae_1 - cnae_2 - cnae_3 - Score_ProScore_cpf - payments_out_sum_12m - payments_out_count_12m - payments_in_avg_12m
    
    """

    model_input = [cnae_0,
                   cnae_1,
                   cnae_2,
                   cnae_3,
                   score, 
                   pout_s12, 
                   pout_c12,
                   pin_a12]

    prob = model.predict_proba(np.array(model_input).reshape(1, -1))[0][1]
    pred = model.predict(np.array(model_input).reshape(1, -1))[0]

    return {"prediction": pred,
            "probability": prob }

    
