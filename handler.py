import pandas as pd
import pickle
import os
from flask import Flask, request, Response
from propensao.Propensao import Propensao

#carregar modelo
model = pickle.load (open ('model/xgbclassifier_final_model.pkl', 'rb') )

app = Flask (__name__)

@app.route('/propensao/predict', methods=['POST'])
def propensao_predict():
    test_json = request.get_json()

    if test_json: 
        if isinstance( test_json, dict ):
            test_raw = pd.DataFrame( test_json, index=[0] )
            
        else: 
            test_raw = pd.DataFrame( test_json, columns=test_json[0].keys() )
   
    
        pipeline = Propensao() 
        
        #cleaning
        df1 = pipeline.data_cleaning(test_raw)
        
        #preparation
        df2 = pipeline.data_preparation(df1)
        
        #prediction
        df_response = pipeline.get_predict(model, test_raw, df2 )
        
        return df_response
    
    else:
        return Response('{}', status=200, mimetype='application/json')
    
if __name__ == '__main__':
    port = os.environ.get('PORT',5000)
    app.run( host='0.0.0.0', port=port )
    