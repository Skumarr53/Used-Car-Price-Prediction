#!/usr/bin/env python
from flask import Flask, render_template, flash, request, jsonify, Markup
import logging, io, os, sys
import pandas as pd
import numpy as np
from modules.custom_transformers import *
#from sklearn.ensemble import GradientBoostingRegressor
import scipy
import pickle

## eb cli init
#>../aws-elastic-beanstalk-cli-setup/scripts/bundled_installer
#>echo 'export PATH="/home/skumar/.ebcli-virtual-env/executables:$PATH"' >> ~/.bash_profile && source ~/.bash_profile

#Freeing up port with Port no $Port_Number
#sudo fuser -k $Port_Number/tcp

# EB looks for an 'application' callable by default.
application = Flask(__name__)

np.set_printoptions(precision=2)

#Model features
gbm_model = None
features = ['Brand', 'Model', 'Location', 'Year', 'Kilometers_Driven', 
        'Fuel_Type', 'Transmission', 'Owner_Type', 'Mileage', 'Engine', 
        'Power', 'Seats']


@application.before_first_request
def startup():

    global gbm_model, model2brand
    
    # gbm model
    with open('static/GBM_Regressor_pipeline.pkl', 'rb') as f:
        gbm_model = pickle.load(f)

        # min, max, default values to categories mapping dictionary
    with open('static/Dictionaries.pkl', 'rb') as f:
        default_dict,min_dict, max_dict, default_dict_mapped = pickle.load(f)

    # Encoded values to categories mapping dictionary
    with open('static/Encoded_dicts.pkl', 'rb') as f:
        le_brands_Encdict,le_models_Encdict,le_locations_Encdict,le_fuel_types_Encdict,le_transmissions_Encdict,le_owner_types_Encdict = pickle.load(f)

    with open('static/model2brand.pkl', 'rb') as f:
        model2brand = pickle.load(f)

@application.errorhandler(500)
def server_error(e):
    logging.exception('some eror')
    return """
    And internal error <pre>{}</pre>
    """.format(e), 500

@application.route("/", methods=['POST', 'GET'])
def index():
     # Encoded values to categories mapping dictionary
      # Encoded values to categories mapping dictionary
    with open('static/Encoded_dicts.pkl', 'rb') as f:
        le_brands_Encdict,le_models_Encdict,le_locations_Encdict,le_fuel_types_Encdict,le_transmissions_Encdict,le_owner_types_Encdict = pickle.load(f)


    return render_template( 'index.html', model2brand = model2brand,le_models_Encdict = le_models_Encdict,le_locations_Encdict = le_locations_Encdict, le_fuel_types_Encdict = le_fuel_types_Encdict, le_transmissions_Encdict = le_transmissions_Encdict, le_owner_types_Encdict = le_owner_types_Encdict, le_brands_Encdict = le_brands_Encdict,price_prediction = 17.09)



# accepts either deafult values or user inputs and outputs prediction 
@application.route('/background_process', methods=['POST', 'GET'])
def background_process():
    Brand = request.args.get('Brand')                                        
    Model = request.args.get('Model')                                        
    Location = request.args.get('Location')
    Year = int(request.args.get('Year'))                                          
    Kilometers_Driven = float(request.args.get('Kilometers_Driven'))                
    Fuel_Type = request.args.get('Fuel_Type')
    Transmission = request.args.get('Transmission')
    Owner_Type = request.args.get('Owner_Type')
    Mileage = float(request.args.get('Mileage'))                                    
    Engine = float(request.args.get('Engine'))                                      
    Power = float(request.args.get('Power'))                                        
    Seats = float(request.args.get('Seats'))

	# values stroed in list later to be passed as df while prediction
    user_vals = [Brand, Model, Location, Year, Kilometers_Driven, 
        Fuel_Type, Transmission, Owner_Type, Mileage, Engine, 
        Power, Seats]


    x_test_tmp = pd.DataFrame([user_vals],columns = features)
    float_formatter = "{:.2f}".format

    pred = float_formatter(np.exp(gbm_model.predict(x_test_tmp[features])[0]))
    return jsonify({'price_prediction':pred})

# when running app locally
if __name__ == '__main__':
    application.debug = False
    application.run(host='0.0.0.0')