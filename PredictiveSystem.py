# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

#loading the saved model 
loaded_model = pickle.load(open('C:/Users/apk92/Desktop/Priyanka/Supervised learning/trained_model.sav','rb'))
input_data = (56,1,3,130,256,1,2,142,1,0.6,2,1,6)

#change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the numpy array as we are predicting for only an instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 0):
    print('The Person has a Healthy Heart')
else : 
    print('The Person has Heart Disease')
