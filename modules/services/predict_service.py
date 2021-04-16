import os
import numpy as np
import pickle
from keras.models import load_model

class PredictService:

    @classmethod
    def predict_accidentLevel(cls, predict_request):
        
        # Set current working directory
        cwd = os.getcwd()
        if cwd == '/app':
            os.chdir(cwd + '/modules/saved_models')
        
        arr = os.listdir('.')
        
        # Reset the directory
        os.chdir('/app')
        
        # Load the best model with f1-score
        if float(arr[0][:6]) > float(arr[1][:6]):
            filename = str(arr[0])
            # Load the model
            model = pickle.load(open(filename,'rb'))

        else:
            filename = str(arr[1])
            # Load the model
            model = load_model(filename)

        accidentLevel = (np.asarray(model.predict(predict_request))).round()
        status = 'success'

        return status, accidentLevel
