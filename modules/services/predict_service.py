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
   
        filename = ('finalised_ann_model.h5')

        # Load the model
        model = load_model(filename)

        accidentLevel = (np.asarray(model.predict(predict_request))).round()
        status = 'success'

        return status, accidentLevel
