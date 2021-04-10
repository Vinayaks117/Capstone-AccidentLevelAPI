import os
import numpy as np
import pickle
from keras.models import load_model

class PredictService:

    @classmethod
    def predict_accidentLevel(cls, predict_request):
        cwd = os.getcwd()
        filelocation = (cwd + '/modules/saved_models/')

        #os.chdir('C:/Users/Prakash/PycharmProjects/AutoMLCapstoneProject/modules/saved_models')
        #filelocation = (cwd + '/PycharmProjects/AutoMLCapstoneProject/modules/saved_models/')  # For Linux
        filename = (filelocation + 'finalised_ann_model.h5')

        # Load the model
        #model = pickle.load(open('finalised_ml_model.sav','rb'))
        model = load_model(filename)

        accidentLevel = (np.asarray(model.predict(predict_request))).round()
        status = 'success'

        return status, accidentLevel