# Import libraries
import numpy as np
from flask import Flask, request, jsonify
from modules.services.predict_service import PredictService
import json

app = Flask(__name__)
predictService = PredictService()

@app.route('/predict/accident_level',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)

    # Make prediction using model loaded from disk as per the data.
    predict_request = [[data["Year"], data["Month"], data["Day"], data["Weekday"], data["WeekofYear"], data["Season"],
                        data['TFIDF_activity'], data['TFIDF_area'], data['TFIDF_causing'], data['TFIDF_employee'],
                        data['TFIDF_hand'], data['TFIDF_injury'], data['TFIDF_left'], data['TFIDF_operator'],
                        data['TFIDF_right'], data['TFIDF_time'], data['TFIDF_causing injury'],
                        data['TFIDF_described injury'], data['TFIDF_employee reports'], data['TFIDF_finger left'],
                        data['TFIDF_injury described'], data['TFIDF_left foot'], data['TFIDF_left hand'],
                        data['TFIDF_medical center'], data['TFIDF_right hand'], data['TFIDF_time accident'],
                        data['TFIDF_causing injury described'], data['TFIDF_described time accident'],
                        data['TFIDF_finger left hand'], data['TFIDF_finger right hand'],
                        data['TFIDF_generating described injury'], data['TFIDF_hand causing injury'],
                        data['TFIDF_injury time accident'], data['TFIDF_left hand causing'],
                        data['TFIDF_right hand causing'], data['TFIDF_time accident employee'],
                        data['Country_02'], data['Country_03'], data['Local_02'], data['Local_03'], data['Local_04'],
                        data['Local_05'], data['Local_06'], data['Local_07'], data['Local_08'], data['Local_09'],
                        data['Local_10'],
                        data['Local_11'], data['Local_12'], data['Male'], data['IS_Mining'], data['IS_Others'],
                        data['EmpType_Third_Party'], data['EmpType_Third_Party_(Remote)'],
                        data['CR_Blocking_and_isolation_of_energies'], data['CR_Burn'],
                        data['CR_Chemical_substances'], data['CR_Confined_space'], data['CR_Cut'],
                        data['CR_Electrical_Shock'], data['CR_Electrical_installation'], data['CR_Fall'],
                        data['CR_Fall_prevention'], data['CR_Fall_prevention_(same_level)'],
                        data['CR_Individual_protection_equipment'], data['CR_Liquid_Metal'],
                        data['CR_Machine_Protection'], data['CR_Manual_Tools'], data['CR_Not_applicable'],
                        data['CR_Others'], data['CR_Plates'], data['CR_Poll'], data['CR_Power_lock'],
                        data['CR_Pressed'],
                        data['CR_Pressurized_Systems'],
                        data['CR_Pressurized_Systems_/_Chemical_Substances'], data['CR_Projection'],
                        data['CR_Projection/Burning'], data['CR_Projection/Choco'],
                        data['CR_Projection/Manual_Tools'], data['CR_Projection_of_fragments'],
                        data['CR_Suspended_Loads'], data['CR_Traffic'], data['CR_Vehicles_and_Mobile_Equipment'],
                        data['CR_Venomous_Animals'], data['CR_remains_of_choco']]]

    predict_request = np.array(predict_request)

    status, accidentLevel = predictService.predict_accidentLevel(predict_request)

    response = dict()
    response["status"] = status
    response["accidentLevel"] = str(accidentLevel)

    return jsonify(response)

if __name__ == '__main__':
    app.run(port=8111, debug=True)
