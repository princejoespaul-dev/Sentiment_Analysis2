import os
import joblib
import numpy as np
from preprocess import preprocess
from flask import Flask, request, jsonify


app = Flask(__name__)

#LOADING SAVED MODEL
model = joblib.load('model.joblib')

#LOADING SAVED TF-IDF VECOTORIZER
vectorizer = joblib.load('tf_idf.joblib')

@app.route("/predict",methods=['POST'])

def predict():

    #READING THE INCOMING JSON DATA
    data = request.get_json()

    #EXTRACT THE FEATURE VALUES FROM MESSAGE
    reviews = data['reviews']

    #PREPROCESS THE REVIEW POSTED
    process_review = preprocess(reviews)

    #EXTRACT FEATURES FROM THE REVIEW
    process_review = vectorizer.transform([process_review])

    #PREDICT THE SENTIMENT
    prediction = model.predict(process_review)

    #OUTPUT THE SENTIMENT
    return jsonify({"prediction":prediction[0]})


#USED TO CHECK IF THE SERVER IS RUNNING. IT WILL OUTPUT {"STATUS":"OK"} IF SERVER IS RUNNING
@app.route("/health",methods=['GET'])
def health():
    return jsonify({"status":"OK"})


if __name__ == "__main__":
    # app.run(debug=True,host="0.0.0.0",port=5000
    port = int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0',port=port)
