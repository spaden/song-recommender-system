from flask import Flask, request, jsonify
import json

from keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer

from flask_cors import CORS

import re
reg = re.compile(r'[a-zA-Z]')

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import random

import pickle

from sklearn.feature_extraction.text import CountVectorizer


stemming = PorterStemmer()
stop_list = set(stopwords.words('english'))
savedModel = load_model('emotion_review_model.h5')
vectorizer = pickle.load(open("vector.pickel", "rb"))

app = Flask(__name__)
CORS(app)


trainedModelOutput = open('trainPred.json',)
quotes_data = json.load(trainedModelOutput)

def remove_noise(quote_word):
    words = word_tokenize(quote_word)
    currentQuote = []
    for word in words:
        if word not in stop_list and word.isalpha() and word not in [",", ".", "!", " ", ";"] and reg.match(word):
                currentQuote.append(stemming.stem(word.lower()))

    return " ".join(currentQuote)


@app.route('/test')
def index():
    return "Hello World"

@app.route('/loginUser', methods=['POST'])
def loginUser():
    data = request.json

    response = jsonify(True)
    response.headers.add("Access-Control-Allow-Origin", "*")

    if data['name'] == 'Max':
        response = jsonify(True)
    else:
        response = jsonify(False)

    return response


@app.route("/getSimilarQuote", methods=['POST'])
def postTest():

    data = request.json

    print(data['userInput'])

    inp = remove_noise(data['userInput'])
    Xt = vectorizer.transform([inp])

    pred_group = savedModel.predict([Xt])[0]


    print(pred_group.argmax(axis=0))
    pred_group_list = quotes_data[str(pred_group.argmax(axis=0))]
    response = jsonify({'res': random.choice(pred_group_list)})
    response.headers.add("Access-Control-Allow-Origin", "*")

    return response



if __name__ == "__main__":
    app.run(debug=False, host='127.0.0.1', port = '8002', threaded=True)

