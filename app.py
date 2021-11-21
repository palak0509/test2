from __future__ import division, print_function

import numpy as np
import pandas as pd
#import joblib as joblib

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify

app = Flask(__name__)

#ratingsMatrix = joblib.load('user_rating.pkl')
#productClass = joblib.load('sentiment_class.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/", methods=['POST'])
def home():
    flag = False
    data = ""
    if request.method == 'POST':
        flag = True
        user = request.form["userid"]
        data=recommend.getTopProducts(user)
        return render_template('index.html', data=data, flag=flag)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=5000)
    app.debug=True
    app.run()
