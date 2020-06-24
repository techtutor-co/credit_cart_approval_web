import joblib

from flask import Flask
from flask import render_template, request

app = Flask(__name__)
#model = joblib.load("../models/classifier.pkl")


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)

