import os
import joblib
import pandas as pd

from flask import Flask
from flask import render_template, request, redirect
from werkzeug.utils import secure_filename

import mlflow.sklearn
import model_utils

app = Flask(__name__)
#model = joblib.load("static/model/model.joblib")
model = mlflow.sklearn.load_model(
        model_uri=f's3://co.techtutor/mlflow/artifacts/0/32af3e573b234271a5b6eee2d26cd076/artifacts/model')

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = ['csv']


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
def index():
    file_path = ''
    dataset = []
    if request.method == 'POST':
        if 'file' not in request.files:
            error = 'No file part'
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            error = 'No selected file'
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save('static/' + file_path)
            dataset = pd.read_csv('static/' + file_path)
            dataset_encoded = model_utils.encode_dataset(dataset)

            dataset['prediction'] = model.predict(dataset_encoded)
            dataset = dataset.to_dict(orient='records')
            print("type(dataset):")
            print(type(dataset))
            print(dataset[0])

    return render_template(
        'index.html',
        file_path=file_path,
        dataset=dataset
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)

