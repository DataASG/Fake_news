import json
import os
import joblib

from google.cloud import storage
from google.oauth2 import service_account
from termcolor import colored
PROJECT_ID = 'fakenews-296818'
BUCKET_NAME='fakenews475'
BUCKET_TRAINING_FOLDER='trainings'
REGION='europe-west1'
PYTHON_VERSION=3.7
PACKAGE_NAME='Fake_news'
MODEL_NAME = 'LSTM'

def get_credentials():
    credentials_raw = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if '.json' in credentials_raw:
        credentials_raw = open(credentials_raw).read()
    creds_json = json.loads(credentials_raw)
    creds_gcp = service_account.Credentials.from_service_account_info(creds_json)
    return creds_gcp

def storage_upload(model_version=1, bucket=BUCKET_NAME, rm=False, filename='pipeline.pkl'):
    client = storage.Client().bucket(bucket)

    storage_location = 'models/{}/versions/{}/{}'.format(
        MODEL_NAME,
        model_version,
        filename)
    blob = client.blob(storage_location)
    blob.upload_from_filename(filename)


def download_model(model_version=1, bucket=BUCKET_NAME, rm=True):
    creds = get_credentials()
    client = storage.Client(credentials=creds, project=PROJECT_ID).bucket(bucket)

    storage_location = 'models/{}/versions/{}/{}'.format(
        MODEL_NAME,
        model_version,
        'model.joblib')
    blob = client.blob(storage_location)
    blob.download_to_filename('model.joblib')
    print(f"=> pipeline downloaded from storage")
    model = joblib.load('model.joblib')
    if rm:
        os.remove('model.joblib')
    return model
