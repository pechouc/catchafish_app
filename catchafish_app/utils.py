import json
import numpy as np
import pandas as pd
from skimage.io import imread, imsave
import os

from google.cloud import storage
import googleapiclient.discovery
from google.oauth2 import service_account

PROJECT_ID='wagon-project-catchafish'
BUCKET_NAME='catchafish_gcp'

BUCKET_MODEL_NAME='vgg16'
MODEL_VERSION='v1'

NAMES_MAPPING = {
    0 : ("fish_01", "Dascyllus reticulatus"),
    1 : ("fish_02", "Plectroglyphidodon dickii"),
    2 : ("fish_03", "Chromis chrysura"),
    3 : ("fish_04", "Amphiprion clarkii"),
    4 : ("fish_05", "Chaetodon lunulatus"),
    5 : ("fish_07", "Myripristis kuntee"),
    6 : ("fish_08", "Acanthurus nigrofuscus"),
    7 : ("fish_09", "Hemigymnus fasciatus"),
    8 : ("fish_10", "Neoniphon sammara"),
    9 : ("fish_16", "Lutjanus fulvus"),
    10: ("fish_17", "Carassius auratus")
    }

def get_credentials():
    credentials_raw = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')

    if '.json' in credentials_raw:
        credentials_raw = open(credentials_raw).read()

    creds_json = json.loads(credentials_raw)
    creds_gcp = service_account.Credentials.from_service_account_info(creds_json)

    return creds_gcp

def predict(project, model, version, instances):
    '''Call model for prediction'''

    service = googleapiclient.discovery.build('ml', 'v1') # google api endpoint /ml/v1

    name = 'projects/{}/models/{}/versions/{}'.format(project, model, version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    results = response['predictions']

    return np.argmax(results[0]['dense_1'])

def get_additional_images(predicted_class):
    creds = get_credentials()
    client = storage.Client(credentials = creds, project = PROJECT_ID)
    client = client.bucket(BUCKET_NAME)

    folder_name = NAMES_MAPPING[predicted_class][0]
    storage_location = f'data/{folder_name}/'

    storage_location_0 = storage_location + '0.jpg'
    blob_0 = client.blob(storage_location_0)
    blob_0.download_to_filename('img_0.jpg')
    img_0 = imread('img_0.jpg')

    storage_location_1 = storage_location + '1.jpg'
    blob_1 = client.blob(storage_location_1)
    blob_1.download_to_filename('img_1.jpg')
    img_1 = imread('img_1.jpg')

    storage_location_2 = storage_location + '2.jpg'
    blob_2 = client.blob(storage_location_2)
    blob_2.download_to_filename('img_2.jpg')
    img_2 = imread('img_2.jpg')

    return img_0, img_1, img_2

def add_image_to_dataset(image, species):
    folder_name = [elt[0] for elt in NAMES_MAPPING.values() if elt[1] == species][0]
    return f'Uploaded successfully to /data/{folder_name} - Dummy message'

