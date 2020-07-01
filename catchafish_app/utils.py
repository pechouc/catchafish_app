import json
import numpy as np
import pandas as pd
from google.cloud import storage
import googleapiclient.discovery

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

    predicted_class = np.argmax(results[0]['dense_1'])

    return NAMES_MAPPING[predicted_class][1]
