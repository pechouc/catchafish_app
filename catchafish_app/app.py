import streamlit as st
from skimage.io import imread,imsave
from skimage.transform import resize
import numpy as np
import wikipedia

from catchafish_app.utils import predict
from catchafish_app.utils import PROJECT_ID, BUCKET_NAME, BUCKET_MODEL_NAME, MODEL_VERSION

#Display title
st.title('Catch a fish !')

#upload image to predict
uploaded_image = st.file_uploader("Choose image to predict")
if uploaded_image is not None:
    image = imread(uploaded_image)
    st.image(image)
    image = resize(image, (128, 128))
    image = np.expand_dims(image, axis = 0)

#predict fish
if uploaded_image:
    instances = image.tolist()

    result = predict(
        project=PROJECT_ID,
        model=BUCKET_MODEL_NAME,
        version=MODEL_VERSION,
        instances=instances
        )

    st.markdown(f'{result}')
    st.markdown('This fish is a [gold fish](https://fr.wikipedia.org/wiki/Poisson-clown).')
    st.markdown(wikipedia.summary("goldfish",sentences=2))
