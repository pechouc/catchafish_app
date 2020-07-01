import streamlit as st
from skimage.io import imread, imsave
from skimage.transform import resize
import numpy as np
import wikipedia
import os

from catchafish_app.utils import predict, get_additional_images, add_image_to_dataset
from catchafish_app.utils import PROJECT_ID, BUCKET_NAME, BUCKET_MODEL_NAME, MODEL_VERSION, NAMES_MAPPING

#credentials = os.environ["GOOGLE_CREDENTIALS"] #dictionary
#json_file = open(os.environ['GOOGLE_APPLICATION_CREDENTIALS'], 'w')
#json_file.write(credentials)
#json_file.close()

mode = st.sidebar.selectbox("👇 Select your profile", ["Curious", "Expert"])

if mode == "Curious":

    # Display title
    st.title('Catch a fish!')

    # Upload image to predict
    uploaded_image = st.file_uploader("Upload an image and get the species of your fish 🐠")

    if uploaded_image is not None:
        # Storing the image into a NumPy array and plotting it
        image = imread(uploaded_image)
        st.image(image, use_column_width = True)

        # Predict fish
        image_preprocessed = resize(image, (128, 128))
        image_preprocessed = np.expand_dims(image_preprocessed, axis = 0)

        instances = image_preprocessed.tolist()

        predicted_class = predict(
            project=PROJECT_ID,
            model=BUCKET_MODEL_NAME,
            version=MODEL_VERSION,
            instances=instances
            )

        result = NAMES_MAPPING[predicted_class][1]

        # Display model prediction
        if result.startswith(('A', 'E', 'I', 'O', 'U')):
            st.markdown(f'## 👉 &nbsp;This fish is an **{result}**')
        else:
            st.markdown(f'## 👉 &nbsp;This fish is a **{result}**')

        # Get information from wikipedia and display it
        scrapped = wikipedia.summary(result, sentences = 2)

        if '\n' in scrapped:
            scrapped = scrapped.replace('\n', '').replace('.', '. ')

        if '==' in scrapped:
            scrapped = wikipedia.summary(result, sentences = 1)

        st.markdown(scrapped)

        # Adding a separator
        st.markdown('---')

        # Displaying other images from the same class
        st.markdown('## 📷 &nbsp;Other images posted on the app for this species')

        img_0, img_1, img_2 = get_additional_images(predicted_class)
        st.image([img_0, img_1, img_2], width = 210)
        os.system('rm img_0.jpg img_1.jpg img_2.jpg')

        # Adding a separator
        st.markdown('---')


elif mode == "Expert":

    # Display title
    st.title('Catch a fish!')

    # Upload image to predict
    uploaded_image = st.file_uploader("Upload an image and get the species of your fish 🐠")

    if uploaded_image is not None:
        # Storing the image into a NumPy array and plotting it
        image = imread(uploaded_image)
        st.image(image, use_column_width = True)

        # Predict fish
        image_preprocessed = resize(image, (128, 128))
        image_preprocessed = np.expand_dims(image_preprocessed, axis = 0)

        instances = image_preprocessed.tolist()

        predicted_class = predict(
            project=PROJECT_ID,
            model=BUCKET_MODEL_NAME,
            version=MODEL_VERSION,
            instances=instances
            )

        result = NAMES_MAPPING[predicted_class][1]

        # Display model prediction
        if result.startswith(('A', 'E', 'I', 'O', 'U')):
            st.markdown(f'## 👉 This fish is an **{result}**')
        else:
            st.markdown(f'## 👉 This fish is a **{result}**')

        # Get information from wikipedia and display it
        scrapped = wikipedia.summary(result, sentences = 2)

        if '\n' in scrapped:
            scrapped = scrapped.replace('\n', '').replace('.', '. ')

        if '==' in scrapped:
            scrapped = wikipedia.summary(result, sentences = 1)

        st.markdown(scrapped)

        # Adding a separator
        st.markdown('---')

        # Giving the possibility to annotate the image
        st.markdown('## ⁉️ You feel like the identification is wrong?')
        st.markdown('If you are positive the algorithm is mistaken on this call, please provide us with the expected species and this additional data point will be used for further model training!')
        species = ['Mysterius fishii?'] + [elt[1] for elt in NAMES_MAPPING.values()]
        correct_pred = st.selectbox('One small step for man, one giant leap for the model 🔭', species)

        # Adding a separator
        st.markdown('---')

        # Feeding the new data point to the model
        if correct_pred != 'Mysterius fishii?':
            message = add_image_to_dataset(image, correct_pred)
            st.markdown(f'*{message}*')
