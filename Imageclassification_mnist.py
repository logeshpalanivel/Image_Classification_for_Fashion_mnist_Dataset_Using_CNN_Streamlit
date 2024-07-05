import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

def load_model():
    model = tf.keras.models.load_model('model.hdf5')
    return model

with st.spinner('Model is being loaded'):
    model = load_model()

st.title("Fashion Classification")
file = st.file_uploader("Please upload a dress image", type=['jpg', 'png', 'AVIF'])

def predict(image_data, model):
    size = (28, 28)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert image to grayscale
    img_reshape = img.reshape((1, 28, 28, 1))  # Reshape for the model
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = predict(image, model)
    score = tf.nn.softmax(predictions[0])

    st.write(predictions)
    st.write(score)

    class_names = ['0','1','2','3','4','5','6','7','8','9']
    st.write(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )







 