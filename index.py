import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input
from PIL import Image
import numpy as np

# Load model
model = MobileNetV2(weights="imagenet")

st.title("Image Classification App")
st.write("Upload an image and let the model classify it")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    predictions = model.predict(img_array)
    decoded_preds = decode_predictions(predictions, top=3)[0]

    st.write("### Predictions:")
    for i, (imagenet_id, label, prob) in enumerate(decoded_preds):
        st.write(f"{i+1}. {label} ({prob*100:.2f}%)")
