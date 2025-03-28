import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

# Load the pre-trained emotion model
emotion_model = load_model('emotion_model.h5')

# Emotion and emoji mapping
emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful", 
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
}
emoji_dist = {
    0: "./emojis/angry.png", 1: "./emojis/disgusted.png", 
    2: "./emojis/fearful.png", 3: "./emojis/happy.png",
    4: "./emojis/neutral.png", 5: "./emojis/sad.png", 
    6: "./emojis/surprised.png"
}

# Streamlit UI
st.title("Photo to Emoji - Emotion Detection")
st.write("Upload an image or use your webcam to detect emotions and display corresponding emojis.")

# Webcam input
image_file = st.camera_input("Take a photo")

if image_file is not None:
    image = Image.open(image_file)
    image = np.array(image.convert('RGB'))  # Ensure it's in RGB format

    # Load OpenCV face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray_frame = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        roi_color = image[y:y+h, x:x+w]  # Extract face in color

        # Resize to match model's expected input (224x224x3)
        resized_img = cv2.resize(roi_color, (224, 224))
        resized_img = np.expand_dims(resized_img, axis=0)  # Add batch dimension

        # Make emotion prediction
        prediction = emotion_model.predict(resized_img)
        maxindex = int(np.argmax(prediction))

        # Display results
        st.image(image, caption=f"Detected Emotion: {emotion_dict[maxindex]}", use_column_width=True)
        st.image(emoji_dist[maxindex], caption=f"Emoji: {emotion_dict[maxindex]}", width=200)
    else:
        st.warning("No face detected. Please try again.")
