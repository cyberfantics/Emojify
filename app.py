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
        roi_gray = gray_frame[y:y+h, x:x+w]  # Extract face in grayscale
        resized_img = cv2.resize(roi_gray, (48, 48))  # Resize to 48x48
        resized_img = resized_img / 255.0  # Normalize pixel values
        resized_img = resized_img.reshape(1, 48, 48, 1)  # Add batch and channel dimensions

        # Make emotion prediction
        prediction = emotion_model.predict(resized_img)
        maxindex = int(np.argmax(prediction))

        # Draw rectangle and label on the image
        image_with_rectangle = image.copy()
        cv2.rectangle(image_with_rectangle, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            image_with_rectangle, 
            emotion_dict[maxindex], 
            (x, y-10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2, 
            cv2.LINE_AA
        )

        # Show the results
        st.image(image_with_rectangle, caption=f"Detected Emotion: {emotion_dict[maxindex]}", use_column_width=True)
        st.image(emoji_dist[maxindex], caption=f"Emoji: {emotion_dict[maxindex]}", width=200)
    else:
        st.warning("No face detected. Please try again.")
