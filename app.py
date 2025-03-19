import streamlit as st
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from PIL import Image

# Load emotion model
emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('emotion_model.weights.h5')

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
emoji_dist = {0: "./emojis/angry.png", 1: "./emojis/disgusted.png", 2: "./emojis/fearful.png", 3: "./emojis/happy.png", 
              4: "./emojis/neutral.png", 5: "./emojis/sad.png", 6: "./emojis/surprised.png"}

# Streamlit UI
st.title("Photo to Emoji - Emotion Detection")
st.write("Upload an image or use your webcam to detect emotions and display corresponding emojis.")

# Webcam input
image_file = st.camera_input("Take a photo")

if image_file is not None:
    image = Image.open(image_file)
    image = np.array(image.convert('RGB'))
    gray_frame = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        roi_gray = gray_frame[y:y+h, x:x+w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        
        st.image(image, caption=f"Detected Emotion: {emotion_dict[maxindex]}", use_column_width=True)
        emoji_path = emoji_dist[maxindex]
        st.image(emoji_path, caption=f"Emoji: {emotion_dict[maxindex]}", width=200)
    else:
        st.warning("No face detected. Please try again.")
