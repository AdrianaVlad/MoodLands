import keras
import json
import sys
import tensorflow as tf
from keras.layers import Input
import numpy as np
import argparse
from keras.utils.data_utils import get_file
import face_recognition
import cv2
from tensorflow.keras.models import load_model
import time
import dlib
age_map=['0-3','4-7','8-13','14-20','21-32','33-43','44-59','60+']
emotion_map = ['anger','contempt','disgust','fear','happy','neutral','sad','surprise']

gender_model = load_model("./gender_model_resnet.h5")
age_model = load_model("./age_model_resnet.h5")
glasses_model = load_model("./glasses_model_resnet.h5")
emotions_model = load_model("./affectnet_model.h5")

prev_time = 0
prediction_interval = 2
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def align_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return image 
    face = faces[0]
    landmarks = predictor(gray, face)
    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)
    angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated_image

def process_frame(frame):
    face_locations = face_recognition.face_locations(frame)
    if len(face_locations) == 0:
        return None, None, None, None, None
    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_img_glasses = frame[top-max(top,60):bottom+max(frame.shape[0],40), left-max(left,50):right+max(frame.shape[1],50)]
        face_img_resized = cv2.resize(face_img_glasses, (224, 224))
        face_img_resized = np.array([face_img_resized]).reshape((1, 224, 224, 3))
        glasses_prediction = glasses_model.predict(face_img_resized)
        face_img_gender_age = frame[top-max(top,25):bottom+max(frame.shape[0],25), left-max(left,25):right+max(frame.shape[1],25)]
        face_img_resized = cv2.resize(face_img_gender_age, (224, 224))
        face_img_resized = np.array([face_img_resized]).reshape((1, 224, 224, 3))
        gender_prediction = gender_model.predict(face_img_resized)
        age_prediction = age_model.predict(face_img_resized)
        face_img_emotions = frame[top-10:bottom+10, left-10:right+10]
        if face_img_emotions.size == 0:
            face_img_emotions = frame
        emotions_img_resized = cv2.resize(face_img_emotions, (96, 96))
        emotions_img_resized_gray = np.array([emotions_img_resized]).reshape((1, 96, 96, 3))
        emotions_prediction = emotions_model.predict(emotions_img_resized_gray)
        gender = "F" if gender_prediction[0][0] >= 0.5 else "M"
        age_index = age_prediction.tolist()[0].index(max(age_prediction.tolist()[0]))
        age = age_map[age_index]
        print(emotions_prediction)
        glasses = 1 - glasses_prediction[0][0]
        emotion = emotion_map[emotions_prediction.tolist()[0].index(max(emotions_prediction.tolist()[0]))]
        print(emotion)
        return gender, age_index, age, glasses, emotion
        
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    output_file = open("predictions.txt", "w")
    output_file.write("NO WEBCAM")
    output_file.close()
    sys.exit()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    current_time = time.time()
    if current_time - prev_time >= prediction_interval:
        prev_time = current_time
        aligned_image = align_face(frame)
        gender, age_index, age, glasses, emotion = process_frame(aligned_image)
        prediction = {
            'gender': gender,
            'ageIndex': age_index,
            'age': age,
            'hasGlasses': glasses,
            'emotion': emotion
        }
        with open("predictions.txt", "w") as output_file:
            json.dump(prediction, output_file)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()