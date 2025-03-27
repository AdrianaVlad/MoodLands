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

age_map=['0-3','4-7','8-13','14-20','21-32','33-43','44-59','60+']
emotion_map = ['angry','disgust','fear','happy','neutral','sad','surprise']



gender_model = load_model('../RNModel/gender_model_resnet.h5')
age_model = load_model("../RNModel/age_model_resnet.h5")
glasses_model = load_model("../RNModel/glasses_model_resnet.h5")
emotions_model = load_model("../RNModel/fer_model2.h5")

img = face_recognition.load_image_file('image1.png')
face_location1 = face_recognition.face_locations(img)
img1 = img[face_location1[0][0]-25: face_location1[0][2]+25, face_location1[0][3]-25: face_location1[0][1]+25]
img1_resize = cv2.resize(img1,(224,224))
img1_resize = np.array([img1_resize]).reshape((1, 224,224,3))

#F, 21-32: correct
gender1=gender_model.predict(img1_resize) 
age1=age_model.predict(img1_resize)


img = face_recognition.load_image_file('image2.png')
face_location2 = face_recognition.face_locations(img)
img2 = img[face_location2[0][0]-25: face_location2[0][2]+25, face_location2[0][3]-25: face_location2[0][1]+25]
img2_resize = cv2.resize(img2,(224,224))
img2_resize = np.array([img2_resize]).reshape((1, 224,224,3))

#M, 44-59:almost, predicted age 60+
gender2=gender_model.predict(img2_resize) 
age2=age_model.predict(img2_resize)

#parents and me, gender correct, age 2/3, the other 1 category too old
img = face_recognition.load_image_file('image3.png')
face_location3 = face_recognition.face_locations(img)
faces=[]
for face in face_location3:
    img3 = img[face[0]-25: face[2]+25, face[3]-25: face[1]+25]
    img3_resize = cv2.resize(img3,(224,224))
    faces.append(img3_resize)
faces = np.asarray(faces)
gender3=gender_model.predict(faces)
age3=age_model.predict(faces)

#wanted to see which face was witch so i did them separately too
img3 = img[face_location3[0][0]-25: face_location3[0][2]+25, face_location3[0][3]-25: face_location3[0][1]+25]
img3_resize = cv2.resize(img3,(224,224))
img3_resize = np.array([img3_resize]).reshape((1, 224,224,3))
gender3=gender_model.predict(img3_resize)
print("F") if gender3[0][0] >= 0.5 else print("M")
age3=age_model.predict(img3_resize)
print(age_map[age3.tolist()[0].index(max(age3.tolist()[0]))])
glasses3 = glasses_model.predict(img3_resize)
print("Glasses") if glasses3[0][0] >= 0.5 else print("No glasses")
img3_resize2 = cv2.resize(img3,(96,96))
img3_resize2 = np.array([img3_resize2]).reshape((1, 96,96,3))
emotions3 = emotions_model.predict(img3_resize2)
print(emotion_map[emotions3.tolist()[0].index(max(emotions3.tolist()[0]))])


img4 = img[face_location3[1][0]-25: face_location3[1][2]+25, face_location3[1][3]-25: face_location3[1][1]+25]
img4_resize = cv2.resize(img4,(224,224))
img4_resize = np.array([img4_resize]).reshape((1, 224,224,3))
gender4=gender_model.predict(img4_resize)
print("F") if gender4[0][0] >= 0.5 else print("M")
age4=age_model.predict(img4_resize)
print(age_map[age4.tolist()[0].index(max(age4.tolist()[0]))])
glasses4 = glasses_model.predict(img4_resize)
print("Glasses") if glasses4[0][0] >= 0.5 else print("No glasses")
img4_resize2 = cv2.resize(img4,(96,96))
img4_resize2 = np.array([img4_resize2]).reshape((1, 96,96,3))
emotions4 = emotions_model.predict(img4_resize2)
print(emotion_map[emotions4.tolist()[0].index(max(emotions4.tolist()[0]))])


img5 = img[face_location3[2][0]-25: face_location3[2][2]+25, face_location3[2][3]-25: face_location3[2][1]+25]
img5_resize = cv2.resize(img5,(224,224))
img5_resize = np.array([img5_resize]).reshape((1, 224,224,3))
gender5=gender_model.predict(img5_resize)
print("F") if gender5[0][0] >= 0.5 else print("M")
age5=age_model.predict(img5_resize)
print(age_map[age5.tolist()[0].index(max(age5.tolist()[0]))])
glasses5 = glasses_model.predict(img5_resize)
print("Glasses") if glasses5[0][0] >= 0.49 else print("No glasses")
img5_resize2 = cv2.resize(img5,(96,96))
img5_resize2 = np.array([img5_resize2]).reshape((1, 96,96,3))
emotions5 = emotions_model.predict(img5_resize2)
print(emotion_map[emotions5.tolist()[0].index(max(emotions5.tolist()[0]))])


##CONCLUSION: if age is wrong, only off by one category. occ worse that scc

import matplotlib.pyplot as plt
from noddingpigeon.inference import predict_video
import dlib

#live webcam v1:
prev_time = 0
prediction_interval = 10  # Perform predictions once per second
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../RNModel/shape_predictor_68_face_landmarks.dat")


def align_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = detector(gray)
    if len(faces) == 0:
        return image  # No face detected, return original image
    
    # Assume only one face in the image for simplicity
    face = faces[0]
    landmarks = predictor(gray, face)
    
    # Calculate angle of rotation based on facial landmarks
    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)
    angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
    print(angle)
    # Rotate image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated_image
# Function to process a frame
def process_frame(frame):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title('Face Resized')
    plt.axis('off')
    plt.show()
    # Find faces in the resized frame
    face_locations = face_recognition.face_locations(frame)
    if len(face_locations) == 0:
        return None, None, None, None, None, None, None, None

    # Process each detected face
    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_img_glasses = frame[top-60:bottom+40, left-50:right+50]
        face_img_resized = cv2.resize(face_img_glasses, (224, 224))

        # Reshape for model input
        face_img_resized = np.array([face_img_resized]).reshape((1, 224, 224, 3))
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(face_img_resized[0], cv2.COLOR_BGR2RGB))
        plt.title('Face Resized')
        plt.axis('off')
        glasses_prediction = glasses_model.predict(face_img_resized)
        # Extract the face region
        face_img_gender_age = frame[top-25:bottom+25, left-25:right+25]
        face_img_resized = cv2.resize(face_img_gender_age, (224, 224))

        # Reshape for model input
        face_img_resized = np.array([face_img_resized]).reshape((1, 224, 224, 3))

        # Perform predictions
        gender_prediction = gender_model.predict(face_img_resized)
        age_prediction = age_model.predict(face_img_resized)
        
        face_img_emotions = frame[top-15:bottom+15, left-15:right+15]
        emotions_img_resized = cv2.resize(face_img_emotions, (48, 48))
        emotions_img_resized_gray = cv2.cvtColor(emotions_img_resized, cv2.COLOR_BGR2GRAY)
        emotions_img_resized_gray = np.array([emotions_img_resized_gray]).reshape((1, 48, 48, 1))
        emotions_prediction = emotions_model.predict(emotions_img_resized_gray)
        
        #debug
        
        
        plt.subplot(1, 2, 2)
        plt.imshow(emotions_img_resized_gray[0, :, :, 0], cmap='gray')
        plt.title('Emotions Grayscale Resized')
        plt.axis('off')
        
        plt.show()
        print(gender_prediction)
        print(age_prediction)
        print(glasses_prediction)
        print(emotions_prediction)
        # Print predictions
        gender = "F" if gender_prediction[0][0] >= 0.5 else "M"
        age = age_map[age_prediction.tolist()[0].index(max(age_prediction.tolist()[0]))]
        glasses = "Glasses" if glasses_prediction[0][0] < 0.5 else "No glasses"
        emotion = emotion_map[emotions_prediction.tolist()[0].index(max(emotions_prediction.tolist()[0]))]
        print(f"Gender: {gender}, Age: {age}, Glasses: {glasses}, Emotion: {emotion}")
        return gender, age, glasses, emotion, top, right, bottom, left
        

# Capture video from webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    current_time = time.time()
    if current_time - prev_time >= prediction_interval:
        prev_time = current_time
        aligned_image = align_face(frame)
        gender, age, glasses, emotion, top, right, bottom, left = process_frame(aligned_image)
        if gender != None:
            cv2.putText(frame, f"Gender: {gender}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Age: {age}", (left, top + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Glasses: {glasses}", (left, top + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Emotion: {emotion}", (left, top + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        if gender != None:
            cv2.putText(frame, f"Gender: {gender}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Age: {age}", (left, top + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Glasses: {glasses}", (left, top + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Emotion: {emotion}", (left, top + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()