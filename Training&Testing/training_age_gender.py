import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
import face_recognition
import cv2
import seaborn as sns
from sklearn.metrics import confusion_matrix

def testing(model,x_test,y_test):
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = y_test.values.flatten()

    correct_guesses = (y_pred_classes == y_test_classes)
    wrong_guesses = ~correct_guesses
    correct_counts = Counter(y_test_classes[correct_guesses])
    wrong_counts = Counter(y_test_classes[wrong_guesses])
    categories = np.unique(y_test_classes)
    correct_counts = {cat: correct_counts.get(cat, 0) for cat in categories}
    wrong_counts = {cat: wrong_counts.get(cat, 0) for cat in categories}

    correct = [correct_counts[cat] for cat in categories]
    wrong = [wrong_counts[cat] for cat in categories]

    bar_width = 0.35
    index = np.arange(len(categories))
    fig, ax = plt.subplots()
    bar1 = ax.bar(index, correct, bar_width, label='Correct')
    bar2 = ax.bar(index + bar_width, wrong, bar_width, label='Wrong')
    ax.set_xlabel('Category')
    ax.set_ylabel('Counts')
    ax.set_title('Correct and Wrong Guesses per Category')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(categories)
    ax.legend()

    for bar in bar1 + bar2:
        height = bar.get_height()
        ax.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.show()

def display_split(y_test, y_train):
    unique_train_labels, train_counts = np.unique(y_train, return_counts=True)
    unique_test_labels, test_counts = np.unique(y_test, return_counts=True)

    labels = np.union1d(unique_train_labels, unique_test_labels)
    train_counts = [train_counts[unique_train_labels.tolist().index(label)] if label in unique_train_labels else 0 for label in labels]
    test_counts = [test_counts[unique_test_labels.tolist().index(label)] if label in unique_test_labels else 0 for label in labels]

    fig, ax = plt.subplots()
    bar_width = 0.35
    index = np.arange(len(labels))

    bar1 = ax.bar(index, train_counts, bar_width, label='Train')
    bar2 = ax.bar(index + bar_width, test_counts, bar_width, label='Test')
    ax.set_xlabel('Label')
    ax.set_ylabel('Count')
    ax.set_title('Label Distribution in Train and Test Sets')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(labels)
    ax.legend()

    for bar in bar1 + bar2:
        height = bar.get_height()
        ax.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.show()

def sim_unity_comm(test_images):
    webcam_sim_img = []
    for img in test_images:
        face_location = face_recognition.face_locations(img)
        if len(face_location) != 0:
            expand_by = int(img.shape[0] * 0.05)
            top = max(face_location[0][0] - expand_by, 0)
            right = min(face_location[0][1] + expand_by, img.shape[1])
            bottom = min(face_location[0][2] + expand_by, img.shape[0])
            left = max(face_location[0][3] - expand_by, 0)
            img_face = img[top:bottom, left:right]
            img_face_resize = cv2.resize(img_face, (224, 224))
        webcam_sim_img.append(img_face_resize)
    test_images_sim = np.asarray(webcam_sim_img)
    return test_images_sim
def plot_confusion_matrix(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = y_test.values.flatten()

    cm = confusion_matrix(y_test_classes, y_pred_classes)
    class_names = np.unique(y_test_classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

data0 = pd.read_csv("fold_0_data.txt",sep = "\t" )
data1 = pd.read_csv("fold_1_data.txt",sep = "\t")
data2 = pd.read_csv("fold_2_data.txt",sep = "\t")
data3 = pd.read_csv("fold_3_data.txt",sep = "\t")
data4 = pd.read_csv("fold_4_data.txt",sep = "\t")
all_labeled_data = pd.concat([data0, data1, data2, data3, data4], ignore_index=True)

print(len(all_labeled_data))

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dropout, Flatten, Dense, Dropout, LayerNormalization, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

relevant_data = all_labeled_data[['age', 'gender']].copy()
img_path = []
for row in all_labeled_data.iterrows():
    img_path.append("faces/" + row[1].user_id + "/coarse_tilt_aligned_face." + str(row[1].face_id) + "." + row[1].original_image)

relevant_data['img_path'] = img_path

relevant_data['age'].unique()
relevant_data.groupby(['age']).count()

age_mapping = [('(0, 2)', '0-3'), ('2', '0-3'), ('3', '0-3'), ('(4, 6)', '4-7'), ('(8, 12)', '8-13'), ('13', '8-13'), ('22', '21-32'), ('(8, 23)','14-20'), ('23', '21-32'), ('(15, 20)', '14-20'), ('(25, 32)', '21-32'), ('(27, 32)', '21-32'), ('32', '21-32'), ('34', '33-43'), ('29', '21-32'), ('(38, 42)', '33-43'), ('35', '33-43'), ('36', '33-43'), ('42', '33-43'), ('45', '44-59'), ('(38, 43)', '33-43'), ('(38, 42)', '33-43'), ('(38, 48)', '33-43'), ('46', '44-59'), ('(48, 53)', '44-59'), ('55', '44-59'), ('56', '44-59'), ('(60, 100)', '60+'), ('57', '44-59'), ('58', '44-59')]
relevant_data = relevant_data.dropna(subset=['age'])
for mapping in age_mapping:
    relevant_data['age'] = relevant_data['age'].replace(mapping[0], mapping[1])
clean_data = relevant_data[relevant_data.gender != 'u'].copy()
clean_data = clean_data.dropna(subset=['gender'])

gender_to_label_map = {
    'f' : 0,
    'm' : 1
}
clean_data['gender'] = clean_data['gender'].apply(lambda g: gender_to_label_map[g])

age_to_label_map = {
    '0-3'  :0,
    '4-7'  :1,
    '8-13' :2,
    '14-20':3,
    '21-32':4,
    '33-43':5,
    '44-59':6,
    '60+'  :7
}

clean_data['age'] = clean_data['age'].apply(lambda age: age_to_label_map[age])
print(len(clean_data))

del data0, data1, data2, data3, data4, all_labeled_data, relevant_data, age_mapping, age_to_label_map, img_path

x = clean_data[['img_path']]
y = clean_data[['gender']]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42, stratify=y)

display_split(y_test, y_train)

train_images = []
test_images = []

for row in x_train.iterrows():
    image = Image.open(row[1].img_path)
    image = image.resize((224, 224))
    data = np.asarray(image)
    train_images.append(data)

for row in x_test.iterrows():
    image = Image.open(row[1].img_path)
    image = image.resize((224, 224))
    data = np.asarray(image)
    test_images.append(data)

train_images = sim_unity_comm(np.asarray(train_images))
test_images = sim_unity_comm(np.asarray(test_images))

with tf.device('/cpu:0'):
   x = tf.convert_to_tensor(train_images, np.float32)
   y = tf.convert_to_tensor(y_train, np.float32)
 
del x_train, x_test, train_images, y_train

from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

model1 = Sequential([
    Conv2D(input_shape=(224, 224, 3), filters=32, strides=2, kernel_size=(5, 5), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2), strides=2), 
    Conv2D(filters=64, kernel_size=(5, 5), strides=1, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2), strides=2), 
    Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2), strides=2), 
    Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(256,activation='relu'),
    Dropout(rate=0.25),
    Dense(256,activation='relu'),
    Dropout(rate=0.25),
    Dense(units=2, activation='softmax')
])


model1.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model1.fit(x, y, batch_size=32, validation_data=(test_images, y_test), epochs=25, verbose=2,
            callbacks = [ReduceLROnPlateau(patience=5, verbose=1),
                            ModelCheckpoint('gender_model1.h5', 
                                            save_best_only=True, 
                                            monitor='val_accuracy', 
                                            mode='max')])


model1 = load_model("gender_model1.h5")
testing(model1, test_images, y_test)
del model1

import gc
from tensorflow.keras.applications import ResNet50
gc.collect()

model2 = Sequential([
    ResNet50(include_top=False, pooling='avg', weights='imagenet'),
    LayerNormalization(),
    Dense(256,activation='relu'),
    Dropout(rate=0.25),
    Dense(256,activation='relu'),
    Dropout(rate=0.25),
    Dense(units=2, activation='softmax')
])

model2.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model2.fit(x, y, batch_size=8, validation_data=(test_images, y_test), epochs=25, verbose=2,
            callbacks = [ReduceLROnPlateau(patience=5, verbose=1),
                            ModelCheckpoint('gender_model2.h5', 
                                            save_best_only=True, 
                                            monitor='val_accuracy', 
                                            mode='max')])

model2 = load_model("gender_model2.h5")
testing(model2, test_images, y_test)

model1 = load_model("gender_model1.h5")

del model1, model2

x = clean_data[['img_path']]
y = clean_data[['age']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=24, stratify=y)

display_split(y_test, y_train)

train_images = []
test_images = []

for row in x_train.iterrows():
    image = Image.open(row[1].img_path)
    image = image.resize((224, 224))
    data = np.asarray(image)
    train_images.append(data)

for row in x_test.iterrows():
    image = Image.open(row[1].img_path)
    image = image.resize((224, 224))
    data = np.asarray(image)
    test_images.append(data)

train_images = sim_unity_comm(np.asarray(train_images))
test_imags = sim_unity_comm(np.asarray(test_images))

with tf.device('/cpu:0'):
   x = tf.convert_to_tensor(train_images, np.float32)
   y = tf.convert_to_tensor(y_train, np.float32)

del x_train, x_test, train_images, y_train

model1 = Sequential([
    Conv2D(input_shape=(224, 224, 3), filters=32, strides=2, kernel_size=(5, 5), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2), strides=2), 
    Conv2D(filters=64, kernel_size=(5, 5), strides=1, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(256,activation='relu'),
    Dropout(rate=0.25),
    Dense(256,activation='relu'),
    Dropout(rate=0.25),
    Dense(units=8, activation='softmax')
])


model1.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model1.fit(x, y, batch_size=32, validation_data=(test_images, y_test), epochs=25, verbose=2,
            callbacks = [ReduceLROnPlateau(patience=5, verbose=1),
                            ModelCheckpoint('age_model1.h5', 
                                            save_best_only=True, 
                                            monitor='val_accuracy', 
                                            mode='max')])

model1 = load_model("age_model1.h5")
testing(model1, test_images, y_test)
plot_confusion_matrix(model1,test_images,y_test)
del model1

model2 = Sequential([
    ResNet50(include_top=False, pooling='avg', weights='imagenet'),
    LayerNormalization(),
    Dense(256,activation='relu'),
    Dropout(rate=0.25),
    Dense(256,activation='relu'),
    Dropout(rate=0.25),
    Dense(units=8, activation='softmax')
])

model2.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model2.fit(x, y, batch_size=8, validation_data=(test_images, y_test), epochs=50, verbose=2,
            callbacks = [ReduceLROnPlateau(patience=5, verbose=1),
                            ModelCheckpoint('age_model2.h5', 
                                            save_best_only=True, 
                                            monitor='val_accuracy', 
                                            mode='max')])

model2 = load_model("age_model2.h5")
testing(model2, test_images, y_test)
plot_confusion_matrix(model2,test_images,y_test)
del model2


def testing_with_tolerance(model, x_test, y_test, tolerance=1):
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = y_test.values.flatten()

    correct_guesses = np.abs(y_pred_classes - y_test_classes) <= tolerance
    wrong_guesses = ~correct_guesses

    correct_counts = Counter(y_test_classes[correct_guesses])
    wrong_counts = Counter(y_test_classes[wrong_guesses])
    categories = np.unique(y_test_classes)
    correct_counts = {cat: correct_counts.get(cat, 0) for cat in categories}
    wrong_counts = {cat: wrong_counts.get(cat, 0) for cat in categories}

    correct = [correct_counts[cat] for cat in categories]
    wrong = [wrong_counts.get(cat, 0) for cat in categories]

    bar_width = 0.35
    index = np.arange(len(categories))
    fig, ax = plt.subplots()
    bar1 = ax.bar(index, correct, bar_width, label='Correct')
    bar2 = ax.bar(index + bar_width, wrong, bar_width, label='Wrong')
    ax.set_xlabel('Category')
    ax.set_ylabel('Counts')
    ax.set_title('Correct and Wrong Guesses per Category (with ±1 tolerance)')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(categories)
    ax.legend()

    for bar in bar1 + bar2:
        height = bar.get_height()
        ax.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.show()

model1 = load_model("age_model1.h5")
testing_with_tolerance(model1, test_images, y_test)

model2 = load_model("age_model2.h5")
testing_with_tolerance(model2, test_images, y_test)
