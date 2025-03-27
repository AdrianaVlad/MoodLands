import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import face_recognition
import cv2
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dropout, Flatten, Dense, Dropout, Conv2D, MaxPool2D, BatchNormalization, LayerNormalization, Resizing, Rescaling, RandomRotation, RandomFlip, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint


def testing(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

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
    train_counts = np.sum(y_train, axis=0)
    test_counts = np.sum(y_test, axis=0)
    labels = np.arange(len(train_counts))
    
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


def plot_confusion_matrix(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_test_classes, y_pred_classes)
    class_names = np.unique(y_test_classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
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
            img_face_resize = cv2.resize(img_face, (96, 96))
        webcam_sim_img.append(img_face_resize)
    test_images_sim = np.asarray(webcam_sim_img)
    return test_images_sim



gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


datagen_train = ImageDataGenerator()
datagen_val = ImageDataGenerator()
train_set = datagen_train.flow_from_directory(
    './images/train',
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=128,
    class_mode='categorical',
    shuffle=True
)

x_train, y_train = [], []
for i in range(len(train_set)):
    x, y = train_set[i]
    x_train.append(x)
    y_train.append(y)

x_train= np.concatenate(x_train)
y_train = np.concatenate(y_train)

test_set = datagen_val.flow_from_directory(
    './images/validation',
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=128,
    class_mode='categorical',
    shuffle=False
)


x_test, y_test = [], []
for i in range(len(test_set)):
    x, y = test_set[i]
    x_test.append(x)
    y_test.append(y)

x_test = np.concatenate(x_test)
y_test = np.concatenate(y_test)

x_train = sim_unity_comm(np.asarray(x_train))
x_test = sim_unity_comm(np.asarray(x_test))

display_split(y_test,y_train)

model1 = Sequential([
    Conv2D(input_shape=(48, 48, 1), filters=32, kernel_size=(3, 3), activation='relu', padding = 'same'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(filters=128, kernel_size=(2, 2), strides=1, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(filters=256, kernel_size=(2, 2), strides=1, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2,2)),
    Flatten(),
    Dense(256,activation='relu'),
    Dropout(rate=0.25),
    Dense(256,activation='relu'),
    Dropout(rate=0.25),
    Dense(units=7, activation='softmax')
])
model1.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model1.fit(x_train,y_train, batch_size=32, validation_data=(x_test,y_test), epochs=50, verbose=2, 
           callbacks = [ReduceLROnPlateau(patience=5, verbose=1),
                            ModelCheckpoint('emotion_model1.h5', 
                                            save_best_only=True, 
                                            monitor='val_accuracy', 
                                            mode='max')])

model1 = load_model("emotion_model1.h5")
testing(model1, x_test,y_test)
plot_confusion_matrix(model1, x_test, y_test)

del model1,x_train,y_train,datagen_train,datagen_val

from tensorflow.keras.applications import ResNet50

model2 = Sequential([
        Input(shape=(48, 48, 1)),
        RandomRotation(factor=0.20, seed=100),
        RandomFlip(mode="horizontal", seed=100),
        Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu'),
        ResNet50(include_top=False, pooling='avg', weights='imagenet'),
        LayerNormalization(),
        Dense(256,activation='relu'),
        Dropout(rate=0.25),
        Dense(256,activation='relu'),
        Dropout(rate=0.25),
        Dense(units=7, activation='softmax')
    ])
model2.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model2.fit(x_train,y_train, batch_size=8, validation_data=(x_test,y_test), epochs=25, verbose=2, 
           callbacks = [ReduceLROnPlateau(patience=5, verbose=1),
                            ModelCheckpoint('emotion_model2.h5', 
                                            save_best_only=True, 
                                            monitor='val_accuracy', 
                                            mode='max')])
model2 = load_model("emotion_model2.h5")
testing(model2,x_test,y_test)
plot_confusion_matrix(model2, x_test, y_test)

del model2, x_test,y_test,test_set,train_set

from keras.utils import to_categorical
import os
from sklearn.model_selection import train_test_split


def testing_with_tolerance(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    tolerance_groups = [
        {0, 2},
        {1, 4},
        {3, 7},
        {5, 6}
    ]
    def within_tolerance(pred, actual):
        for group in tolerance_groups:
            if pred in group and actual in group:
                return True
        return pred == actual
    correct_guesses = np.array([within_tolerance(p, a) for p, a in zip(y_pred_classes, y_test_classes)])
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
    ax.set_title('Correct and Wrong Guesses per Category (with Tolerance)')
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

input_path = "./imagesAffectNet"
emotions = [f.name for f in os.scandir(input_path) if f.is_dir()]

def image_generator(input_path, emotions):
    for index, emotion in enumerate(emotions):
        for filename in os.listdir(os.path.join(input_path, emotion)):
            img = cv2.imread(os.path.join(input_path, emotion, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
            yield img, index

x, y = [], []
for img, label in image_generator(input_path, emotions):
    x.append(img)
    y.append(label)
print(len(x))
x = sim_unity_comm(np.array(x))
y = to_categorical(np.array(y))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123,stratify=y)
display_split(y_test, y_train)

model3 = Sequential([
    Conv2D(input_shape=(96, 96, 3), filters=32, kernel_size=(3, 3), activation='selu', padding = 'same'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='selu'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(filters=128, kernel_size=(2, 2), strides=1, padding='same', activation='selu'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(filters=256, kernel_size=(2, 2), strides=1, padding='same', activation='selu'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2,2)),
    Flatten(),
    Dense(256,activation='relu'),
    Dropout(rate=0.25),
    Dense(256,activation='relu'),
    Dropout(rate=0.25),
    Dense(units=8, activation='softmax')
])
model3.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model3.fit(x_train,y_train, batch_size=32, validation_data=(x_test, y_test), epochs=50, verbose=2, 
           callbacks = [ReduceLROnPlateau(patience=5, verbose=1),
                            ModelCheckpoint('emotion_model3.h5', 
                                            save_best_only=True, 
                                            monitor='val_accuracy', 
                                            mode='max')])
model3 = load_model("emotion_model3.h5")
testing(model3,x_test,y_test)
testing_with_tolerance(model3,x_test,y_test)
plot_confusion_matrix(model3, x_test, y_test)

del model3, x, y

model4 = Sequential([
    ResNet50(include_top=False, pooling='avg', weights='imagenet'),
    LayerNormalization(),
    Dense(256,activation='relu'),
    Dropout(rate=0.25),
    Dense(256,activation='relu'),
    Dropout(rate=0.25),
    Dense(units=8, activation='softmax')
])

model4.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model4.fit(x_train,y_train, batch_size=8, validation_data=(x_test, y_test), epochs=25, verbose=2, 
           callbacks = [ReduceLROnPlateau(patience=5, verbose=1),
                            ModelCheckpoint('emotion_model4.h5', 
                                            save_best_only=True, 
                                            monitor='val_accuracy', 
                                            mode='max')])
model4 = load_model("emotion_model4.h5")
testing(model4,x_test,y_test)
testing_with_tolerance(model4,x_test,y_test)
plot_confusion_matrix(model4, x_test, y_test)
