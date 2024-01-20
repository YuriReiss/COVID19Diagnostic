import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential

# Função para carregar imagens e em tons de cinza
def load_grayscale_images(path, urls, target, image_size=(100, 100)):
    images = []
    labels = []
    for i in range(len(urls)):
        img_path = os.path.join(path, urls[i])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Carrega em tons de cinza
        img = cv2.resize(img, image_size)
        img = img / 255.0
        images.append(img)
        labels.append(target)
    images = np.asarray(images)
    return images, labels

# Carregando imagens para as três classes
covid_path = "COVID-19_Radiography_Dataset/COVID/images"
covid_urls = os.listdir(covid_path)
covid_images, covid_targets = load_grayscale_images(covid_path, covid_urls, 0)

viral_pneumonia_path = "COVID-19_Radiography_Dataset/Viral Pneumonia/images"
viral_pneumonia_urls = os.listdir(viral_pneumonia_path)
viral_pneumonia_images, viral_pneumonia_targets = load_grayscale_images(viral_pneumonia_path, viral_pneumonia_urls, 1)

normal_path = "COVID-19_Radiography_Dataset/Normal/images"
normal_urls = os.listdir(normal_path)
normal_images, normal_targets = load_grayscale_images(normal_path, normal_urls, 2)

# Concatenando dados e rótulos
data = np.concatenate([covid_images, viral_pneumonia_images, normal_images])
targets = np.concatenate([covid_targets, viral_pneumonia_targets, normal_targets])

# Codificando rótulos de forma one-hot
targets = tf.keras.utils.to_categorical(targets, num_classes=3)

# Dividindo em conjuntos de treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.2)

# Definindo o modelo
model = Sequential([
    Conv2D(128, 3, input_shape=(100, 100, 1), activation='relu'),  # Apenas 1 canal (tons de cinza)
    MaxPooling2D(),
    Conv2D(64, 3, activation='relu'),
    MaxPooling2D(),
    Conv2D(32, 3, activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(3, activation='softmax'),
])

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinando o modelo
history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test))

# Plots de precisão e perda
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='test accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='test loss')
plt.legend()
plt.show()

model.save("models/CovidDiagnostic.keras")
