import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suprime warnings do tensorflow
from keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random

while True:
    option = input("Escolha o modelo:\n1 - Minimamente modificado\n2 - Por Exploratório\n3 - Arq. Densenet121\n0 - Sair\n> ")
    # Carregue o modelo treinado
    cinza = False
    if option == '1':
        model = load_model('models/CovidDiagnostic_minimamenteModificado.keras')
    elif option == '2':
        model = load_model('models/CovidDiagnostic_porExploratorio.keras')
        cinza = True
    elif option == '3':
        model = load_model('models/CovidDiagnostic_usandoArquiteturaDensenet.keras')
    else:
        quit()

    model.summary()
    while True:
        option = input("Escolha uma categoria:\n1 - Covid\n2 - Pneumonia\n3 - Normal\n0 - Sair\n> ")
        if option == '1':
            images_path = "COVID-19_Radiography_Dataset/COVID/images/"
            urls = os.listdir(images_path)
            titulo = "Pulmão com Covid"
        elif option == '2':
            images_path = "COVID-19_Radiography_Dataset/Viral Pneumonia/images/"
            urls = os.listdir(images_path)
            titulo = "Pulmão com Pneumonia"
        elif option == '3':
            images_path = "COVID-19_Radiography_Dataset/Normal/images/"
            urls = os.listdir(images_path)
            titulo = "Pulmão saudável"
        else:
            quit()

        # Carregue a imagem de teste
        print("Carregando imagem aleatória")
        if cinza == True:
            test_image = cv2.imread(images_path + random.choice(urls), cv2.IMREAD_GRAYSCALE)
        else:
            test_image = cv2.imread(images_path + random.choice(urls))
        test_image = cv2.resize(test_image, (100, 100))
        test_image = test_image / 255.
        test_image = np.asarray(test_image)
        plt.imshow(test_image)
        plt.title(titulo)
        plt.show()

        test_image = np.array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        result = model.predict(test_image, verbose=0)
        predicted_class = np.argmax(result, axis=1)[0]
        classes = {0: 'Covid-19.\n', 1: 'Viral Pneumonia.\n', 2: 'Saudável.\n'}
        prediction = classes[predicted_class]

        print("Diagnóstico: " + prediction)
        option = input("1 - Testar nova imagem\n2 - Escolher outro modelo\n0 - Sair\n> ")
        if option == '1':
            continue
        elif option == '2':
            break
        else:
            quit()
