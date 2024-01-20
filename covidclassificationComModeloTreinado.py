from keras.models import load_model
from PIL import Image
import numpy as np

# Carregue o modelo treinado
model = load_model('models/CovidDiagnostic_backup.keras')

# Carregue a imagem de teste
test_image = Image.open('COVID-19_Radiography_Dataset/Covid/images/Covid-55.png')
#test_image = Image.open('COVID-19_Radiography_Dataset/Viral Pneumonia/images/Viral Pneumonia-65.png')
#test_image = Image.open('COVID-19_Radiography_Dataset/Normal/images/Normal-257.png')

# Redimensionamento
test_image = test_image.resize((100, 100))

# Converte a imagem para escala de cinza (se não estiver em escala de cinza)
if test_image.mode != "L":
    test_image = test_image.convert("L")

# Converte a imagem para um array numpy 3D (adicionando uma dimensão para canais de cor)
test_image = np.array(test_image)
test_image = np.expand_dims(test_image, axis=-1)

# Normalização
test_image = test_image / 255.0  

# Adiciona uma dimensão para simular o batch (1 imagem no batch)
test_image = np.expand_dims(test_image, axis=0)  

# Faça a previsão com o modelo carregado
result = model.predict(test_image)

print("\nDiagnóstico:")

predicted_class = np.argmax(result, axis=1)[0]
classes = {0: 'Covid-19.\n', 1: 'Viral Pneumonia.\n', 2: '\n\nO paciente não está com Covid ou Pneumonia.\n'}
prediction = classes[predicted_class]

print(prediction)