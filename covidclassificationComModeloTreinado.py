from keras.models import load_model

# Carregue o modelo treinado
model = load_model('models/covid_detection_model_backup.keras')


from PIL import Image
import numpy as np

# Carregue a imagem de teste
#test_image = Image.open('COVID-19_Radiography_Dataset/Covid/images/Covid-55.png')
#test_image = Image.open('COVID-19_Radiography_Dataset/Viral Pneumonia/images/Viral Pneumonia-65.png')
test_image = Image.open('COVID-19_Radiography_Dataset/Normal/images/Normal-257.png')

# Pré-processamento da imagem (redimensionamento, conversão para array, normalização, expansão de dimensão)
test_image = test_image.resize((100, 100))
if test_image.mode != "RGB":
    test_image = test_image.convert("RGB")
test_image = np.array(test_image)
test_image = test_image / 255.0  # Normalização
test_image = np.expand_dims(test_image, axis=0)  # Expansão da dimensão

# Faça a previsão com o modelo carregado
result = model.predict(test_image)

print(result)

predicted_class = np.argmax(result, axis=1)[0]
classes = {0: 'Covid', 1: 'Viral Pneumonia', 2: 'Normal'}
prediction = classes[predicted_class]

print(prediction)
