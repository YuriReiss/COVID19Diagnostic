import cv2

# Exemplo de carregamento de uma imagem colorida
image_path = 'COVID-19_Radiography_Dataset/COVID/images/COVID-55.png'
image = cv2.imread(image_path)

# Convertendo para tons de cinza
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Exemplo de exibição da imagem original e da imagem em tons de cinza
cv2.imshow('Imagem Original', image)
cv2.imshow('Imagem em Tons de Cinza', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
