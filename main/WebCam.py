import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
from keras.utils import load_img, img_to_array

image_x, image_y = 64, 64

classifier = load_model('./model.h5')

classes = classifier.output_shape[-1]
letras = {'0': '1', '1': '2', '2': '3', '3': '4', '4': '5', '5': '6', '6': '7',
          '7': '9', '8': 'A', '9': 'Abracar', '10': 'Ainda', '11': 'Ajudar', '12': 'Alface',
          '13': 'B', '14': 'Banco', '15': 'C', '16': 'Cachorro', '17': 'Coruja', '18': 'Cuidar',
          '19': 'D', '20': 'E', '21': 'Eu amo voce', '22': 'F', '23': 'G', '24': 'I', '25': 'K',
          '26': 'L', '27': 'Loja', '28': 'M', '29': 'Meu', '30': 'Morar', '31': 'Morcego',
          '32': 'N', '33': 'O', '34': 'P', '35': 'Posto de abastecimento', '36': 'Q', '37': 'Quarta-feira',
          '38': 'Quinta-feira', '39': 'R', '40': 'S', '41': 'Segunda-feira', '42': 'T', '43': 'Terca-feira',
          '44': 'U', '45': 'V', '46': 'Voce', '47': 'W', '48': 'Y'}

letrass = {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'I', '8': 'K',
        '9': 'L', '10': 'M', '11': 'N', '12': 'O', '13': 'P', '14': 'Q', '15': 'R', '16': 'S', '17': 'T',
        '18': 'U', '19': 'V', '20': 'W', '21': 'Y'}


def predictor():
    test_image = load_img('./image/img.png', target_size=(64, 64))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)

    bigger, class_index = -1, -1

    for x in range(classes):
        if result[0][x] > bigger:
            bigger = result[0][x]
            class_index = x

    return [result, letras[str(class_index)]]


webCam = cv2.VideoCapture(0)

img_text = ['', '']


if webCam.isOpened():
    validation, frame = webCam.read()

    while validation:
        validation, frame = webCam.read()

        frame = cv2.flip(frame, 1) # Invertendo horizontalmente a camera

        # Desenha um retangulo para capturar a imagem a 64x64
        img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2, lineType=8, shift=0)

        imcrop = img[102:298, 427:623] # Recortando
        imcrop = cv2.cvtColor(imcrop, cv2.COLOR_BGR2GRAY) # Mudando para escala de cinza 
        imcrop = np.array(imcrop)

        cv2.putText(frame, str(img_text[1]), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
        cv2.imshow("Webcam", frame)

        img_name = "./image/img.png"
        save_img = cv2.resize(imcrop, (image_x, image_y))
        cv2.imwrite(img_name, save_img)
        img_text = predictor()

        print(img_text[0])

        output = np.ones((150, 150, 3)) * 255
        cv2.putText(output, str(img_text[1]), (15, 130), cv2.FONT_HERSHEY_TRIPLEX, 6, (255, 0, 0))
        cv2.imshow("PREDICT", output)

        if cv2.waitKey(1) == 27:
            break
else:
    print("Camera não encontrada")


webCam.release()
cv2.destroyAllWindows()
