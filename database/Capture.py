import cv2
import numpy as np
import os

# Tamanho da imagem
image_x, image_y = 64, 64

# Chaves
ESC = 27 
CAPTURE = 32

pathTraining = './database/training'

QTD_TRAIN = 2000

def createFolder(folderName):
    if not os.path.exists(pathTraining + folderName):
        os.mkdir(pathTraining + folderName)
    
               
def capture_images(letter, name):
    createFolder(str(letter))
    
    cam = cv2.VideoCapture(0)

    imgCounter = 0
    t_counter = 1
    training_set_image_name = 1
    folder = ''
    
    while True:
        validation, frame = cam.read()
        frame = cv2.flip(frame, 1)

        img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2, lineType=8, shift=0)

        result = img[102:298, 427:623]              

        cv2.putText(frame, folder +": "+str(imgCounter), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
        cv2.imshow("frame", frame)
        cv2.imshow("result", result)
    

        if cv2.waitKey(1) == ESC:

            if t_counter <= QTD_TRAIN:
                img_name = pathTraining + str(letter) + "/"+name+"{}.png".format(training_set_image_name)
                save_img = cv2.resize(result, (image_x, image_y))
                cv2.imwrite(img_name, save_img)
                print(f"{img_name} written!")
                training_set_image_name += 1
                imgCounter = training_set_image_name
                folder = "TRAIN"  
                                   
            t_counter += 1

            
            if t_counter > QTD_TRAIN:
                print('[INFO] FIM')
                break
      
    cam.release()
    cv2.destroyAllWindows()
    
letter = input("LETRA: ")
name = input("NOME: ")
capture_images(letter, name)