from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical, plot_model
from keras.optimizers import SGD, Adam
from keras import backend
from PIL import Image
import tensorflow as tf
# import tensorflow_datasets as tfds
# from sklearn.metrics import classification_report
import numpy as np
import datetime
import h5py
import time
import os


def getDateStr():  # Retorna o dia,mes,ano,hora,minuto
    return str('{date:%d_%m_%Y_%H_%M}').format(date=datetime.datetime.now())


def getTimeMin(start, end):
    return (end - start)/60


EPOCHS = 20  # Quantidade de vezes que o codigo irá repetir o treino
CLASS = 32  # Quantidade de Letras
FILE_NAME = 'Model_Libras_'

print("\n\n ----------------------INICIO --------------------------\n")
print('[INFO] [INICIO]: ' + getDateStr())

batch_size = 32
img_height = 64
img_width = 64

train_dataset = tf.keras.utils.image_dataset_from_directory(
    './database/training_noise',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=123)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    './database/training_noise',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=123)

model = tf.keras.Sequential([
    # Normalização dos valores dos pixels entre 0 e 1
    tf.keras.layers.Lambda(lambda x: x/255.0),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(CLASS, activation='softmax')
])


print("[INFO] Inicializando e otimizando a CNN...")
start = time.time()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

classifier = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=test_dataset,
    shuffle=True,
    verbose=2
)

print("[INFO] Salvando modelo treinado ...")

# Para todos arquivos ficarem com a mesma data e hora. Armazeno na variavel
file_date = getDateStr()
model.save('./prototypes/'+FILE_NAME+file_date+'.h5')
print('[INFO] modelo: ./prototypes/'+FILE_NAME+file_date+'.h5 salvo!')

end = time.time()

print("[INFO] Tempo de execução da CNN: %.1f min" % (getTimeMin(start, end)))

print('[INFO] Summary: ')
model.summary()

print("\n[INFO] Avaliando a CNN...")
score = model.evaluate(test_dataset, verbose=1)
print('[INFO] Accuracy: %.2f%%' % (score[1]*100), '| Loss: %.5f' % (score[0]))

print('\n[INFO] [FIM]: ' + getDateStr())
