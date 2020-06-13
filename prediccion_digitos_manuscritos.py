# Commented out IPython magic to ensure Python compatibility.
# Primero vamos a usar algo que nose que hace pero al parecer lo necesitamos 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
# %matplotlib inline

np.random.seed(30)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping


sns.set(style='white', context='notebook', palette='deep')

# Primero de todo vamos a cargar los datos usando pandas, estos son los sets de datos obtenidos previamente
train = pd.read_csv("./train/train.csv")
test = pd.read_csv("./test/test.csv")

# Ahora definimos donde se encuentran las etiquetas, en nuestro caso es en la primera fila
Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1) 

# etiquetas con los valores correspondientes a la imagen
g = sns.countplot(Y_train)

# numero de valores de cada etiqueta
Y_train.value_counts()

# comprobamos que no hay datos vacios
X_train.isnull().any().describe()
test.isnull().any().describe()

"""Ahora que tenemos los datos vamos a normalizarlos todos:"""

# Normalizamos las imagenes de [0,255] a [0,1] para que no hayan problemas de iluminación
X_train = X_train / 255.0
test = test / 255.0

# Ahora reescalamos las imagenes para que tengan 28px de alto, 28px de ancho
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

# Agrupamos las etiquetas en un único vector que marca que tipo de numero es, siendo tipo [0 1 2 3 4 5 6 7 8 9] (ej.- 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_train = to_categorical(Y_train, num_classes = 10)

# separamos el set de entrenamiento en el set de entrenamiento 90% y un set de validación para posteriormente comprobar el funcionamiento del modelo 10%
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)

# visualizar un ejemplo de imagen ->
g = plt.imshow(X_train[0][:,:,0])

# El modelo que vamos a utilizar es un CNN de la siguiente manera:
# Input -> [[Conv2D -> rectificador(relu)*2] -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Output

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

# Optimizador
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# Compilar el modelo
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics=['accuracy'])


reduccion_de_tasa_de_aprendizaje = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

epochs = 30
batch_size = 86


datagen = ImageDataGenerator(
        featurewise_center=False,  # establece la media de input a 0 sobre el dataset
        samplewise_center=False,  # establece la media de cada muestra a 0
        featurewise_std_normalization=False,  # divide el input por el std del dataset
        samplewise_std_normalization=False,  # divide cada input por su std
        zca_whitening=False,  # ZCA whitening
        rotation_range=10,  # rota de forma aleatoria las imagenes en un rando de (grados, 0 to 180)
        zoom_range = 0.1, # Zoom aleatorio
        width_shift_range=0.1,  # cambia de posición horizontalmente las imagenes de forma aleatoria
        height_shift_range=0.1,  # cambia de posición verticalmente las imagenes de forma aleatoria
        horizontal_flip=False,  # da la vuelta a las imagenes aleatoriamente (horizontalmente)
        vertical_flip=False)  # da la vuelta a las imagenes aleatoriamente (verticalemente)


datagen.fit(X_train)


# Por último lo que vamos a hacer es introducir el modelo

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[reduccion_de_tasa_de_aprendizaje])


fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


# Ahora vamos a guardar nuestro modelo para poder usarlo con ROS
model.save("/tmp/pdm.h5")
