import os
import cv2
from keras.utils import to_categorical
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mping
trainTestSplit = 0.2

X_train = np.array([cv2.resize(mping.imread('../input/database1/db/DB/'+img),(224,224)) for img in os.listdir('../input/database1/db/DB/')])
y_train = [img[:2] for img in os.listdir('../input/database1/db/DB/')]
_,y_train = np.unique(y_train, return_inverse=True)
y_train = to_categorical(y_train)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=trainTestSplit, random_state=42)


from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Input
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam

adam = Adam(lr=0.001)

base_model = VGG16(include_top=True, weights='imagenet')
x = Dense(5, activation='softmax', name='predictions')(base_model.layers[-2].output)
model = Model(inputs=base_model.input, outputs=x)

for layer in base_model.layers:
    layer.trainable = False

model.summary()
model.compile(optimizer = adam, loss='categorical_crossentropy', metrics=['accuracy'])

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
model_name = "model1.h5"

model.fit(X_train, y_train, batch_size=16, validation_data=(X_test, y_test), epochs=100, callbacks=[EarlyStopping(monitor='val_loss', patience=9, min_delta=0.1, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', patience=3, min_delta=0.1, factor=0.25, min_lr=0.0001, verbose=1),
            ModelCheckpoint(model_name, save_best_only=True, save_weights_only=True),])



y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(matrix)

score = model.evaluate(X_test, y_test)
print("Confusion Matrix:")
print("Accuracy = ", score[1])

model.save('my_model.h5')
