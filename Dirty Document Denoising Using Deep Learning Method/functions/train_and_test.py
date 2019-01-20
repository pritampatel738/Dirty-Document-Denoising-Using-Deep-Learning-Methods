import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from keras.layers import Dense,Flatten,Activation,Input
from keras.models import Sequential,Model

from keras.layers import Conv2D,MaxPooling2D,UpSampling2D


def train(train_cleaned,train_noised,epochs=50,batch_size=32):

	img = Input(shape=(128,128,3))

	encoder = Conv2D(128,(3,3),activation='relu',padding='same')(img)
	encoder1 = MaxPooling2D((2,2))(encoder)

	encoder2 = Conv2D(64,(3,3),activation='relu',padding='same')(encoder1)
	encoder3 = MaxPooling2D((2,2))(encoder2)

	encoder2 = Conv2D(32,(3,3),activation='relu',padding='same')(encoder3)
	encoder3 = MaxPooling2D((2,2))(encoder2)

	decoder = Conv2D(32,(3,3),activation='relu',padding='same')(encoder3)
	decoder1 = UpSampling2D((2,2))(decoder)

	decoder = Conv2D(64,(3,3),activation='relu',padding='same')(decoder1)
	decoder1 = UpSampling2D((2,2))(decoder)

	decoder2 = Conv2D(128,(3,3),activation='relu',padding='same')(decoder1)
	decoder3 = UpSampling2D((2,2))(decoder2)

	decoded = Conv2D(3,(3,3),activation='sigmoid',padding='same')(decoder3)
	autoencoder = Model(img,decoded)

	autoencoder.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	autoencoder.fit(train_noised,train_cleaned,epochs=epochs,batch_size=batch_size)

	return autoencoder

def predict(test,autoencoder):

	pred = autoencoder.predict(test)
	return pred