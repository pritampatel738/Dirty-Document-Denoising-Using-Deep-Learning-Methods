import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import img_to_array,load_img

def preprocessing(data,name):

	arr = []
	for i in data['label']:

		val = "C:/Users/arspiedy/Desktop/DirtyDocumentsDenoising/" + str(name) + "/" +str(i)
		arr.append(val)
	data['label'] = arr
	return data

def load_data(path):

	data = pd.read_csv(path)

	return data

def head(data):
	print(data.head())

def shape(data):
	print("The shape of the data is : ",data.shape)
	pass

def tail(data):
	print(data.tail())

def load_imgg(path):

	ret_arr = []
	for i in path:
		img = load_img(i,target_size=(128,128))
		img = img_to_array(img)
		img = img.astype('float32') / 255.
		ret_arr.append(img)

	ret_arr = np.array(ret_arr)
	return ret_arr

def plot(img):

	plt.imshow(cmap='img,gray')


