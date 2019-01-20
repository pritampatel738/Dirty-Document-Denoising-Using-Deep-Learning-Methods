import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions.utilities import *
from functions.train_and_test import *


def main():

	train_cleaned = load_data('C:/Users/arspiedy/Desktop/DirtyDocumentsDenoising/Dirty Document Denoising Using Deep Learning Method/train_cleaned.csv')
	train_noised = load_data('C:/Users/arspiedy/Desktop/DirtyDocumentsDenoising/Dirty Document Denoising Using Deep Learning Method/train_noised.csv')
	test = load_data('C:/Users/arspiedy/Desktop/DirtyDocumentsDenoising/Dirty Document Denoising Using Deep Learning Method/test.csv')

	head(train_cleaned)
	head(train_noised)
	train_cleaned = preprocessing(train_cleaned,"train_cleaned")
	train_noised = preprocessing(train_noised,"train_noised")
	test = preprocessing(test,"test")

	#head(train_cleaned)

	train_cleaned_data = load_imgg(train_cleaned['label'])
	#plot(train_cleaned_data[0])
	train_noised_data = load_imgg(train_noised['label'])
	test_data = load_imgg(test['label'])

	shape(train_cleaned_data)
	shape(train_noised_data)
	shape(test)

	autoencoder = train(train_cleaned_data,train_noised_data,epochs=5)
	print(autoencoder)

	test_predicted = predict(test_data,autoencoder)
	shape(test_predicted)

	pass


if __name__ == "__main__":

	main()
	pass