from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, Conv2DTranspose
# from keras.callbacks import ModelCheckpoint , EarlyStopping
from keras.optimizers import Adam,Adagrad 
from keras.models import Model
from keras import backend as K
import keras

import numpy as np
from sklearn.utils import shuffle
import cv2
#from skimage.util import view_as_windows
import argparse
import tensorflow as tf

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import metrics
import sys
import glob
import pdb

class PredictionsLoader():
	def __init__(self):
		pass


class PredictionsLoaderNPY(PredictionsLoader):
	def __init__(self):
		pass
	def loadPredictions(self,path_predictions, path_labels):
		return np.load(path_predictions, allow_pickle=True), np.load(path_labels, allow_pickle=True)

class PredictionsLoaderModel(PredictionsLoader):
	def __init__(self, path_test):
		self.path_test=path_test
	def loadPredictions(self,path_model):
		print("============== loading model =============")
		model=load_model(path_model, compile=False)
		print("Model", model)
		test_in=np.load(self.path_test+'patches_in.npy',mmap_mode='r')
		test_label=np.load(self.path_test+'patches_label.npy')

		test_predictions = model.predict(test_in)
		print(test_in.shape, test_label.shape, test_predictions.shape)
		print("Test predictions dtype",test_predictions.dtype)
		del test_in
		return test_predictions, test_label, model


class PredictionsLoaderModelForecasting(PredictionsLoaderModel):
	def predictSequence(self,model,x):
		# x: shape (patches_n, t_len, h, w, channel_n)
		batch_size = 16
		patches_n, t_len, h, w, channel_n = x.shape # t_len is 12 
		batch_n = patches_n // batch_size
		prediction = np.zeros((patches_n,1,h,w,channel_n))
		for batch_id in range(batch_n):
		
			idx0 = batch_id*batch_size
			idx1 = (batch_id+1)*batch_size

			batch_x_part1 = x[idx0:idx1,:-1]
			prediction[idx0:idx1,:-1] = model.predict(batch_x_part1)
			
			batch_x_part2 = batch_x_part1.copy()
			batch_x_part2[:,0] = x[idx0:idx1,-1] # only first t_step is used here
			
			prediction[idx0:idx1,0] = model.predict(batch_x_part2)[:,0]
			model.reset_states()
		return prediction
	def predictOnePatch(self,model,x):
		# x: shape (1, t_len, h, w, channel_n) t_len is 12
		batch_size = 16
		x = np.repeat(x, batch_size, axis=0)
		x_part1 = x[:,:-1]
		_ = model.predict(x_part1)
		x_part2 = x_part1.copy()
		x_part2[:,0] = x[:,-1]
		prediction = model.predict(x_part2)[0,0] #first batch element, first t should be t=12 , producing t_y=13
		#print("Prediction shape",prediction.shape)
		model.reset_states()
		#pdb.set_trace()
		return np.expand_dims(prediction,axis=0),model
	def predictOnePatch(self,model,x):
		# x: shape (1, t_len, h, w, channel_n) t_len is 12
		batch_size = 16
		x = np.repeat(x, batch_size, axis=0)
		prediction = model.predict(x)[0,-1]
		model.reset_states()
		#pdb.set_trace()
		return np.expand_dims(prediction,axis=0),model
	def predictOnePatchSlidingWindow(self,model,x):
		# x: shape (1, t_len, h, w, channel_n) t_len is 12
		batch_size = 16
		x = np.repeat(x, batch_size, axis=0)
		x = x[:,1:] # t_len is 11
		prediction = model.predict(x)[0,-1] #first batch element, last t should be t=12 , producing t_y=13
		#print("Prediction shape",prediction.shape)
		model.reset_states()
		#pdb.set_trace()
		return np.expand_dims(prediction,axis=0),model

	def loadPredictions(self,path_model):
		print("============== loading model =============")
		model=load_model(path_model, compile=False)
		print("Model", model)
		test_in=np.load(self.path_test+'patches_in.npy',mmap_mode='r')
		
		test_x = test_in[:,:-1] # t len 12.
		test_y = test_in[:,-1] # t len 1. shape (patches_n, h, w, channel_n)
		test_y = np.expand_dims(test_y,axis=1) # (patches_n, 1, h, w, channel_n)

		test_predictions = self.predictSequence(model,test_x)
		print(test_in.shape, test_y.shape, test_predictions.shape)
		print("Test predictions dtype",test_predictions.dtype)
		del test_in
		return test_predictions, test_y
