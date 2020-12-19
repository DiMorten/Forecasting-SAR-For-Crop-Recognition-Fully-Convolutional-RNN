
"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import os
import math
#import json
import random
#import pprint
#import scipy.misc
import numpy as np
from time import gmtime, strftime
#from osgeo import gdal
import glob
#from skimage.transform import resize
#from sklearn import preprocessing as pre
#import matplotlib.pyplot as plt
import cv2
import pathlib
from pathlib import Path
#from sklearn.feature_extraction.image import extract_patches_2d
#from skimage.util import view_as_windows
import sys
import pickle
# Local
import deb
import argparse
from sklearn.preprocessing import StandardScaler
#import natsort
from abc import ABC, abstractmethod
import time, datetime
import pdb
class DataSource(object):
	def __init__(self, band_n, foldernameInput, label_folder,name):
		self.band_n = band_n
		self.foldernameInput = foldernameInput
		self.label_folder = label_folder
		self.name=name
		self.channelsToMask=range(self.band_n)
	
	@abstractmethod
	def im_load(self,filename,conf):
		pass

	def addHumidity(self):
		self.band_n=self.band_n+1

class SARSource(DataSource):

	def __init__(self):
		name='SARSource'
		band_n = 2
		foldernameInput = "in_np2/"
		label_folder = 'labels'
		super().__init__(band_n, foldernameInput, label_folder,name)


	def im_seq_normalize3(self,im,mask):
		im_check_flag=False
		t_steps,h,w,channels=im.shape
		#im=im.copy()
		im_flat=np.transpose(im,(1,2,3,0))
		#im=np.reshape(im,(h,w,t_steps*channels))
		im_flat=np.reshape(im_flat,(h*w,channels*t_steps))
		if im_check_flag==True:
			im_check=np.reshape(im_flat,(h,w,channels,t_steps))
			im_check=np.transpose(im_check,(3,0,1,2))
			deb.prints(im_check.shape)
			deb.prints(np.all(im_check==im))
		deb.prints(im.shape)
		mask_flat=np.reshape(mask,-1)
		#train_flat=im_flat[mask_flat==1,:]

		deb.prints(im_flat[mask_flat==1,:].shape)
		print(np.min(im_flat[mask_flat==1,:]),np.max(im_flat[mask_flat==1,:]),np.average(im_flat[mask_flat==1,:]))

		scaler=StandardScaler()
		scaler.fit(im_flat[mask_flat==1,:])
		#train_norm_flat=scaler.transform(train_flat)
		#del train_flat

		im_norm_flat=scaler.transform(im_flat)
		del im_flat
		im_norm=np.reshape(im_norm_flat,(h,w,channels,t_steps))
		del im_norm_flat
		deb.prints(im_norm.shape)
		im_norm=np.transpose(im_norm,(3,0,1,2))
		deb.prints(im_norm.shape)
		#for t_step in range(t_steps):
		#	print("Normalized time",t_step)
		#	print(np.min(im_norm[t_step]),np.max(im_norm[t_step]),np.average(im_norm[t_step]))
		print("FINISHED NORMALIZING, RESULT:")
		print(np.min(im_norm),np.max(im_norm),np.average(im_norm))
		return im_norm
	def clip_undesired_values(self, full_ims):
		full_ims[full_ims>1]=1
		return full_ims
	def im_load(self,filename,conf):
		return np.load(filename)


class SARHSource(SARSource): #SAR+Humidity
	def __init__(self):
		
		super().__init__()
		#self.name='SARHSource'
		self.band_n = 3
		self.channelsToMask=[0,1]
		#self.channelsToMask=range(self.band_n)
		
	def im_load(self,filename,conf):
		im_out=np.load(filename)
		humidity_filename=conf['path']/('humidity/'+filename[18:26]+'_humidity.npy')
		deb.prints(humidity_filename)
		#pdb.set_trace()
		humidity_im=np.expand_dims(np.load(humidity_filename).astype(np.uint8),axis=-1)
		im_out=np.concatenate((im_out,humidity_im),axis=-1)
		deb.prints(im_out.shape)
		#pdb.set_trace()
		return im_out
class OpticalSource(DataSource):
	
	def __init__(self):
		name='OpticalSource'
		band_n = 3
		#self.t_len = self.dataset.getT_len() implement dataset classes here. then select the dataset/source class
		foldernameInput = "in_optical/"
		label_folder = 'optical_labels'
		# to-do: add input im list names: in_filenames=['01_aesffes.tif', '02_fajief.tif',...]
		super().__init__(band_n, foldernameInput, label_folder,name)

	def im_seq_normalize3(self,im,mask): #to-do: check if this still works for optical
		
		t_steps,h,w,channels=im.shape
		#im=im.copy()
		im_flat=np.transpose(im,(1,2,3,0))
		#im=np.reshape(im,(h,w,t_steps*channels))
		im_flat=np.reshape(im_flat,(h*w,channels*t_steps))
		im_check=np.reshape(im_flat,(h,w,channels,t_steps))
		im_check=np.transpose(im_check,(3,0,1,2))

		deb.prints(im_check.shape)
		deb.prints(np.all(im_check==im))
		deb.prints(im.shape)
		mask_flat=np.reshape(mask,-1)
		train_flat=im_flat[mask_flat==1,:]
		# dont consider cloud areas for scaler fit. First images dont have clouds
		# train_flat=train_flat[self.getCloudMaskedFlatImg(train_flat),:]
		

		deb.prints(train_flat.shape)
		print(np.min(train_flat),np.max(train_flat),np.average(train_flat))

		scaler=StandardScaler()
		scaler.fit(train_flat)
		train_norm_flat=scaler.transform(train_flat) # unused

		im_norm_flat=scaler.transform(im_flat)
		im_norm=np.reshape(im_norm_flat,(h,w,channels,t_steps))
		deb.prints(im_norm.shape)
		im_norm=np.transpose(im_norm,(3,0,1,2))
		deb.prints(im_norm.shape)
		#for t_step in range(t_steps):
		#	print("Normalized time",t_step)
		#	print(np.min(im_norm[t_step]),np.max(im_norm[t_step]),np.average(im_norm[t_step]))
		print("FINISHED NORMALIZING, RESULT:")
		print(np.min(im_norm),np.max(im_norm),np.average(im_norm))
		print("Train masked im:")
		print(np.min(train_norm_flat),np.max(train_norm_flat),np.average(train_norm_flat))
		
		return im_norm
	def getCloudMaskedFlatImg(self, im_flat, threshold=7500):
		# shape is [len, channels]
		cloud_mask=np.zeros_like(im_flat)[:,0]
		deb.prints(np.max(im_flat))
		for chan in range(im_flat.shape[1]):
			deb.prints(np.max(im_flat[:,chan]))
			cloud_mask_chan = np.zeros_like(im_flat[:,chan])
			cloud_mask_chan[im_flat[:,chan]>threshold]=1
			cloud_mask=np.logical_or(cloud_mask,cloud_mask_chan)
		cloud_mask = np.logical_not(cloud_mask)
		deb.prints(np.unique(cloud_mask,return_counts=True))
		return cloud_mask

	def clip_undesired_values(self, full_ims):
		#full_ims[full_ims>8500]=8500
		return full_ims
	def im_load(self,filename):
		return np.load(filename)[:,:,(3,1,0)] #3,1,0 means nir,g,b. originally it was bands 2,3,4,8. So now I pick 8,3,2
class OpticalSourceWithClouds(OpticalSource):
	def __init__(self):
		
		super().__init__()
		self.name='OpticalSourceWithClouds'

class Dataset(object):
	def __init__(self,path,im_h,im_w,class_n,class_list,name):
		self.path=Path(path)
		self.class_n=class_n
		self.im_h=im_h
		self.im_w=im_w
		self.class_list=class_list
		self.name=name
	@abstractmethod
	def addDataSource(self,dataSource):
		pass
	def getBandN(self):
		return self.dataSource.band_n
	def getClassN(self):
		return self.class_n
	def getClassList(self):
		return self.class_list
	def getTimeDelta(self):
		time_delta=[]
		for im in self.im_list:
			date=im[:8]
			print(date)
			time_delta.append(time.mktime(datetime.datetime.strptime(date, 
                                             "%Y%m%d").timetuple()))
		print(time_delta)
		return np.asarray(time_delta)
	def im_load(self,patch,im_names,label_names,add_id,conf):
		fname=sys._getframe().f_code.co_name
		for t_step in range(0,conf["t_len"]):	
			print(t_step,add_id)
			deb.prints(conf["in_npy_path"]/(im_names[t_step]+".npy"))
			#patch["full_ims"][t_step] = np.load(conf["in_npy_path"]+names[t_step]+".npy")[:,:,:2]
			patch["full_ims"][t_step] = self.dataSource.im_load(conf["in_npy_path"]/(im_names[t_step]+".npy"),conf).astype(np.float16)
			#patch["full_ims"][t_step] = np.load(conf["in_npy_path"]+names[t_step]+".npy")
			deb.prints(patch["full_ims"].dtype)
			deb.prints(np.average(patch["full_ims"][t_step]))
			deb.prints(np.max(patch["full_ims"][t_step]))
			deb.prints(np.min(patch["full_ims"][t_step]))
			
			#deb.prints(patch["full_ims"][t_step].dtype)
			patch["full_label_ims"][t_step] = cv2.imread(str(conf["path"]/(self.dataSource.label_folder+"/"+label_names[t_step]+".tif")),0)
			print(conf["path"]/(self.dataSource.label_folder+"/"+label_names[t_step]+".tif"))
			deb.prints(conf["path"]/(self.dataSource.label_folder+"/"+label_names[t_step]+".tif"))
			deb.prints(np.unique(patch["full_label_ims"][t_step],return_counts=True))
			#for band in range(0,conf["band_n"]):
			#	patch["full_ims_train"][t_step,:,:,band][patch["train_mask"]!=1]=-1
			# Do the masking here. Do we have the train labels?
		deb.prints(patch["full_ims"].shape,fname)
		deb.prints(patch["full_label_ims"].shape,fname)
		deb.prints(patch["full_ims"].dtype,fname)
		deb.prints(patch["full_label_ims"].dtype,fname)
		
		deb.prints(np.unique(patch['full_label_ims'],return_counts=True))
		return patch
	def getChannelsToMask(self):
		return self.dataSource.channelsToMask
class CampoVerde(Dataset):
	def __init__(self):
		name='cv'
		path="../cv_data/"
		class_n=13
		im_h=8492
		im_w=7995
		class_list = ['Background','Soybean','Maize','Cotton','Sorghum','Beans','NCC','Pasture','Eucaplyptus','Soil','Turfgrass','Cerrado']
		super().__init__(path,im_h,im_w,class_n,class_list,name)

	def addDataSource(self,dataSource):
		self.dataSource = dataSource
		if self.dataSource.name == 'SARSource':
			self.im_list=['20151029_S1', '20151110_S1', '20151122_S1', '20151204_S1', '20151216_S1', '20160121_S1', '20160214_S1', '20160309_S1', '20160321_S1', '20160508_S1', '20160520_S1', '20160613_S1', '20160707_S1', '20160731_S1']
			self.label_list=self.im_list.copy()
		elif self.dataSource.name == 'OpticalSource':
			self.im_list=[]
			self.label_list=self.im_list.copy()
		self.t_len=len(self.im_list)
class LEM(Dataset):
	def __init__(self):
		name='lm'
		path="../lm_data/"
		class_n=15
		im_w=8658
		im_h=8484
		class_list = ['Background','Soybean','Maize','Cotton','Coffee','Beans','Sorghum','Millet','Eucalyptus','Pasture/Grass','Hay','Cerrado','Conversion Area','Soil','Not Identified']

		super().__init__(path,im_h,im_w,class_n,class_list,name)

	def addDataSource(self,dataSource):
		deb.prints(dataSource.name)
		self.dataSource = dataSource
		if self.dataSource.name == 'SARSource':
			self.im_list=['20170612_S1', '20170706_S1', '20170811_S1', '20170916_S1', '20171010_S1', '20171115_S1', '20171209_S1', '20180114_S1', '20180219_S1', '20180315_S1', '20180420_S1', '20180514_S1', '20180619_S1']
			# less first date
			#self.im_list=['20170706_S1', '20170811_S1', '20170916_S1', '20171010_S1', '20171115_S1', '20171209_S1', '20180114_S1', '20180219_S1', '20180315_S1', '20180420_S1', '20180514_S1', '20180619_S1']
			# less last date
			#self.im_list=['20170612_S1', '20170706_S1', '20170811_S1', '20170916_S1', '20171010_S1', '20171115_S1', '20171209_S1', '20180114_S1', '20180219_S1', '20180315_S1', '20180420_S1', '20180514_S1']

			# extended forecasting jun 18
			self.im_list=['20161214_S1','20170119_S1','20170212_S1','20170308_S1','20170413_S1','20170519_S1',
				'20170612_S1', '20170706_S1', '20170811_S1', '20170916_S1', '20171010_S1', '20171115_S1', 
				'20171209_S1', '20180114_S1', '20180219_S1', '20180315_S1', '20180420_S1', '20180514_S1', 
				'20180619_S1']
			
			self.label_list=self.im_list.copy()
		elif self.dataSource.name == 'OpticalSource':
			#self.im_list=['20170729_S2_10m','20170803_S2_10m','20170907_S2_10m','20171017_S2_10m','20171022_S2_10m','20180420_S2_10m','20180430_S2_10m','20180510_S2_10m','20180614_S2_10m','20180619_S2_10m','20180624_S2_10m']
			self.im_list=['20170604_S2_10m','20170729_S2_10m','20170803_S2_10m','20170907_S2_10m','20171017_S2_10m','20180420_S2_10m','20180510_S2_10m','20180619_S2_10m']
			
			self.label_list=self.im_list.copy()
		elif self.dataSource.name == 'OpticalSourceWithClouds':
			###self.im_list=['20170729_S2_10m','20170803_S2_10m','20170907_S2_10m','20171017_S2_10m','20171022_S2_10m','20180301_S2_10m','20180420_S2_10m','20180430_S2_10m','20180510_S2_10m','20180614_S2_10m','20180619_S2_10m','20180624_S2_10m']
			#self.im_list=['20170729_S2_10m','20170803_S2_10m','20170907_S2_10m','20171017_S2_10m','20171022_S2_10m','20180301_S2_10m','20180420_S2_10m','20180430_S2_10m','20180510_S2_10m','20180614_S2_10m','20180619_S2_10m','20180624_S2_10m']
			
			#self.label_list=['20170729_S2_10m','20170803_S2_10m','20170907_S2_10m','20171017_S2_10m','20171022_S2_10m','20180315_S2_10m','20180420_S2_10m','20180430_S2_10m','20180510_S2_10m','20180614_S2_10m','20180619_S2_10m','20180624_S2_10m']

			#self.im_list=['20170729_S2_10m','20170803_S2_10m','20170907_S2_10m','20171017_S2_10m','20171022_S2_10m','20171111_S2_10m','20171206_S2_10m','20180110_S2_10m','20180301_S2_10m','20180420_S2_10m','20180430_S2_10m','20180510_S2_10m','20180614_S2_10m','20180619_S2_10m','20180624_S2_10m']
			#self.im_list=['20170604_S2_10m','20170729_S2_10m','20170803_S2_10m','20170907_S2_10m','20171017_S2_10m','20171111_S2_10m','20171206_S2_10m','20180110_S2_10m','20180214_S2_10m','20180301_S2_10m','20180420_S2_10m','20180510_S2_10m','20180619_S2_10m']
			self.im_list=['20170604_S2_10m','20170729_S2_10m','20170803_S2_10m','20170907_S2_10m','20171017_S2_10m','20171116_S2_10m','20171206_S2_10m','20180110_S2_10m','20180214_S2_10m','20180301_S2_10m','20180420_S2_10m','20180510_S2_10m','20180619_S2_10m']
			
			self.label_list=self.im_list.copy()
		self.t_len=len(self.im_list)
		
		deb.prints(self.t_len)

class Humidity():
	def __init__(self,dataset):
		self.dataset=dataset
	def loadIms(self):
		out = np.zeros((self.dataset.t_len,self.dataset.im_h,self.dataset.im_w)).astype(np.int8)
		for im_id,t in zip(self.dataset.im_list,range(self.dataset.t_len)):
			filename=self.dataset.path+'humidity/'+im_id[:8]+'_humidity.npy'
			print("humidity filename",filename)
			#pdb.set_trace()
			out[t]=np.load(filename)


		return np.expand_dims(out,axis=-1)





