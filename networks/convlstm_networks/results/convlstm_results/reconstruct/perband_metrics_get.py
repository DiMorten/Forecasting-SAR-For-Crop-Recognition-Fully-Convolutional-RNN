
import numpy as np
import cv2
import glob
import argparse
import pdb
import sys
#sys.path.append('../../../../../train_src/analysis/')
import pathlib
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.models import Model

from PredictionsLoader import PredictionsLoaderNPY, PredictionsLoaderModel,PredictionsLoaderModelForecasting
from utils import seq_add_padding, add_padding
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
parser = argparse.ArgumentParser(description='')
parser.add_argument('-ds', '--dataset', dest='dataset',
					default='cv', help='t len')
parser.add_argument('-mdl', '--model', dest='model_type',
					default='densenet', help='t len')

a = parser.parse_args()

dataset=a.dataset
model_type=a.model_type

direct_execution=True
if direct_execution==True:
	dataset='lm'
	model_type='unet'

data_path='../../../../../dataset/dataset/'
mask_path=data_path+dataset+'_data/TrainTestMask.tif'

# Load mask
mask=cv2.imread(mask_path,-1)
mask[mask==2]=0 # testing as background... maybe later not?
print("Mask shape",mask.shape)

full_path = data_path+dataset+'_data/full_ims/' 
full_ims_test = np.load(full_path+'full_ims_train.npy') # shape (t_len, h, w, channel_n)


#test_x = full_ims_test[:-1] # t len 12. shape (12, h, w, channel_n)
test_y = full_ims_test[-1] # t len 1. shape (h, w, channel_n)

#pred = np.load('prediction_rebuilt.npy') # shape (1,h,w,channel_n)
#pred = np.load('prediction_rebuilt_slidingwindow_bunet4convlstm.npy')
pred = np.load('prediction_rebuilt_stateful_bunet4convlstm.npy')
pred = np.load('prediction_rebuilt_UUnet4ConvLSTM_lem_regression_maskedrmse_balanced_rep1.npy')
pred = np.load('prediction_rebuilt_UUnet4ConvLSTM_lem_regression_maskedrmse_normhwt.npy')
pred = np.load('prediction_rebuilt_UUnet4ConvLSTM_regression_maskedrmse_mar18.npy')

pred = np.load('prediction_rebuilt_UUnet4ConvLSTM_regression_jun18_ext.npy')
pred = np.load('prediction_rebuilt_UUnet4ConvLSTM_lem_jun18_nonorm.npy')

##pred = np.load('prediction_rebuilt_UUnet4ConvLSTM_lem_jun18_extendfromoct_nonorm.npy')
#pred = np.load('prediction_rebuilt_stateful_uunet4convlstm.npy') # shape (1,h,w,channel_n)
#pred = np.load('prediction_rebuilt_stateful_bunet4convlstm.npy') # shape (1,h,w,channel_n)

# RMSE for channel0

#pred=pred[:,:,:]
channel = 1

channel_names = ['VH', 'VV']
def metrics_get(prediction, label,mask): #requires batch['prediction'],batch['label']



    mask=np.expand_dims(mask,axis=-1)
    #mask = np.repeat(mask,2,axis=-1)
#		mask2[:,:,:,:,0]=mask.copy()
#		mask2[:,:,:,:,1]=mask.copy()


    prediction = np.squeeze(prediction)
    print("======================= METRICS GET")
    print("Prediction, label and mask shape",prediction.shape,label.shape,mask.shape)
    prediction = prediction[:,:,channel].flatten()
    label = label[:,:,channel].flatten()
    mask=mask.flatten()
    print("Prediction and label shape after flatten",prediction.shape,label.shape)

    metrics={}
    metrics['rmse_nomask']=mean_squared_error(label,prediction,squared=False)
    metrics['r2_score_nomask']=r2_score(label,prediction)


    # histogram
    #plt.hist(prediction,400,histtype='step',color='blue')
    #plt.hist(label,400,histtype='step',color='green')
    #plt.show()

    print('unique label',np.unique(label,return_counts=True))
    print("Average prediction={} label={}".format(np.average(prediction),np.average(label)))
    print("Std prediction={} label={}".format(np.std(prediction),np.std(label)))

    prediction = prediction[mask!=0]
    label = label[mask!=0]
    print("Prediction and label shape after removal of bcknd",prediction.shape,label.shape)

    metrics['rmse']=mean_squared_error(label,prediction,squared=False)

    print('unique label',np.unique(label,return_counts=True))
    print("Average prediction={} label={}".format(np.average(prediction),np.average(label)))
    print("Std prediction={} label={}".format(np.std(prediction.astype(np.float32)),np.std(label.astype(np.float32))))



    metrics['r2_score']=r2_score(label,prediction)
    # histogram
    hist_min, hist_max = -2, 8
    hist_min, hist_max = -0.1, 0.5
    hist_min, hist_max = 0, 0.06
    hist_min, hist_max = 0, 0.21

    plt.figure()
    matplotlib.rcParams.update({'font.size': 10})

    
    plt.hist(prediction,np.linspace(hist_min, hist_max,400),histtype='step',color='blue',label='prediction')
    plt.hist(label,np.linspace(hist_min, hist_max,400),histtype='step',color='green',label='GT')
    plt.xlabel('SAR intensity bins')
    plt.ylabel('Pixel count')
    plt.legend()
    plt.xlim((0,0.06))
    plt.xlim((0,0.21))

    plt.ylim((0,120000))

    plt.show()

    plt.figure()
    idxs = np.random.randint(0,prediction.shape[0],size=100000)
    plt.plot(label[idxs],prediction[idxs],'.')
    plt.xlabel('SAR intensity')
    plt.ylabel('SAR intensity')
    plt.plot([0,0.25],[0,0.25],'k--')
    plt.xlim((0,0.25))
    plt.ylim((0,0.25))
    plt.show()

    return metrics
metrics = metrics_get(pred,test_y,mask)
print(metrics)


