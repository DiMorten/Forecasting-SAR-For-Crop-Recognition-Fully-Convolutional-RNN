import numpy as np
import cv2
from pathlib import Path
folder = 'sample_ims/'
Path(folder).mkdir(parents=True, exist_ok=True)
im_name='prediction_rebuilt_UUnet4ConvLSTM_lem_regression_maskedrmse_balanced_rep1.npy'
im_name='prediction_rebuilt_UUnet4ConvLSTM_lem_jun18_nonorm.npy'
im_name='prediction_rebuilt_UUnet4ConvLSTM_lem_jun18_extendfromoct_nonorm.npy'
def im_to_png(im_name):
    im=np.squeeze(np.load(im_name))
    print(im.min(),np.average(im),im.max())
#    im = ((im+2)*200/5).astype(np.uint8)#(im.max()-im.min())
    im = ((im+0)*250/0.15).astype(np.uint8)#(im.max()-im.min()) # min is 0 max is 0.2

    print(im.min(),np.average(im),im.max())
    print("saving in ",folder+im_name[:-4]+'.png')
    print(im.shape)
    cv2.imwrite(folder+im_name[:-4]+'_VH.png',im[:,:,0])
    cv2.imwrite(folder+im_name[:-4]+'_VV.png',im[:,:,1])

    # masked
    mask_path = '../../../../../dataset/dataset/lm_data/TrainTestMask.tif'
    mask = cv2.imread(mask_path,-1)
    im[mask!=1]=0
    print("saving in ",folder+im_name[:-4]+'_VHm.png')
    cv2.imwrite(folder+im_name[:-4]+'_VHm.png',im[:,:,0])
    cv2.imwrite(folder+im_name[:-4]+'_VVm.png',im[:,:,1])

im_to_png(im_name)

def input_to_png(ims_name,t_step=-1):
    im=np.load(ims_name)[-1]
    print(im.min(),np.average(im),im.max())
#    im = ((im+2)*200/5).astype(np.uint8)#(im.max()-im.min())
    im = ((im+0)*250/0.15).astype(np.uint8)#(im.max()-im.min()) # min is 0 max is 0.2
    
    print(im.min(),np.average(im),im.max())
    # masked
    mask_path = '../../../../../dataset/dataset/lm_data/TrainTestMask.tif'
    mask = cv2.imread(mask_path,-1)
    im[mask!=1]=0
    print("saving in ",folder+'input_im'+'_VH.png')
    print(im.shape)
    cv2.imwrite(folder+'input_im'+'_VHnonorm.png',im[:,:,0])
    cv2.imwrite(folder+'input_im'+'_VVnonorm.png',im[:,:,1])
    
im_name='../../../../../dataset/dataset/lm_data/full_ims/full_ims_train.npy'
input_to_png(im_name)