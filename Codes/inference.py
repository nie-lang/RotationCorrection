import tensorflow as tf
import os
import time
import numpy as np
import pickle
import cv2 as cv

from model import RotationCorrection
from utils import load, save, DataLoader
import skimage
import imageio



import constant
os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = constant.GPU

test_folder = constant.TEST_FOLDER
batch_size = constant.TEST_BATCH_SIZE
snapshot_dir = constant.SNAPSHOT_DIR + '/pretrained_model/model.ckpt-150000'
#snapshot_dir = constant.SNAPSHOT_DIR + '/model.ckpt-150000'
 



# define dataset
with tf.name_scope('dataset'):
    test_inputs_clips_tensor = tf.placeholder(shape=[batch_size, None, None, 3 * 2], dtype=tf.float32)
    test_input = test_inputs_clips_tensor[...,0:3]
    test_gt = test_inputs_clips_tensor[...,3:6]
    print('test input = {}'.format(test_input))
    print('test gt = {}'.format(test_gt))



# define testing RotationCorrection function 
with tf.variable_scope('generator', reuse=None):
    test_mesh, test_horizon, test_flow, test_horizon2 = RotationCorrection(test_input)
    print('testing = {}'.format(tf.get_variable_scope().name))


config = tf.ConfigProto()
config.gpu_options.allow_growth = True      
with tf.Session(config=config) as sess:
    # dataset
    input_loader = DataLoader(test_folder)

    # initialize weights
    sess.run(tf.global_variables_initializer())
    print('Init global successfully!')

    # tf saver
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)

    restore_var = [v for v in tf.global_variables()]
    loader = tf.train.Saver(var_list=restore_var)

    def inference_func(ckpt):
        print("============")
        print(ckpt)
        load(loader, sess, ckpt)
        print("============")
        length = len(os.listdir(test_folder+"/input"))
        psnr_list = []
        ssim_list = []
        psnr_list2 = []
        ssim_list2 = []

        for i in range(0, length):
            input_clip = np.expand_dims(input_loader.get_data_clips(i), axis=0)
            
            #Attention: both inputs and outpus are the types of numpy
            mesh, rotation, flow, rotation2 = sess.run([test_mesh, test_horizon, test_flow, test_horizon2], feed_dict={test_inputs_clips_tensor: input_clip})
            
            input_image = (input_clip[0,:,:,0:3]+1)/2*255
            rotation_gt = (input_clip[0,:,:,3:6]+1)/2*255
            
            rotation = (rotation[0]+1)*127.5
            rotation2 = (rotation2[0]+1)*127.5
            
            if not os.path.exists("../result_mesh/"):
                os.makedirs("../result_mesh/")
            path = "../result_mesh/" + str(i+1).zfill(5) + ".jpg"
            cv.imwrite(path, rotation)
            
            if not os.path.exists("../result_meshflow/"):
                os.makedirs("../result_meshflow/")
            path = "../result_meshflow/" + str(i+1).zfill(5) + ".jpg"
            cv.imwrite(path, rotation2)
            
            psnr = skimage.measure.compare_psnr(rotation, rotation_gt, 255)
            ssim = skimage.measure.compare_ssim(rotation, rotation_gt, data_range=255, multichannel=True)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            
            psnr2 = skimage.measure.compare_psnr(rotation2, rotation_gt, 255)
            ssim2 = skimage.measure.compare_ssim(rotation2, rotation_gt, data_range=255, multichannel=True)
            psnr_list2.append(psnr2)
            ssim_list2.append(ssim2)
            
            print('i = {} / {} psnr2 = {}'.format( i+1, length, psnr2))
            
        print("===================Results Analysis==================")   
        print("mesh:")
        print('average psnr:', np.mean(psnr_list))
        print('average ssim:', np.mean(ssim_list))
        print("--------------")
        print("mesh+flow:")
        print('average psnr2:', np.mean(psnr_list2))
        print('average ssim2:', np.mean(ssim_list2))
                
    inference_func(snapshot_dir)





