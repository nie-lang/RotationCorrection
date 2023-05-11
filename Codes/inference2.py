import tensorflow as tf
import os
import time
import numpy as np
import pickle
import cv2 as cv

from model import RotationCorrection2
from utils import load, save, DataLoader
import skimage
import imageio
import glob


import constant
os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = constant.GPU

test_other_folder = constant.TEST_OTHER_FOLDER
batch_size = constant.TEST_BATCH_SIZE
snapshot_dir = constant.SNAPSHOT_DIR + '/pretrained_model/model.ckpt-150000'
#snapshot_dir = constant.SNAPSHOT_DIR + '/model.ckpt-150000'
 

def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(image_name)
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.5)
    return

# define dataset
with tf.name_scope('dataset'):
    test_inputs_clips_tensor = tf.placeholder(shape=[batch_size, None, None, 3], dtype=tf.float32)
    test_input = test_inputs_clips_tensor
    print('test input = {}'.format(test_input))



# define testing RotationCorrection function 
with tf.variable_scope('generator', reuse=None):
    test_final_result = RotationCorrection2(test_input)
    print('testing = {}'.format(tf.get_variable_scope().name))


config = tf.ConfigProto()
config.gpu_options.allow_growth = True      
with tf.Session(config=config) as sess:

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
        
        # prepare data 
        test_list = glob.glob(os.path.join(test_other_folder+"/input/", '*.jpg'))
        length = len(test_list)


        for i in range(0, length):
            
            # load image
            ori_img = cv.imread(test_list[i])
            input_clip = ori_img.astype(dtype=np.float32)
            input_clip = (input_clip / 127.5) - 1.0
            input_clip = np.expand_dims(input_clip, axis=0)
            
            #Attention: both inputs and outpus are the types of numpy
            final_result = sess.run(test_final_result, feed_dict={test_inputs_clips_tensor: input_clip})
            
            final_result = (final_result[0]+1)*127.5
            
            if not os.path.exists(test_other_folder+"/correction/"):
                os.makedirs(test_other_folder+"/correction/")
            path = test_other_folder+"/correction/" + str(i+1).zfill(5) + ".jpg"
            cv.imwrite(path, final_result)
            
            print('i = {} / {}'.format( i+1, length))
            
        print("===================End==================")   
                
    inference_func(snapshot_dir)




