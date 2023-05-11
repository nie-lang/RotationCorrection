import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import conv2d, conv2d_transpose
import tf_spatial_transform_local
import math
import tf_mesh2flow



grid_w = 8
grid_h = 6

#-------------  Warping layer for optical flow -------------------
def get_grid(x):
    batch_size, height, width, filters = tf.unstack(tf.shape(x))
    Bg, Yg, Xg = tf.meshgrid(tf.range(batch_size), tf.range(height), tf.range(width),
                             indexing = 'ij')
    # return indices volume indicate (batch, y, x)
    # return tf.stack([Bg, Yg, Xg], axis = 3)
    return Bg, Yg, Xg # return collectively for elementwise processing

def nearest_warp(x, flow):
    grid_b, grid_y, grid_x = get_grid(x)
    flow = tf.cast(flow, tf.int32)

    warped_gy = tf.add(grid_y, flow[:,:,:,1]) # flow_y
    warped_gx = tf.add(grid_x, flow[:,:,:,0]) # flow_x
    # clip value by height/width limitation
    _, h, w, _ = tf.unstack(tf.shape(x))
    warped_gy = tf.clip_by_value(warped_gy, 0, h-1)
    warped_gx = tf.clip_by_value(warped_gx, 0, w-1)
            
    warped_indices = tf.stack([grid_b, warped_gy, warped_gx], axis = 3)
            
    warped_x = tf.gather_nd(x, warped_indices)
    return warped_x

def bilinear_warp(x, flow):
    _, h, w, _ = tf.unstack(tf.shape(x))
    grid_b, grid_y, grid_x = get_grid(x)
    grid_b = tf.cast(grid_b, tf.float32)
    grid_y = tf.cast(grid_y, tf.float32)
    grid_x = tf.cast(grid_x, tf.float32)

    fx, fy = tf.unstack(flow, axis = -1)
    fx_0 = tf.floor(fx)
    fx_1 = fx_0+1
    fy_0 = tf.floor(fy)
    fy_1 = fy_0+1

    # warping indices
    h_lim = tf.cast(h-1, tf.float32)
    w_lim = tf.cast(w-1, tf.float32)
    gy_0 = tf.clip_by_value(grid_y + fy_0, 0., h_lim)
    gy_1 = tf.clip_by_value(grid_y + fy_1, 0., h_lim)
    gx_0 = tf.clip_by_value(grid_x + fx_0, 0., w_lim)
    gx_1 = tf.clip_by_value(grid_x + fx_1, 0., w_lim)
    
    g_00 = tf.cast(tf.stack([grid_b, gy_0, gx_0], axis = 3), tf.int32)
    g_01 = tf.cast(tf.stack([grid_b, gy_0, gx_1], axis = 3), tf.int32)
    g_10 = tf.cast(tf.stack([grid_b, gy_1, gx_0], axis = 3), tf.int32)
    g_11 = tf.cast(tf.stack([grid_b, gy_1, gx_1], axis = 3), tf.int32)

    # gather contents
    x_00 = tf.gather_nd(x, g_00)
    x_01 = tf.gather_nd(x, g_01)
    x_10 = tf.gather_nd(x, g_10)
    x_11 = tf.gather_nd(x, g_11)

    # coefficients
    c_00 = tf.expand_dims((fy_1 - fy)*(fx_1 - fx), axis = 3)
    c_01 = tf.expand_dims((fy_1 - fy)*(fx - fx_0), axis = 3)
    c_10 = tf.expand_dims((fy - fy_0)*(fx_1 - fx), axis = 3)
    c_11 = tf.expand_dims((fy - fy_0)*(fx - fx_0), axis = 3)

    return c_00*x_00 + c_01*x_01 + c_10*x_10 + c_11*x_11
    
#---------------------------------------------------------------------------------
    
def shift2mesh(mesh_shift, width, height):
    batch_size = tf.shape(mesh_shift)[0]
    h = height/grid_h
    w = width/grid_w
    ori_pt = []
    for i in range(grid_h + 1):
        for j in range(grid_w + 1):
            ww = j * w
            hh = i * h
            p = tf.constant([ww, hh], shape=[2], dtype=tf.float32)
            ori_pt.append(tf.expand_dims(p, 0))
    ori_pt = tf.concat(ori_pt, axis=0)
    ori_pt = tf.reshape(ori_pt, [grid_h+1, grid_w+1, 2])
    ori_pt = tf.tile(tf.expand_dims(ori_pt, 0),[batch_size, 1, 1, 1])

    tar_pt = ori_pt + mesh_shift
    return tar_pt


def flow_resize_operation(flow_input, height, width):
    flow_tmp = tf.image.resize_images(flow_input, [height,width],method=1)
    flow_x = flow_tmp[:, :, :, 0] * tf.cast(width, tf.float32) /512.
    flow_y = flow_tmp[:, :, :, 1] * tf.cast(height, tf.float32) /384.
    flow_output = tf.stack([flow_x, flow_y], 3)
    return flow_output


# rotation correction pipeline for DRC-D dataset
def RotationCorrection(train_input, width=512., height=384.):

    batch_size = tf.shape(train_input)[0]
    
    mesh, rotation_mesh, residual_flow = build_model(train_input, is_reuse = None) 

    
    flow = tf_mesh2flow.mesh2flow(mesh) + residual_flow
    rotation_flow = bilinear_warp(train_input, flow)
    
    return mesh, rotation_mesh, flow, rotation_flow


# rotation correction pipeline for other datasets with arbitrary resolutions
def RotationCorrection2(train_input):
    
    train_input_ori = train_input
    batch_size = tf.shape(train_input_ori)[0]
    height = tf.shape(train_input_ori)[1]
    width = tf.shape(train_input_ori)[2]
    
    train_input = tf.image.resize_images(train_input, [384,512],method=0)
    mesh, rotation_mesh, residual_flow = build_model(train_input, is_reuse = None) 
    # final flow
    flow = tf_mesh2flow.mesh2flow(mesh) + residual_flow
    # scale the flows to original resolutions
    flow_ori = flow_resize_operation(flow, height, width)
    # warp
    final_result = bilinear_warp(train_input_ori, flow_ori)
    
    return final_result


def _maxpool2d(x, kernel_size, stride):
    p = np.floor((kernel_size -1)/2).astype(np.int32)
    p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
    return slim.max_pool2d(p_x, kernel_size, stride=stride)


def feature_extractor(image_tf):
    feature = []
    with tf.variable_scope('conv_block1'): # H
      conv1 = conv2d(inputs=image_tf, num_outputs=64, kernel_size=3, rate=1, activation_fn=tf.nn.relu)
      conv1 = conv2d(inputs=conv1, num_outputs=64, kernel_size=3, rate=1, activation_fn=tf.nn.relu)
      feature.append(conv1)
      maxpool1 = _maxpool2d(conv1, 2, 2) # H/2
    with tf.variable_scope('conv_block2'):
      conv2 = conv2d(inputs=maxpool1, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
      conv2 = conv2d(inputs=conv2, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
      feature.append(conv2)
      maxpool2 = _maxpool2d(conv2, 2, 2) # H/4
    with tf.variable_scope('conv_block3'):
      conv3 = conv2d(inputs=maxpool2, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
      conv3 = conv2d(inputs=conv3, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
      feature.append(conv3)
      maxpool3 = _maxpool2d(conv3, 2, 2) # H/8
    with tf.variable_scope('conv_block4'):
      conv4 = conv2d(inputs=maxpool3, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
      conv4 = conv2d(inputs=conv4, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
      feature.append(conv4)
      maxpool4 = _maxpool2d(conv4, 2, 2) #32*24
    with tf.variable_scope('conv_block5'):
      conv5 = conv2d(inputs=maxpool4, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu)
      conv5 = conv2d(inputs=conv5, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu)
      feature.append(conv5)
    
    return feature



    
def regression_Net(feature):
    
    maxpool1 = _maxpool2d(feature, 2, 2) #16*12
    conv2 = conv2d(inputs=maxpool1, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu)
    conv2 = conv2d(inputs=conv2, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu)
    
    maxpool2 = _maxpool2d(conv2, 2, 2)    #8
    conv3 = conv2d(inputs=maxpool2, num_outputs=512, kernel_size=3, activation_fn=tf.nn.relu)
    conv3 = conv2d(inputs=conv3, num_outputs=512, kernel_size=3, activation_fn=tf.nn.relu)
    
    maxpool3 = _maxpool2d(conv3, 2, 2)    #4
    
    
    fc1 = conv2d(inputs=maxpool3, num_outputs=2048, kernel_size=[3,4], activation_fn=tf.nn.relu, padding="VALID")
    fc2 = conv2d(inputs=fc1, num_outputs=1024, kernel_size=1, activation_fn=tf.nn.relu)
    fc3 = conv2d(inputs=fc2, num_outputs=(grid_w+1)*(grid_h+1)*2, kernel_size=1, activation_fn=None)

    mesh_motion = tf.reshape(fc3, (-1, grid_h+1, grid_w+1, 2))
    
    
    return mesh_motion
    

    
def decoder2(feature):

    h_deconv1 = conv2d_transpose(inputs=feature[-1], num_outputs=128, kernel_size=2, stride=2)
    h_deconv_concat1 = tf.concat([feature[-2], h_deconv1], axis=3)
    conv1 = conv2d(inputs=h_deconv_concat1, num_outputs=128, kernel_size=3)
    conv1 = conv2d(inputs=conv1, num_outputs=128, kernel_size=3)
    
    h_deconv2 = conv2d_transpose(inputs=conv1, num_outputs=128, kernel_size=2, stride=2)
    h_deconv_concat2 = tf.concat([feature[-3], h_deconv2], axis=3)
    conv2 = conv2d(inputs=h_deconv_concat2, num_outputs=128, kernel_size=3)
    conv2 = conv2d(inputs=conv2, num_outputs=128, kernel_size=3)

    h_deconv3 = conv2d_transpose(inputs=conv2, num_outputs=64, kernel_size=2, stride=2)
    h_deconv_concat3 = tf.concat([feature[-4], h_deconv3], axis=3)
    conv3 = conv2d(inputs=h_deconv_concat3, num_outputs=64, kernel_size=3)
    conv3 = conv2d(inputs=conv3, num_outputs=64, kernel_size=3)
    
    h_deconv4 = conv2d_transpose(inputs=conv3, num_outputs=64, kernel_size=2, stride=2)
    h_deconv_concat4 = tf.concat([feature[-5], h_deconv4], axis=3)
    conv4 = conv2d(inputs=h_deconv_concat4, num_outputs=64, kernel_size=3)
    conv4 = conv2d(inputs=conv4, num_outputs=64, kernel_size=3)
    
    
    flow = conv2d(inputs=conv4, num_outputs=2, kernel_size=1, activation_fn=None)
    
    
    return flow


def build_model(train_input, is_reuse):
    with tf.variable_scope('model', reuse = is_reuse):
        batch_size = tf.shape(train_input)[0]
    
    
        with tf.variable_scope('feature_extract', reuse = None): 
            feature = feature_extractor(train_input)
      
        with tf.variable_scope('regression', reuse = None): 
            mesh_motion = regression_Net(feature[-1])
      
        mesh = shift2mesh(mesh_motion, 512, 384)
        rotation_mesh = tf_spatial_transform_local.transformer(train_input, mesh)
        #mesh2flow = tf_mesh2flow.mesh2flow(mesh)
        #rotation_mesh = bilinear_warp(train_input, mesh2flow)
    
    
        with tf.variable_scope('feature_extract2', reuse = None): 
            feature_rotation_mesh = feature_extractor(rotation_mesh)
    
        with tf.variable_scope('decoder2', reuse = None): 
            residual_flow = decoder2(feature_rotation_mesh)
      
    
    return mesh, rotation_mesh, residual_flow
        
