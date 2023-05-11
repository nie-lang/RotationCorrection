import tensorflow as tf
import numpy as np
from collections import OrderedDict
import os
import glob
import cv2

rng = np.random.RandomState(2022)


class DataLoader(object):
    def __init__(self, video_folder):
        self.dir = video_folder
        self.videos = OrderedDict()
        self.setup()

    def __call__(self, batch_size):
        video_info_list = list(self.videos.values())
        length = video_info_list[0]['length']

        def video_clip_generator():
            #frame_id = 0
            while True:
                video_clip = []
                #size_clip = []
                frame_id = rng.randint(0, length-1)

                #######inputs
                video_clip.append(np_load_frame(video_info_list[1]['frame'][frame_id], 384, 512))
                video_clip.append(np_load_frame(video_info_list[0]['frame'][frame_id], 384, 512))
                video_clip = np.concatenate(video_clip, axis=2)
                #######size
                #size_clip.append(np_load_size(video_info_list[1]['frame'][frame_id]))
                #size_clip = np.concatenate(size_clip, axis=0)

                yield video_clip


        dataset = tf.data.Dataset.from_generator(generator=video_clip_generator, output_types=tf.float32, output_shapes=[384, 512, 6])
        dataset = dataset.prefetch(buffer_size=128)
        dataset = dataset.shuffle(buffer_size=128).batch(batch_size)
        print('generator dataset, {}'.format(dataset))

        return dataset

    def __getitem__(self, video_name):
        assert video_name in self.videos.keys(), 'video = {} is not in {}!'.format(video_name, self.videos.keys())
        return self.videos[video_name]

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            if video_name == 'input' or video_name == 'gt' :
                self.videos[video_name] = {}
                self.videos[video_name]['path'] = video
                self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
                self.videos[video_name]['frame'].sort()
                self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])

        print(self.videos.keys())

    # for inference on DRC-D dataset
    def get_data_clips(self, index):

        batch = []
        video_info_list = list(self.videos.values())
        
        batch.append(np_load_frame(video_info_list[1]['frame'][index], 384, 512))
        batch.append(np_load_frame(video_info_list[0]['frame'][index], 384, 512))
       
        return np.concatenate(batch, axis=2)



def np_load_frame(filename, resize_height, resize_width):
    image_decoded = cv2.imread(filename)
    
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized




def load(saver, sess, ckpt_path):
    print(ckpt_path)
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')




