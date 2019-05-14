import pickle
import numpy as np
import tensorflow as tf
# import cv2
import matplotlib.pyplot as plt
import data.data_read as dtrd
import os

class Reader(object):
    def __init__(self):
        self.train_dataset = None
        self.test_dataset = None
        self.iter = None
        self.train_sz = 0
        self.test_sz = 0

    def create_dataset(self):
        test_path = os.path.join('E:\\asus\\Python\\my_nn\\data\\split_data','test')
        train_path = os.path.join('E:\\asus\\Python\\my_nn\\data\\split_data','trainX5equal')
        rd = dtrd.Read()
        rd.nm_lb(test_path)
        train_data = rd.get_train(train_path)
        test_data = rd.get_test(test_path)
        self.train_sz = len(train_data[0])
        self.test_sz = len(test_data[0])
        self.train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
        self.test_dataset = tf.data.Dataset.from_tensor_slices(test_data)

    def _load_img(self, path, label):
        img_string = tf.read_file(path)
        img = tf.image.decode_jpeg(img_string, channels=3)
        img = tf.image.rgb_to_grayscale(img)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img, label
    
    def _load_pred_img(self, path):
        img_string = tf.read_file(path)
        img = tf.image.decode_jpeg(img_string, channels=3)
        img = tf.image.rgb_to_grayscale(img)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img
    
    def data_distort(self, img, label):
        with tf.name_scope('data_distort'):
            # img = tf.image.resize_images(img,[128,128])
            # img = tf.image.random_crop(img, [100, 100, 1])
            # img = tf.image.random_flip_left_right(img)
            img = tf.image.random_contrast(img, lower=0.3, upper=1.0)
            img = tf.image.random_brightness(img, max_delta=0.2)
        return img, label

    def _get_train_data(self, batch_sz=10):
        # train_dataset = train_dataset.shuffle(3*batch_sz).batch(batch_sz)
        # train_dataset = self.train_dataset.map(self._load_img)
        # train_dataset = self.train_dataset.prefetch(1)
        train_dataset = self.train_dataset.shuffle(self.train_sz).map(
            self._load_img, num_parallel_calls=4).map(self.data_distort,num_parallel_calls=4)
        train_dataset = train_dataset.batch(batch_sz)
        train_dataset = train_dataset.prefetch(tf.contrib.data.AUTOTUNE)
        return train_dataset

    def _get_test_data(self):
        test_dataset = self.test_dataset.shuffle(self.test_sz).map(self._load_img, num_parallel_calls=4)
        test_dataset = test_dataset.batch(128).prefetch(tf.contrib.data.AUTOTUNE)
        # test_dataset = test_dataset.batch(128).prefetch(1)
        return test_dataset
    
    def get_next_data(self, batch_sz=10):
        with tf.name_scope('dataset'):
            train_dataset = self._get_train_data(batch_sz)
            test_dataset = self._get_test_data()
        with tf.name_scope('iter'):
            iter = tf.data.Iterator.from_structure(
                train_dataset.output_types, test_dataset.output_shapes)
        with tf.name_scope('train_init_op'):
            train_init_op = iter.make_initializer(train_dataset)
        with tf.name_scope('test_init_op'):
            test_init_op = iter.make_initializer(test_dataset)
        data, label = iter.get_next()
        return data, label, train_init_op, test_init_op
     
    def get_pred(self, img_list):
        data = tf.data.Dataset.from_tensor_slices(img_list)
        data = data.map(self._load_pred_img).batch(1)
        iterator = data.make_one_shot_iterator()
        next_ele = iterator.get_next()
        return next_ele

if __name__ == "__main__":
    img_list = ['E:\\asus\\Python\\my_nn\\data\\split_data\\test\\Adam Sandler\\28.jpg']
    read = Reader()
    # nxt = read.get_pred(img_list)
    # with tf.Session() as sess:
    #     for i in range(len(img_list)):
    #         dt = sess.run(nxt)
    #         print(dt)