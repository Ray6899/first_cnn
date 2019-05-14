import tensorflow as tf
import numpy as np
import json
import data.read as rd
import os

class inferNN(object):
    def __init__(self):
        self.weight_regularizer = 0.01

    def conv(self, input, size, channel_in, channel_out, name='conv'):
        with tf.variable_scope(name):
            w=tf.get_variable(initializer=tf.truncated_normal([size,size,channel_in,channel_out],stddev=0.1),
                name='W')
            b=tf.get_variable(initializer=tf.constant(0.1, shape=[channel_out]),name='b')
            conv = tf.nn.conv2d(input,w,strides=[1,1,1,1],padding="SAME")
            act = tf.nn.relu(conv+b)
            return act
    
    def max_pool(self,input,scope='pool'):
        with tf.variable_scope(scope):
            pool = tf.nn.max_pool(input,[1,2,2,1],[1,2,2,1],padding="SAME")
        return pool
    
    def fc(self, input, channel_in, channel_out, name='fc'):
        with tf.variable_scope(name):
            w=tf.get_variable(initializer=tf.truncated_normal([channel_in, channel_out], stddev=0.1),
                name='W')
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(w))
            b=tf.get_variable(initializer=tf.constant(0.1,shape=[channel_out]),name='b')
            return tf.nn.relu(tf.matmul(input,w)+b)

    def batch_norm(self, x, is_training, scope, epsilon=0.001, decay=0.99):
        '''
        Performs a batch normalization layer

        Args:
            x: input tensor
            scope: scope name
            is_training: python boolean value
            epsilon: the variance epsilon - a small float number to avoid dividing by 0
            decay: the moving average decay

        Returns:
            The ops of a batch normalization layer
        '''
        with tf.variable_scope(scope):
            shape = x.get_shape().as_list()
            # gamma: a trainable scale factor
            gamma = tf.get_variable("gamma", shape[-1], initializer=tf.constant_initializer(1.0), trainable=True)
            # beta: a trainable shift value
            beta = tf.get_variable("beta", shape[-1], initializer=tf.constant_initializer(0.0), trainable=True)
            moving_avg = tf.get_variable("moving_avg", shape[-1], initializer=tf.constant_initializer(0.0), trainable=False)
            moving_var = tf.get_variable("moving_var", shape[-1], initializer=tf.constant_initializer(1.0), trainable=False)
            if is_training:
                # tf.nn.moments == Calculate the mean and the variance of the tensor x
                avg, var = tf.nn.moments(x, np.arange(len(shape)-1), keep_dims=True)
                avg=tf.reshape(avg, [avg.shape.as_list()[-1]])
                var=tf.reshape(var, [var.shape.as_list()[-1]])
                #update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, decay)
                update_moving_avg=tf.assign(moving_avg, moving_avg*decay+avg*(1-decay))
                #update_moving_var = moving_averages.assign_moving_average(moving_var, var, decay)
                update_moving_var=tf.assign(moving_var, moving_var*decay+var*(1-decay))
                control_inputs = [update_moving_avg, update_moving_var]
            else:
                avg = moving_avg
                var = moving_var
                control_inputs = []
            with tf.control_dependencies(control_inputs):
                output = tf.nn.batch_normalization(x, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon)
        return output

    def logits(self, input, channel_in, channel_out, name='logit'):
        with tf.variable_scope(name):
            w=tf.get_variable(initializer=tf.truncated_normal([channel_in, channel_out], stddev=0.1),name='W')
            b=tf.get_variable(initializer=tf.constant(0.1,shape=[channel_out]),name='b')
            return tf.matmul(input,w)+b

    def dropout(self, input, is_training, name='dropout'):
        with tf.variable_scope(name):
            if is_training:
                keep_prob = 0.5
            else:
                keep_prob = 1.0
            return tf.nn.dropout(input, keep_prob, name='dropout')

    def model(self, data, is_training):
        conv1 = self.conv(data,3,1,32,'conv1')
        bn1 = self.batch_norm(conv1, is_training,'bn1')
        pool1 = self.max_pool(bn1,'pool1')

        conv2 = self.conv(pool1,3,32,64,'conv2')
        bn2 = self.batch_norm(conv2, is_training, 'bn2')
        pool2 = self.max_pool(bn2,'pool2')

        conv3 = self.conv(pool2,3,64,128,'conv3')
        bn3 = self.batch_norm(conv3, is_training, 'bn3')
        pool3 = self.max_pool(bn3,'pool3')

        conv4 = self.conv(pool3,3,128,128,'conv4')
        bn4 = self.batch_norm(conv4, is_training, 'bn4')
        pool4 = self.max_pool(bn4,'pool4')

        # conv5 = self.conv(pool4,3,128,128,'conv5')
        # bn5 = self.batch_norm(conv5, is_training, 'bn5')
        # pool5 = self.max_pool(bn5,'pool5')
        
        flattened = tf.reshape(pool4, [-1,7*7*128])

        # fc1 = self.fc(flattened, 4*4*128, 1024, 'fc1')
        fc1 = self.fc(flattened, 7*7*128, 512, 'fc1')
        drop = self.dropout(fc1, is_training, 'dropout1')
        # fc2 = self.fc(drop,1024,512,'fc2')
        fc2 = self.fc(drop,512,256,'fc2')
        logits = self.logits(fc2,256,83,'logits')
        out = tf.cast(tf.argmax(logits,1),tf.int32)
        return out

    def creat_model(self,data):
        with tf.variable_scope('model') as scope:
            # self.train_out = self.model(data,label,lr,is_training=True)
            # scope.reuse_variables()
            self.pred_out = self.model(data,is_training=False)
    
    def load_map(self):
        path = os.path.join('E:\\asus\\Python\\my_nn\\data\\split_data','nm_map.json')
        with open(path) as fp:
            self.map = json.load(fp)
    
    def infer(self, img_list, path, batch=1):
        read = rd.Reader()
        dt = read.get_pred(img_list)
        self.load_map()
        self.creat_model(dt)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, path)
            for i in range(len(img_list)):
                pred = sess.run(self.pred_out)
                print('{}:{}'.format(img_list[i], self.map[str(pred[0])]))
        
if __name__ == "__main__":
    img_list = ['E:\\asus\\Python\\my_nn\\data\\split_data\\test\\Will Smith\\45.jpg']
    model_path = 'ckpt3/best_model'
    cnn = inferNN()
    cnn.infer(img_list, model_path)
