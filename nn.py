import tensorflow as tf
import numpy as np

import data.read as rd

'''Conv > Activation> Normalization > Dropout > Pooling'''

class NN(object):
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
                keep_prob = 0.4
            else:
                keep_prob = 1.0
            return tf.nn.dropout(input, keep_prob, name='dropout')

    def model(self, data, label, lr, is_training=False):
        '''模型结构'''
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
        drop1 = self.dropout(fc1, is_training, 'dropout1')
        # fc2 = self.fc(drop,1024,512,'fc2')
        fc2 = self.fc(drop1,512,256,'fc2')
        drop2 = self.dropout(fc2, is_training, 'dropout2')
        logits = self.logits(drop2,256,83,'logits')

        with tf.variable_scope('loss'):
            reg_loss = tf.losses.get_regularization_loss()
            xent=tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits,labels=label))
            loss = self.weight_regularizer*reg_loss + xent
            tf.summary.scalar('loss', loss)
        if is_training:
            with tf.variable_scope('trian_step'):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    trian_op = tf.train.AdamOptimizer(lr).minimize(loss)
        
        with tf.variable_scope('acc'):
            corret_pred = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32),label)
            acc = tf.reduce_mean(tf.cast(corret_pred, tf.float32))
            tf.summary.scalar('acc', acc)
        self.summ = tf.summary.merge_all()
        if is_training:
            out = trian_op, xent, acc
        else:
            out = xent, acc
        return out

    def creat_model(self,data,label,lr):
        '''构建模型'''
        with tf.variable_scope('model') as scope:
            self.train_out = self.model(data,label,lr,is_training=True)
            scope.reuse_variables()
            self.test_out = self.model(data,label,lr,is_training=False)

    def train(self, batch, lr):
        '''从头开始训练一个模型'''
        read = rd.Reader()
        read.create_dataset()
        dt, lb, train_iter_op, test_iter_op = read.get_next_data(batch)
        self.creat_model(dt,lb,lr)
        
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            train_wt = tf.summary.FileWriter('log/adam/'+str(lr))
            train_wt.add_graph(sess.graph)
            sess.run([train_iter_op, init])
            max_acc = 0
            for i in range(800):
                print('epoch:{}'.format(i))
                try:
                    sess.run(train_iter_op)
                    n_batch, train_loss, train_acc= 0,0,0
                    while True:
                        _, summary = sess.run([self.train_out, self.summ])
                        train_wt.add_summary(summary)
                except tf.errors.OutOfRangeError:
                    pass
                # test acc
                try:
                    sess.run(test_iter_op)
                    test_loss, test_acc, n_batch = 0, 0, 0
                    while True:
                        loss_, acc_ = sess.run(self.test_out)
                        test_acc += acc_
                        test_loss += loss_
                        n_batch +=1
                except tf.errors.OutOfRangeError:
                    print("test loss:{:.4f}, test acc:{:.4f}".format(test_loss/n_batch, test_acc/n_batch))
                # save model per 5 epoch
                if i % 5 == 0:
                    saver.save(sess,'ckpt4/epoch',i)
                # save best model 
                if test_acc>max_acc:
                    max_acc = test_acc
                    saver.save(sess, 'ckpt4/best_model')
                    print('current best model:{}'.format(i))
            train_wt.close()
    
    def retrain(self, path, batch, lr):
        '''从已有模型训练一个网络'''
        read = rd.Reader()
        read.create_dataset()
        dt, lb, train_iter_op, test_iter_op = read.get_next_data(batch)
        # tf.summary.image('input', dt)
        self.creat_model(dt,lb,lr)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, path)
            train_wt = tf.summary.FileWriter('log/adam/'+str(lr))
            train_wt.add_graph(sess.graph)
            sess.run([train_iter_op])
            max_acc = 0.81
            for i in range(800):
                print('epoch:{}'.format(i))
                try:
                    sess.run(train_iter_op)
                    n_batch, train_loss, train_acc= 0,0,0
                    while True:
                        _, summary = sess.run([self.train_out, self.summ])
                        train_wt.add_summary(summary)
                except tf.errors.OutOfRangeError:
                    pass
                # test acc 
                
                try:
                    sess.run(test_iter_op)
                    test_loss, test_acc, n_batch = 0, 0, 0
                    while True:
                        loss_, acc_ = sess.run(self.test_out)
                        test_acc += acc_
                        test_loss += loss_
                        n_batch +=1
                except tf.errors.OutOfRangeError:
                    print("test loss:{:.4f}, test acc:{:.4f}".format(test_loss/n_batch, test_acc/n_batch))
                # save model per 5 epoch
                if i % 5 == 0:
                    saver.save(sess,'ckpt4/epoch',i)
                # save best model 
                if test_acc>max_acc:
                    max_acc = test_acc
                    saver.save(sess, 'ckpt4/best_model')
                    print('current best model:{}'.format(i))
            train_wt.close()
        
if __name__ == "__main__":
    # epoch = 4
    batch_size = 200
    lr = 0.0001
    model_path = 'ckpt4/best_model'
    cnn = NN()
    # cnn.train(batch_size, lr)
    cnn.retrain(model_path, batch_size, lr)

'''
epoch:4
test loss:0.7250, test acc:0.8242
current best model:4

epoch:6 lr=0.0001
test loss:0.8464, test acc:0.8127
current best model:6

epoch:41 lr=0.001
test loss:1.2060, test acc:0.7622
current best model:41

epoch:7
test loss:1.1525, test acc:0.7553
current best model:7

epoch:34
test loss:1.5436, test acc:0.6967
current best model:34

epoch:245
test loss:1.6543, test acc:0.6244
current best model:245

epoch:186
test loss:2.1485, test acc:0.4621
current best model:186

epoch:189
test loss:2.1001, test acc:0.5604
current best model:189
'''
