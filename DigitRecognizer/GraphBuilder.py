import tensorflow as tf 
import numpy as np 
import TfBlocks

class tf_graph_builder(object):
    def __init__(self, std):
        self.X = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
        self.Y = tf.placeholder(tf.int64, shape=(None))
        self.lr = tf.placeholder(tf.float32)

        self.fliter1 = tf.Variable(tf.truncated_normal((3,3,1,16), stddev=std))
        self.fliter1 = tf.Variable(tf.truncated_normal((3,3,16,32), stddev=std))

        self.Inception1 = TfBlocks.InceptionModule(self.X, 1, 1, std)
        self.Inception2 = TfBlocks.InceptionModule(self.Inception1.output, 120, 2, std)
        self.Residual = TfBlocks.ResidualBlock(self.Inception2.output, 240, std)
        self.avg_pooled = tf.nn.avg_pool(self.Residual.output, [1,7,7,1], strides=[1,7,7,1], padding="SAME")

        self.W = tf.Variable(tf.truncated_normal((240, 10), stddev=std))
        self.b = tf.Variable(tf.constant(0.1, shape=[10]))

        self.logits = tf.matmul(tf.reshape(self.avg_pooled, shape=(-1,240)),self.W) + self.b

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(self.logits,axis=1), self.Y)))

        self.type = tf.argmax(self.logits, axis=1)

        self.train_step = tf.train.AdagradOptimizer(self.lr).minimize(self.loss)