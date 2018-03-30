import tensorflow as tf 

def bn_relu(a):
    a_mean, a_var = tf.nn.moments(a, axes=[0,1,2], keep_dims=True)
    return tf.nn.relu((a - a_mean) / (a_var + 0.001))

class InceptionModule(object):
    def __init__(self, input, n_c, x, std):
        self.fliter1_1 = tf.Variable(tf.truncated_normal((1,1,n_c,32*x), stddev=std))
        
        self.fliter2_1 = tf.Variable(tf.truncated_normal((1,1,n_c,16*x), stddev=std))
        self.fliter2_2 = tf.Variable(tf.truncated_normal((3,3,16*x,48*x), stddev=std))

        self.fliter3_1 = tf.Variable(tf.truncated_normal((1,1,n_c,16*x), stddev=std))
        self.fliter3_2 = tf.Variable(tf.truncated_normal((5,5,16*x,24*x), stddev=std))

        self.fliter4_1 = tf.Variable(tf.truncated_normal((1,1,n_c,16*x), stddev=std))

        path1 = bn_relu(tf.nn.conv2d(input, self.fliter1_1, strides=[1,2,2,1], padding="SAME"))

        path2 = bn_relu(tf.nn.conv2d(input, self.fliter2_1, strides=[1,1,1,1], padding="SAME"))
        path2 = bn_relu(tf.nn.conv2d(path2, self.fliter2_2, strides=[1,2,2,1], padding="SAME"))

        path3 = bn_relu(tf.nn.conv2d(input, self.fliter3_1, strides=[1,1,1,1], padding="SAME"))
        path3 = bn_relu(tf.nn.conv2d(path3, self.fliter3_2, strides=[1,2,2,1], padding="SAME"))

        path4 = tf.nn.max_pool(input, [1,3,3,1], strides=[1,1,1,1], padding="SAME")
        path4 = bn_relu(tf.nn.conv2d(path4, self.fliter4_1, strides=[1,2,2,1], padding="SAME"))

        self.output = tf.concat((path1,path2,path3,path4), axis=3)

class ResidualBlock(object):
    def __init__(self, input, n_c, std):
        self.fliter1 = tf.Variable(tf.truncated_normal((3,3,n_c,n_c), stddev=std))
        self.fliter2 = tf.Variable(tf.truncated_normal((3,3,n_c,n_c), stddev=std))

        self.conv1 = bn_relu(tf.nn.conv2d(input, self.fliter1, strides=[1,1,1,1], padding="SAME"))
        self.conv2 = bn_relu(tf.nn.conv2d(input, self.fliter1, strides=[1,1,1,1], padding="SAME"))

        self.output = self.conv2 + input
