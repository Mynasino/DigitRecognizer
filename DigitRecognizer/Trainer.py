import tensorflow as tf 
import numpy as np 
import DataLoader

class tf_trainer(object):
    def __init__(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.DataLoad = DataLoader.DataLoad()

    def train(self, graph, n_epoch, learning_rate, batch_size, n_train):
        n_batch = int(n_train / batch_size)
        if n_train % batch_size:
            n_batch = n_batch + 1

        n_test_batch = int((42000 - n_train) / batch_size)
        if (42000 - n_train) % batch_size:
            n_test_batch = n_test_batch + 1

        for i_epoch in range(n_epoch):
            train_loss = train_acc = 0
            for train_X, train_Y in self.DataLoad.data_iter(batch_size, n_train):
                feed_dict = {
                    graph.X:train_X,
                    graph.Y:train_Y,
                    graph.lr:learning_rate,
                }
                self.sess.run(graph.train_step, feed_dict=feed_dict)
                train_loss = train_loss + self.sess.run(graph.loss, feed_dict=feed_dict)
                train_acc = train_acc + self.sess.run(graph.accuracy, feed_dict=feed_dict)
            train_loss = train_loss / n_batch
            train_acc = train_acc / n_batch

            test_loss = test_acc = 0
            test_inds = np.arange(n_train, 42000)
            for i in range(n_test_batch):
                feed_dict = {
                    graph.X:self.DataLoad.all_X[test_inds[i*batch_size:min((i+1)*batch_size, 42000)]],
                    graph.Y:self.DataLoad.all_Y[test_inds[i*batch_size:min((i+1)*batch_size, 42000)]],
                }
                test_loss = test_loss + self.sess.run(graph.loss, feed_dict=feed_dict)
                test_acc = test_acc + self.sess.run(graph.accuracy, feed_dict=feed_dict)
            test_loss = test_loss / n_test_batch
            test_acc = test_acc / n_test_batch

            print("epoch %d train loss %f acc %f test loss %f acc %f" % (i_epoch, train_loss, train_acc, test_loss, test_acc))

    def output(self, graph, feed_X):
        feed_dict = {
            graph.X:feed_X,
        }
        return self.sess.run(graph.type, feed_dict=feed_dict)