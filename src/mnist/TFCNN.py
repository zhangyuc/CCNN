from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data

import tensorflow as tf
import numpy as np
import time, sys
np.set_printoptions(precision=4, suppress=True, threshold=1000, linewidth=500)

def zca_whitening(inputs):
    inputs -= np.mean(inputs, axis=0)
    sigma = np.dot(inputs.T, inputs)/inputs.shape[0]
    U,S,V = np.linalg.svd(sigma)
    epsilon = 0.1
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(S + epsilon))), U.T).astype(np.float32)
    i = 0
    while i < inputs.shape[0]:
        next_i = min(inputs.shape[0], i+10000)
        inputs[i:next_i] = np.dot(inputs[i:next_i], ZCAMatrix.T)
        i = next_i

    return inputs


class DataSet(object):
    def __init__(self, images, labels):
        self._num_examples = images.shape[0]
        images = images.reshape(images.shape[0], images[0].size)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

def read_mnist(name):
    class DataSets(object):
        pass
    data_sets = DataSets()

    X = np.load(open(name + ".image.npy", "rb"))


    label = np.load(open(name + ".label.npy", "rb"))
    y = np.zeros((label.shape[0],10))
    for i in range(0, label.shape[0]):
        y[i, label[i]] = 1

    # normalization
    X = X.reshape((X.shape[0], 784))
    print('Global contrast normalization...')
    X -= np.mean(X, axis=1).reshape((X.shape[0], 1))

    # ZCA whitening
    print('ZCA whitening...')
    X = zca_whitening(X)
    X = X.reshape((X.shape[0], 28, 28, 1))

    train_images = X[0:10000]
    train_labels = y[0:10000]
    validation_images = X[10000:12000]
    validation_labels = y[10000:12000]
    test_images = X[12000:62000]
    test_labels = y[12000:62000]

    data_sets.train = DataSet(train_images, train_labels)
    data_sets.validation = DataSet(validation_images, validation_labels)
    data_sets.test = DataSet(test_images, test_labels)
    return data_sets, 28, 28, 1

def read_data(name):
    return read_mnist(name)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def pool_2x2(x):
  return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID')

name = "../../data/mnist/mnist_bgrot"
print(name)
data, height, width, n_channel = read_data(name)
x = tf.placeholder(tf.float32, shape=[None, height*width*n_channel])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
x_image = tf.reshape(x, [-1, height, width, n_channel])
sess = tf.InteractiveSession()
patch_size = 5

W_conv1 = weight_variable([patch_size, patch_size, n_channel, 16])
b_conv1 = bias_variable([16])

h_conv1_base = conv2d(x_image, W_conv1) + b_conv1
h_conv1 = tf.nn.relu(h_conv1_base)
h_pool1 = pool_2x2(h_conv1)

################## CNN-2 ####################

W_conv2 = weight_variable([5, 5, 16, 32])
b_conv2 = bias_variable([32])

h_conv2_base = conv2d(h_pool1, W_conv2) + b_conv2
h_conv2 = tf.nn.relu(h_conv2_base)
h_pool2 = pool_2x2(h_conv2)

final_height = int(((height - (patch_size-1)) / 2 - (patch_size-1))/2)
final_width = int(((width - (patch_size-1)) / 2 - (patch_size-1))/2)

h_conv2_flat = tf.reshape(h_pool2, [-1, final_height*final_width*32])

W_fc2 = weight_variable([final_height*final_width * 32, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.clip_by_value(tf.matmul(h_conv2_flat, W_fc2) + b_fc2, -20, 20))

learning_rate = 2e-3
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
grads_and_vars = opt.compute_gradients(cross_entropy)
capped_grads_and_vars = [(tf.clip_by_norm(gv[0], 100), gv[1]) for gv in grads_and_vars]
train_step = opt.apply_gradients(capped_grads_and_vars)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
error = 1.0 - tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

print("learning_rate = " + str(learning_rate))
sys.stdout.flush()
time_elapsed = 0
for i in range(10000):
    batch = data.train.next_batch(50)
    if i > 0 and (i+1) % 1000 == 0:
        print(str(time_elapsed) + "\t" + str(error.eval(feed_dict={x: data.test.images, y_: data.test.labels})))
        sys.stdout.flush()

    start = time.time()
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    end = time.time()
    time_elapsed += end - start
