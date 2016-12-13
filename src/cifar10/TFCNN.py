from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data

import tensorflow as tf
import numpy as np
import time
import sys
np.set_printoptions(precision=4, suppress=True, threshold=1000, linewidth=500)

def zca_whitening(inputs):
    inputs -= np.mean(inputs, axis=0)
    sigma = np.dot(inputs.T, inputs)/inputs.shape[0]
    U,S,V = np.linalg.svd(sigma)
    epsilon = 0.1
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(S + epsilon))), U.T).astype(np.float32)
    print("  ZCAMatrix constructed.")

    i = 0
    while i < inputs.shape[0]:
        next_i = min(inputs.shape[0], i+10000)
        inputs[i:next_i] = np.dot(inputs[i:next_i], ZCAMatrix.T)
        i = next_i
        print("  processed sample id " + str(next_i))

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

def read_cifar():
    class DataSets(object):
        pass
    data_sets = DataSets()

    X = np.load(open("../../data/cifar10/cifar10.whitened_image.npy", "rb"))
    X = X.transpose((1,2,0)) / 256.0
    print(X.shape)

    label = np.load(open("../../data/cifar10/cifar10.label.npy", "rb"))
    y = np.zeros((label.shape[0], 10))
    for i in range(0, label.shape[0]):
        y[i, label[i]] = 1

    # # normalization
    # print('Global contrast normalization...')
    # X = X.transpose((0, 2, 1))
    # X -= np.mean(X, axis=2).reshape((X.shape[0], X.shape[1], 1))
    #
    # # ZCA whitening
    # print('ZCA whitening...')
    # for i in range(3):
    #     X[:, i] = zca_whitening(X[:, i])
    #
    # np.save("../cifar10/cifar10.whitened_image.npy", X.transpose((1, 0, 2)) * 256.0)

    train_size = 50000
    test_size = 10000

    train_images = X[0:train_size].reshape((train_size, 32, 32, 3))
    train_labels = y[0:train_size]
    test_images = X[train_size:train_size+test_size].reshape((test_size, 32, 32, 3))
    test_labels = y[train_size:train_size+test_size]
    all_images = np.copy(X.reshape((train_size+test_size, 32, 32, 3)))
    all_labels = np.copy(y)

    data_sets.train = DataSet(train_images, train_labels)
    data_sets.test = DataSet(test_images, test_labels)
    data_sets.all = DataSet(all_images, all_labels)
    return data_sets, 32, 32, 3

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def pool_2x2(x):
  return tf.nn.avg_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def convnet(input):
    # Convonlutional layer 1
    h_conv1_base = conv2d(input, W_conv1) + b_conv1
    h_conv1 = tf.nn.relu(h_conv1_base)
    h_pool1 = pool_2x2(h_conv1)
    h_norm1 = tf.nn.local_response_normalization(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # Convonlutional layer 2
    h_conv2_base = conv2d(h_norm1, W_conv2) + b_conv2
    h_conv2 = tf.nn.relu(h_conv2_base)
    h_pool2 = pool_2x2(h_conv2)
    h_norm2 = tf.nn.local_response_normalization(h_pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # Convonlutional layer 3
    h_conv3_base = conv2d(h_norm2, W_conv3) + b_conv3
    h_conv3 = tf.nn.relu(h_conv3_base)
    h_pool3 = pool_2x2(h_conv3)
    h_norm3 = tf.nn.local_response_normalization(h_pool3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # fully connected layers
    h_conv3_flat = tf.reshape(h_norm3, [-1, final_height*final_width*num_hidden_3])
    y_conv = tf.nn.softmax(tf.clip_by_value(tf.matmul(h_conv3_flat, W_fc2) + b_fc2, -20, 20))
    return y_conv

def training_output(x):
    x_image = tf.reshape(x, [-1, height, width, n_channel])
    image_list = tf.unpack(x_image, mini_batch_size)

    preprocessed_image_list = []
    for i in range(len(image_list)):
        cropped_image = tf.random_crop(image_list[i], [cropped_width, cropped_width, n_channel])
        preprocessed_image_list.append(cropped_image)
    x_preprocessed = tf.pack(preprocessed_image_list)
    return convnet(x_preprocessed)

def testing_output(x):
    x_image = tf.reshape(x, [-1, height, width, n_channel])
    x_center = tf.slice(x_image, begin=[0, 4, 4, 0], size=[-1, cropped_width, cropped_width, n_channel])
    return convnet(x_center)

mini_batch_size = 50
data, height, width, n_channel = read_cifar()
cropped_width = 24
cropped_height = 24

x = tf.placeholder(tf.float32, shape=[None, height*width*n_channel])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
sess = tf.InteractiveSession()
patch_size = 5

# define model parameters
num_hidden_1 = 32
W_conv1 = weight_variable([patch_size, patch_size, n_channel, num_hidden_1])
b_conv1 = bias_variable([num_hidden_1])

num_hidden_2 = 32
W_conv2 = weight_variable([5, 5, num_hidden_1, num_hidden_2])
b_conv2 = bias_variable([num_hidden_2])

num_hidden_3 = 64
W_conv3 = weight_variable([5, 5, num_hidden_2, num_hidden_3])
b_conv3 = bias_variable([num_hidden_3])

final_height = int(cropped_height / 8)
final_width = int(cropped_width / 8)
W_fc2 = weight_variable([final_height*final_width*num_hidden_3, 10])
b_fc2 = bias_variable([10])

# define model outputs
y_train = training_output(x)
y_test = testing_output(x)

# training
learning_rate = 2e-3
cross_entropy = -tf.reduce_sum(y_*tf.log(y_train))
opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
grads_and_vars = opt.compute_gradients(cross_entropy)
train_step = opt.apply_gradients(grads_and_vars)

train_correct_prediction = tf.equal(tf.argmax(y_train,1), tf.argmax(y_,1))
train_error = 1.0 - tf.reduce_mean(tf.cast(train_correct_prediction, tf.float32))

# testing
test_correct_prediction = tf.equal(tf.argmax(y_test,1), tf.argmax(y_,1))
test_error = 1.0 - tf.reduce_mean(tf.cast(test_correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

print("learning_rate = " + str(learning_rate))
time_elapsed = 0
sys.stdout.flush()

for i in range(200000):
    batch = data.train.next_batch(mini_batch_size)
    if (i+1)%1000 == 0:
        print("%f\t%g" % (time_elapsed, test_error.eval(feed_dict={x: data.test.images, y_: data.test.labels})))

    sys.stdout.flush()
    start = time.time()
    train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1]})
    end = time.time()
    time_elapsed += end - start
