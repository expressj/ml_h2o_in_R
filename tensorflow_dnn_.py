import numpy as np
import tensorflow as tf
from tensorflow import layers

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
num_out = 10

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.int32)

h1 = layers.dense(x, 200, activation=tf.nn.relu)
h2 = layers.dense(x, 200, activation=tf.nn.relu)
out = layers.dense(x, num_out, activation=None)

loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y, logits=out))
optim = tf.train.AdamOptimizer(learning_rate=0.005)
train_op = optim.minimize(loss)

correct_prediction = tf.equal(tf.argmax(out,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
init.run()

num_epoch = 1000
batch_size = 100
num_iter_in_epoch = mnist.train.num_examples // batch_size

for i in range(num_epoch):
    train_accuracy = 0
    for _ in range(num_iter_in_epoch):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys})
        train_accuracy += sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
    print("Epoch : {} | Train Accuracy : {0:.3f}".format(i, train_accuracy))

print("\nFinally, Test Accuracy : {}".format(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})))