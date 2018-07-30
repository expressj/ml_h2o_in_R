import numpy as np
import tensorflow as tf
from tensorflow import layers
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=False)
feature_columns = [tf.feature_column.numeric_column("x", shape=[28, 28])]

classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[200, 200], optimizer=tf.train.AdamOptimizer(1e-4), n_classes=10, dropout=0.5, model_dir="./tmp/mnist_model")

def input(dataset):
     return dataset.images, dataset.labels.astype(np.int32)

train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": input(mnist.train)[0]}, y=input(mnist.train)[1], num_epochs=None, batch_size=100, shuffle=True)
test_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": input(mnist.test)[0]}, y=input(mnist.test)[1], num_epochs=1, shuffle=False)

classifier.train(input_fn=train_input_fn, steps=10000)

accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}%\n".format(accuracy_score*100))