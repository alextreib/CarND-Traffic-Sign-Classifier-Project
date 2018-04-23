# traffic_sign input
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Input data formatting
traffic_signs = input_data.read_data_sets("../datasets/traffic-signs-data/", reshape=False)
X_test, y_test             = traffic_signs.test.images, traffic_signs.test.labels
print("Test Set:       {} samples".format(len(X_test)))
X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))


# Add ops to save and restore only `v2` using the name "v2"

sess=tf.Session()    
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('./lenet.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))



# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
    
# Restore variables from disk
graph = tf.get_default_graph()
cross_entropy = graph.get_tensor_by_name("l_rate:0")

print("Model restored.")
# Check the values of the variables
feed_dict = {x: X_test[0]}
classification = sess.run(cross_entropy, feed_dict)
print (classification)
