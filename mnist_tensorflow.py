# Import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("dataset/", one_hot=True)

# Import tensorflow & matplotlib
import tensorflow as tf
import matplotlib.pyplot as plt

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Train
for step in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(128)
    _, loss_val = sess.run([train_step, cross_entropy],
                           feed_dict={x: batch_xs, y_: batch_ys})
    print("[%s]" % step, "\t(Loss) = %s" % loss_val)
    plt.plot(step, loss_val, 'ro')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.xlim(-100, 1100)
    plt.ylim(0.0, 3.0)

plt.show()

# For TensorBoard
# tf.summary.FileWriter('./my_graph', sess.graph)

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Final accuracy is %s' % sess.run(accuracy,
                                        feed_dict={
                                            x: mnist.test.images,
                                            y_: mnist.test.labels
                                        }))

sess.close()
