#####################################################
###  Generate a random sample to estimate
#####################################################
import numpy as np

num_points = 1000   # 1,000 sample points
vectors_set = []    # placeholder

for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)    # mean: 0, std dev: 0.55
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03) # 0.1 is slope, 0.3 is intercept
    vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

import matplotlib.pyplot as plt
plt.plot(x_data, y_data, 'ro')
plt.show()

#####################################################
### Estimate a regression line using tensorflow
#####################################################
import tensorflow as tf

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # start with random integer between -1 and 1
b = tf.Variable(tf.zeros([1])) # start with 0
y = W * x_data + b # linear regression equation

loss = tf.reduce_mean(tf.square(y - y_data)) # define loss function(error) as MSE
optimizer = tf.train.GradientDescentOptimizer(0.5) # optimizer is Gradient Descent with 0.5 learning rate
train = optimizer.minimize(loss) # learning by minimizing the optimizer we defined

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for step in range(10):
    sess.run(train) # Training
    print("[%s]" % step,"]\t(Slope)", sess.run(W), "\t(Intercept)", sess.run(b), "\t(Loss)", sess.run(loss))

    plt.plot(x_data, y_data, 'ro')
    plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
    plt.xlabel('x')
    plt.xlim(-2, 2)
    plt.ylim(0.1, 0.6)
    plt.ylabel('y')
    plt.show()
