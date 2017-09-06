
# coding: utf-8

# In[42]:

#asdasdljajksdaksd
import tensorflow as tf

#magic, make graph
x_train = [1,2,3]

y_train = [1,2,3]

w = tf.Variable(tf.random_normal([1]), name="weight")

b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = w * x_train + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(20000):
    sess.run(cost)
    if i % 20 == 0:
        print(i, sess.run(cost),sess.run(w), sess.run(b))


# In[ ]:




