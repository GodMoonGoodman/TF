
# coding: utf-8

# In[1]:


import tensorflow as tf


# 변수선언

# In[ ]:


W = tf.Variable(tf.random_normal([1]), name="weight")

b = tf.Variable(tf.random_normal([1]), name="bias")

X = tf.placeholder(tf.float32, shape=[None])

Y = tf.placeholder(tf.float32, shape=[None])


# In[19]:


hypothesis = W*X+ b


# In[23]:


cost = tf.reduce_mean(tf.square(hypothesis-Y))


# In[26]:


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)


# In[27]:


train = optimizer.minimize(cost)


# In[28]:


sess = tf.Session()


# In[29]:


sess.run(tf.global_variables_initializer())


# In[35]:


for i in range (2000):
    cost_val, W_val, b_val, _ = sess.run([cost,W,b,train],feed_dict={X:[1,2,3,4,5], Y:[1.1,2.2,3.3,4.4,5.5]})
    if i % 200 == 0:
        print(cost_val, W_val, b_val)


# In[ ]:




