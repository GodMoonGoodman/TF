
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


tf.__version__


# In[6]:


hello = tf.constant("Hello ?")


# In[7]:


sess = tf.Session()


# In[8]:


sess.run(hello)


# In[20]:


a = tf.placeholder(tf.float32)


# In[21]:


b = tf.placeholder(tf.float32)


# In[22]:


c = a+b


# In[23]:


a


# In[24]:


b


# In[25]:


c


# In[28]:


sess.run(c,feed_dict={a:1,b:2})


# In[ ]:




