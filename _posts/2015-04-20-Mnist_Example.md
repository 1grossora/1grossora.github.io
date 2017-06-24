

```python
# Here we will walk through the MNIST example to learn a little more about tensorflow and how to use the framework 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)# This preloads the data sets
```

    Extracting MNIST_data/train-images-idx3-ubyte.gz
    Extracting MNIST_data/train-labels-idx1-ubyte.gz
    Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    Extracting MNIST_data/t10k-labels-idx1-ubyte.gz



```python
# now we import tensorflow 
import tensorflow as tf
#set up an interactive session 
sess = tf.InteractiveSession()
```


```python
# First we will being with a single layer... 
# We need to build nodes for input images and output classes 

x = tf.placeholder(tf.float32,shape=[None,784])
y_ = tf.placeholder(tf.float32,shape=[None,10])

# In this case x and y_ are not specific values... just placeholders. 
#When we run TF they will be take an imput walue. 
x
# Note that 784 is used becuase it will relate to a 28X28 pixel image 
# Note that 10 is used becuase it realates to the length of output number 0-9
```




    <tf.Tensor 'Placeholder:0' shape=(?, 784) dtype=float32>




```python
# Now we have to define weights and biases 
# We do this through variable in TF.

W = tf.Variable(tf.zeros([784,10])) # we have 784 imput weights and 10 output classes
b = tf.Variable(tf.zeros([10]))

# to bring these variables into the session we need to initialize 

sess.run(tf.global_variables_initializer())
```


```python
# Now we will try to impemnt the model. We do this by taking input images * weights and then adding the bias. 
# W*x + b

y = tf.matmul(x,W)+b
```


```python
# now we define the loss fucntion 
# loss function is a representation of how bas our model didi 
# We want to minimize the loss function

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits=y))
# here we see that labels are the actual labels for the image 
# y is the prediction we get from the model 
#nn.softmax_cross_entropy_withLogits applies the softmax to the model which is unnoremalized 
#Then reduce_mean takes the average over the sums 
cross_entropy
```




    <tf.Tensor 'Mean:0' shape=() dtype=float32>




```python
#Now we can train the model 
# Use gradient descent with step size 0.5
train_step =tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# What is going on ? 
# compute gradients, compute parameter update steps, and apply update steps to the parameters

```


```python
# Now to train... .we are going to add a batch of photos run the training on them\
# We will replace x and y_ with the actual values from the given batch

for _ in range(1000):
    batch = mnist.train.next_batch(100)# note 100 is batch size 
    train_step.run(feed_dict={x:batch[0],y_:batch[1]})# here we are feeding in a dictionary of the x and y_ from the batch
```


```python
# Well... how did we do? 
# we would like to figure out in which cases we predicted the correct label
model_label_guess = tf.arg_max(y,1) # this is the label that our model thinks is most likely for each input
label_true = tf.arg_max(y_,1) # This is simply the true label for the imput
# we want to see when they are correct 

correct_prediction = tf.equal(model_label_guess,label_true)
# This is a list of booleans

# to findout which are correct [ True, False, True, True,False] This is then cast as floats [1.,0.,1.,1.,0.]
# if we take the mean of this, we call this our accuracy.... how many it predicted correctly

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
```


```python
# now we want to evalue this accuracy on the test data

accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels})
```




    0.91720003




```python
# lets look at the parameter space for changing a few variables. 
# here we will just use lists ( since we won't append too much ), then convert to np.arrays and plot some things 

# First assume step size ranges from [0,1]: plot as a function of number of training steps 

pretty = False
step_size = 0.5 # just a guess to start , from experience 
batch_size = 100 # we have ten number slots 0-9 so 100 batch size should get some represenation of each number
max_steps = 500 #max number of steps ( remember it takes longer to do more)
nstep_step = 10#step_step

# if you want to make pretty use these 
if pretty: 
    max_steps = 1000 #max number of steps ( remember it takes longer to do more)
    nstep_step = 10#step_step
    
# to make life easy, let's keep this like a list of pairs [ [0_step,accuracy_0],[1_step,accuracy_1]... ]
accuracy_for_various_n_steps = []


# we will use zero ( it will be bad ) but it will also be a way to show random probilibty :) 

for n_steps in np.arange(0, max_steps, nstep_step):
    #Reset a bunch of things
    sess.run(tf.global_variables_initializer())
    temp_y = tf.matmul(x,W)+b
    temp_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits=temp_y))    
    temp_train_step = tf.train.GradientDescentOptimizer(step_size).minimize(temp_cross_entropy)
    for _ in range(int(n_steps)):
        temp_batch = mnist.train.next_batch(Batch_size)
        #Train it
        temp_train_step.run(feed_dict={x:temp_batch[0],y_:temp_batch[1]})
    #how many are correct? 
    temp_correct = tf.equal(tf.arg_max(temp_y,1),tf.arg_max(y_,1))
    temp_accuracy = tf.reduce_mean(tf.cast(temp_correct,tf.float32))
    temp_accuracy_eval = temp_accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels})
    temp_entry = [n_steps,temp_accuracy_eval]
    accuracy_for_various_n_steps.append(temp_entry)

print accuracy_for_various_n_steps




```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-11-375edabfe7aa> in <module>()
         21 # we will use zero ( it will be bad ) but it will also be a way to show random probilibty :)
         22 
    ---> 23 for n_steps in np.arange(0, max_steps, nstep_step):
         24     #Reset a bunch of things
         25     sess.run(tf.global_variables_initializer())


    NameError: name 'np' is not defined



```python
# let's take a look at how the accuracy stacks up. 
%matplotlib inline

import numpy as np 
import matplotlib
import matplotlib.pyplot as plt

def plot_avs():
    avs = np.asarray(accuracy_for_various_n_steps)
    plt.scatter(avs[1:,0],avs[1:,1])   
    plt.grid(True)
    plt.show()
plot_avs()
```


```python

# a pandas DF woulc be very useful for the next part....but let's see what we can do.
#Not everyone has pandas.... but more have numpy 

# So we say that somewhere around 500 steps is probably overkill 
# lets look at some more parts of parameter space 

#We would like to see the effect of varying batch size... maybe we simply dont have enough test samples? 
# Also: maybe the minimization is too fast or two slow.... Let's vary the step size (0.001,1)

# this will take a little while to run... .but it makes a useful plot 

pretty = False
n_training_steps =100 # RG CHANGE THIS BACK TO 500 
max_step_size = 1. # descent step size
step_size_step = 0.2 # various steps size
max_batch_size = 100 # how many samples in training
batch_size_step = 20 # various sample size for training 

b_size_list = [ z for z in range(0,max_batch_size,batch_size_step) ][1:]# remove 0 
s_size_list = np.arange(0,max_step_size,step_size_step)# keep 0.... again more fun math
print b_size_list
print s_size_list
acc_for_batch_step = [] # for list of list [ [bsize, s_size,acc]....] This might get a little heavy for append. we will see

for b_size in b_size_list:
    for s_size in s_size_list:
        sess.run(tf.global_variables_initializer())
        temp_y = tf.matmul(x,W)+b
        temp_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits=temp_y))    
        temp_train_step = tf.train.GradientDescentOptimizer(s_size).minimize(temp_cross_entropy)
        for _ in range(n_training_steps):
            temp_batch = mnist.train.next_batch(b_size)
            #Train it
            temp_train_step.run(feed_dict={x:temp_batch[0],y_:temp_batch[1]})
        #how many are correct? 
        temp_correct = tf.equal(tf.arg_max(temp_y,1),tf.arg_max(y_,1))
        temp_accuracy = tf.reduce_mean(tf.cast(temp_correct,tf.float32))
        temp_accuracy_eval = temp_accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels})
        temp_entry = [b_size,s_size,temp_accuracy_eval]
        acc_for_batch_step.append(temp_entry)

print acc_for_batch_step
```


```python
# make kde plot to see effect 
# make a scatter plot where weight is the accuracy 
def plot_bsa():
    #plot
    bsa = np.asarray(acc_for_batch_step)
    plt.scatter(bsa[:,0],bsa[:,1],c=bsa[:,2])
    plt.show()
plot_bsa()
```


```python
c = tf.Variable(tf.zeros([10]))
sess.run(tf.global_variables_initializer())
c.eval()
b.eval()
```


```python


```


```python
############### CHANGE IT UP..... DO BETTER

#first we need to make our own weight and bias variables 
#weights should have some noise to allow for symetrty breaking . 
# if you have all zeros you might get stuck with the gradiant and not go anywhere 

def weight_variable(shape):
    initial = tf.truncated_normal(shape,0,1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0,1,shape=shape)# note we keep the bias positive to avoid dead nodes
    return tf.Variable(initial)
    

#first lets define useful functions 

def con2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_2X2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
```


```python
#now make the first convolution layer. 
# we will make is 5X5 , 1 channel with 32 features  
# note 32 features is what's called a hyperparameter. It's just a number of features... we can make it anything

W_conv1 = weight_variable([5,5,1,32])
B_conv1 = bias_variable([32]) # note the number of bias is dependednt on number of features

# now reshape the image to a 4d tensor

x_image = tf.reshape(x,[-1,28,28,1]) # magic, dim, dim, color-chan

#now we are ready to convolve the x_image tensor with the weight tensor and then add the bias and apply RelU

h_conv1 = tf.nn.relu(con2d(x_image,W_conv1)+B_conv1)

#then we max_pool
h_pool1 = max_pool_2X2(h_conv1)
```


```python
#cool... now we have our first layer... .we can go deeper 
# When we go deeper we will connect (bad choice of words) the previous layer to the next

# this is layer 2

#note the incoming image is still 14*14 because we pooled by 2X2 28/2 X 28/2

W_conv2 = weight_variable([5,5,32,64]) # we use 32 becuase each incoming field of view has 32 features from the previous layer
B_conv2 = bias_variable([64])

# now do the convolution and pooling 
h_conv2 = tf.nn.relu(con2d(h_pool1,W_conv2)+B_conv2)
#now pool it 
h_pool2 = max_pool_2X2(h_conv2)
#note... this will now give us a size 7X7 image 14/2 X 14/2


```


```python
#now that we have a smaller image field... 
# add a fully connected layer: So the input is going to look like 7*7*64 since each pixel has now 64 features. 
#note, 32, and 64 are just arbitraty 
# so we do the same with this step These are hyper params that sweep 

W_fcl  = weight_variable([7*7*64,1024])#1024 is again arbitary 
B_fcl = bias_variable([1024])

# since it's a fully connected layer we are going to take the previous layer and put it in to a large vector
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
#now do our normal minization of matracies 
h_fcl = tf.nn.relu(tf.matmul(h_pool2_flat,W_fcl)+B_fcl)


```


```python
# if we run this full layer we will have a few issues 
# first issue... it's large 
#second issue... we may over fit 

# to avoid this we use dropout: randomly drop out nodes before testing 
# we want to have these nodes droped in training and kept in testing 
#luckily TF does this for us. 

keep_prob = tf.placeholder(tf.float32)
h_fcl_drop =tf.nn.dropout(h_fcl,keep_prob)
```


```python
#finally we want to end witha  soft max.... again ranging from [0,9]

W_fc2 = weight_variable([1024,10])
B_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fcl_drop,W_fc2)+B_fc2

```


```python
# now we have to just do the standard minimization technique 

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))
#Training step 
gradient_step_size = 1e-4
train_step = tf.train.AdamOptimizer(gradient_step_size).minimize(cross_entropy)
correct_prediction = tf.equal(tf.arg_max(y_conv,1),tf.arg_max(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

```


```python
# now actually run... .make sure we clean up globals 

#Training iterations 
training_itr = 200
batch_size = 50 

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(training_itr):
        batch = mnist.train.next_batch(batch_size)
        if i %100 ==0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1] ,keep_prob:1.0}) #note we keep the whole network together to test 
            print('step %d training accuracy %g' % (i,train_accuracy))
        #keep training 
        train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})#drop out half the network nodes 
    #Finally 
    print 'we are out'
    # we run out of memory for the whole dataset? 
    my_batch = mnist.test.next_batch(10000)
    
    my_acc = accuracy.eval(feed_dict={x:my_batch[0],y_:my_batch[1],keep_prob:1.0})
    
    #my_acc = accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0})
    print 'past'
    print my_acc
    #print('total test accuracy %g' % accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))
```

    step 0 training accuracy 0.02
    step 100 training accuracy 0.26
    we are out



```python
mnist.test.num_examples
```




    10000




```python

```
