
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a Traffic Sign Recognition Classifier
# 
# In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 
# 
# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 
# 
# In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.
# 
# The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.
# 
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ---
# ## Step 0: Load The Data

# In[1]:

# load pickled data
import pickle
import pandas as pd

# TODO: Fill this in based on where you saved the training and testing data
training_file = './data/train.p'
validation_file = './data/valid.p'
testing_file = './data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# Read signnames.csv
df_sign=pd.read_csv('./signnames.csv')
# print (df_sign.head())
# print (df_sign.tail())
# print(df_sign.loc[df_sign.loc[:, 'ClassId']==40, 'SignName'].values[0])


# ---
# 
# ## Step 1: Dataset Summary & Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
# 
# Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

# ### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas

# In[2]:

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = X_valid.shape[0]

# TODO: Number of validation examples
n_validation = X_valid.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of valid examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

print (df_sign.head())
print (df_sign.tail())


# ### Include an exploratory visualization of the dataset

# Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 
# 
# The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.
# 
# **NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?

# In[3]:

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
get_ipython().magic('matplotlib inline')

index = 88
print(y_train[index])
plt.figure(figsize=(1,1))
plt.imshow(X_train[index])
# plt.imshow(rgb2gray(X_train[index]), cmap = plt.cm.gray)
plt.show()

label_counts = pd.Series(train['labels']).value_counts()
# print (label_counts)
ax=label_counts.plot(kind='bar', figsize=(20,6))
ax.set_ylabel('Count')
ax.set_xlabel('Class ID')
ax.set_title('Distribution of Sample Count per Class')


# ----
# 
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 
# 
# With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 
# 
# There are various aspects to consider when thinking about this problem:
# 
# - Neural network architecture (is the network over or underfitting?)
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

# ### Pre-process the Data Set (normalization, grayscale, etc.)

# Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 
# 
# Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 
# 
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.

# In[4]:

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

import numpy as np
from skimage.color import rgb2gray,rgb2grey,rgb2hsv
from keras.preprocessing.image import ImageDataGenerator


# In[5]:

# 1.AUGMENT THE TRAINING DATA
# Augmenting the training set might help improve model performance. Common data augmentation techniques include rotation, translation, zoom, flips, and/or color perturbation. These techniques can be used individually or combined.
# https://keras.io/preprocessing/image/ 
Datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=False,
    vertical_flip = False,
    fill_mode='nearest')

# x_image = X_train[0:2]
# y_image = y_train[0:2]
# # print(x_image.shape, np.max(x_image[1]), np.min(x_image[1]))
print('Before Augment: ', X_train.shape, y_train.shape, y_train[-10:])
# print(len(X_train), len(y_train))

num_augment = 0
limit_augment = 30000

X_aug = None
y_aug = None
# for x_gen, y_gen in Datagen.flow(X_train, y_train, batch_size=1, save_to_dir='./data/augmentation', save_prefix=num_augment, 
#                           save_format='jpg'):
for x_gen, y_gen in Datagen.flow(X_train, y_train, batch_size=1):
    num_augment +=1
#     print(i, np.max(x_gen), np.min(x_gen), x_gen.shape, y_gen)
    if X_aug is None:
        X_aug = x_gen
        y_aug = y_gen
    else:
        X_aug = np.append(X_aug, x_gen, axis=0)
        y_aug = np.append(y_aug, y_gen, axis=0)
    
    if num_augment%500 == 0:
        X_train = np.append(X_train, X_aug, axis=0)
        y_train = np.append(y_train, y_aug, axis=0)
        X_aug = None
        y_aug = None
        print('Augment in progress {}/{}...'.format(num_augment,limit_augment))
    
    if num_augment > limit_augment - 1:
        break
        
print('After Augment: ', X_train.shape, y_train.shape, y_train[-10:])


# In[ ]:

# 2. Grayscale the image
# X_train = np.dot(X_train[...,:3], [0.299, 0.587, 0.144])
# X_train = rgb2gray(X_train)
# X_train = X_train.reshape(X_train.shape + (1,))
# X_valid = rgb2gray(X_valid)
# X_valid = X_valid.reshape(X_valid.shape + (1,))
# X_test = rgb2gray(X_test)
# X_test = X_test.reshape(X_test.shape + (1,))

# print(X_train.shape)

# plt.figure()
# # plt.subplot(221)
# # plt.imshow(X_train[index])

# # plt.subplot(222)
# plt.imshow(X_train[index].reshape((32,32)), cmap = plt.cm.gray)
# plt.show()


# In[6]:

# KFold

# 3.Nomalize
print('Before Normalize: ', np.min(X_train), np.max(X_train), np.mean(X_train))
X_train = np.divide(np.add(X_train, -128), 128)
X_valid = np.divide(np.add(X_valid, -128), 128)
X_test = np.divide(np.add(X_test, -128), 128)
print('After Normalize: ', np.min(X_train), np.max(X_train), np.mean(X_train))


# ### Model Architecture

# In[7]:

### Define your architecture here.
### Feel free to use as many code cells as needed.

import tensorflow as tf
from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    keep_prob = 1.0
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x64.
#     conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma))
#     conv1_b = tf.Variable(tf.zeros(6))
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 64), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(64))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID', name='conv1') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x64. Output = 14x14x64.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x128.
#     conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
#     conv2_b = tf.Variable(tf.zeros(16))
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 64, 128), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(128))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID', name='conv2') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x128. Output = 5x5x128.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x128. Output = 3200.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 3200. Output = 1024.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(3200, 1024), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(1024))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # SOLUTION: Layer 4: Fully Connected. Input = 1024. Output = 512.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(1024, 512), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(512))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob)

    # SOLUTION: Layer 5: Fully Connected. Input = 512. Output = n_classes(43).
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(512, n_classes), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    # L2 Regularization
    regularizers = (tf.nn.l2_loss(fc1_W) + tf.nn.l2_loss(fc1_b) + tf.nn.l2_loss(fc2_W) + tf.nn.l2_loss(fc2_b) + tf.nn.l2_loss(fc3_W) + tf.nn.l2_loss(fc3_b))
    
    return logits, regularizers


# ### Train, Validate and Test the Model

# A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
# sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

# In[10]:

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

# print(X_train[0][0][0])

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = tf.placeholder(tf.float32, (None, 32, 32, 3), name='x') # (n,32,32,3)
y = tf.placeholder(tf.int32, (None), name='y') # (n)
one_hot_y = tf.one_hot(y, n_classes) # (n,43)

# Hyperparameters
epochs = 30
batch_size = 64
learning_rate = 0.001
factor = 5e-4 # Regularization factor

# Train Operation
logits, regularizers = LeNet(x) # (n,43)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits) # (n,43)
loss_operation = tf.reduce_mean(cross_entropy)
# L2 regularization for the fully connected parameters. Add regularization to loss term
loss_operation += factor * regularizers

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_operation = optimizer.minimize(loss_operation)


# In[11]:

# Model Evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1)) # (n,1)
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # (n,1) -> a number

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    sess = tf.get_default_session()
    
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
        
        accuracy = sess.run(accuracy_operation, feed_dict = {x: batch_x, y: batch_y}) # a number
        total_accuracy += (accuracy * len(batch_x))
        
        loss = sess.run(loss_operation, feed_dict = {x: batch_x, y: batch_y})
        total_loss += (loss * len(batch_x))
        
    return total_accuracy / num_examples, total_loss / num_examples


# In[12]:

from sklearn.utils import shuffle

# Train the model
print_number = 0
print_every_number = 200

train_loss = []
train_accs = []
valid_loss = []
valid_accs = []

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(epochs):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, batch_size):
            print_number += 1
            end = offset + batch_size
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
            if print_number % print_every_number == 0:
                accuracy, loss = evaluate(X_valid, y_valid)
                print("EPOCH {}/{}, Valid Loss = {:.3f}, Validation Accuracy = {:.3f}..."
                      .format(i+1, epochs, loss, accuracy))
                valid_loss.append(loss)
                valid_accs.append(accuracy)
                
                accuracy, loss = evaluate(X_train, y_train)
                print("Train Loss = {:.3f}, Train Accuracy = {:.3f}".format(loss, accuracy))
                train_loss.append(loss)
                train_accs.append(accuracy)
#                 print()
        
    saver.save(sess, './lenet')
    print("Model saved")


# In[13]:

# Plot classification accuracy on both training and validation set for better visualization.
fig = plt.figure()
fig.add_subplot(121)
plt.plot(train_accs, linewidth=1)
plt.plot(valid_accs, linewidth=1)
plt.legend(["Train Accuracy", "Valid Accuracy"], loc=4)
plt.grid(True)

fig.add_subplot(122)
plt.plot(train_loss, linewidth=1)
plt.plot(valid_loss, linewidth=1)
plt.legend(["Train Loss", "Valid Loss"], loc=1)
plt.grid(True)
plt.show()


# In[14]:

# Test the model
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy, test_loss = evaluate(X_test, y_test)
    print("Test Loss = {:.3f} ,Test Accuracy = {:.3f}".format(test_loss, test_accuracy))


# ---
# 
# ## Step 3: Test a Model on New Images
# 
# To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.
# 
# You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

# ### Load and Output the Images

# In[28]:

### Load the images and plot them here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt  
import matplotlib.image as mpimg
from PIL import Image
import numpy as np 

X_sample = list()
for i in range(1,6):
    image = Image.open('./data/sample/{}_100.jpg'.format(i))
    image_resize = image.resize((32,32))

    image_final = np.array(image_resize)
#     print(image_final.shape)
#     print(type(image), type(image_final))
    X_sample.append(image_final)

X_sample = np.array(X_sample)
print(X_sample.shape)
plt.figure(figsize=(1,1))
plt.imshow(X_sample[3])


# ### Predict the Sign Type for Each Image

# In[29]:

### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.

# Grayscale the sample
# X_sample = rgb2gray(X_sample)
# X_sample = X_sample.reshape(X_sample.shape + (1,))

# print(X_sample.shape)
# plt.figure(figsize=(1,1))
# plt.imshow(X_sample[3].reshape((32,32)), cmap = plt.cm.gray)
# plt.show()


# In[30]:

# Normalize the sample images
print('Before Normalize: ', np.min(X_sample), np.max(X_sample))
X_sample = np.divide(np.add(X_sample, -128), 128)
print('After Normalize: ', np.min(X_sample), np.max(X_sample))


# In[31]:

# Run the model to predict
prediction = tf.argmax(logits,1) # (n,43) -> (n,1)

preds = []
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    sess = tf.get_default_session()
    preds = sess.run(prediction, feed_dict={x: X_sample})
    print('Predictions: ',preds)


# ### Analyze Performance

# In[32]:

### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
trues = [13,17,14,1,35]
print('Accuracy of sample images: ',np.mean(preds == trues))


# ### Output Top 5 Softmax Probabilities For Each Image Found on the Web

# For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 
# 
# The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.
# 
# `tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.
# 
# Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:
# 
# ```
# # (5, 6) array
# a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
#          0.12789202],
#        [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
#          0.15899337],
#        [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
#          0.23892179],
#        [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
#          0.16505091],
#        [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
#          0.09155967]])
# ```
# 
# Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:
# 
# ```
# TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
#        [ 0.28086119,  0.27569815,  0.18063401],
#        [ 0.26076848,  0.23892179,  0.23664738],
#        [ 0.29198961,  0.26234032,  0.16505091],
#        [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
#        [0, 1, 4],
#        [0, 5, 1],
#        [1, 3, 5],
#        [1, 4, 3]], dtype=int32))
# ```
# 
# Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.

# In[33]:

### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.

softmax = tf.nn.softmax(logits)
softmax_all = []
softmax_top5 = []
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    sess = tf.get_default_session()
    softmax_all = sess.run(softmax, feed_dict={x: X_sample})
    softmax_top5 = sess.run(tf.nn.top_k(tf.constant(softmax_all), k=5))
    print('Softmax: ',softmax_top5)

print(softmax_top5.indices[0])
    
plt.figure(figsize=(5,5))
for i in range(len(X_sample)):
    plt.subplot(5,2,(i+1)*2-1)
    plt.imshow(X_sample[i])
    
    plt.subplot(5,2,(i+1)*2)
    plt.axis('off')
    plt.text(0, 0.5, "Probability: {:.2f} {}, {:.2f} {}, {:.2f} {}, {:.2f} {}, {:.2f} {}".format(
        softmax_top5.values[i][0], df_sign.loc[df_sign.loc[:, 'ClassId']==softmax_top5.indices[i][0], 'SignName'].values[0],
        softmax_top5.values[i][1], df_sign.loc[df_sign.loc[:, 'ClassId']==softmax_top5.indices[i][1], 'SignName'].values[0],
        softmax_top5.values[i][2], df_sign.loc[df_sign.loc[:, 'ClassId']==softmax_top5.indices[i][2], 'SignName'].values[0],
        softmax_top5.values[i][3], df_sign.loc[df_sign.loc[:, 'ClassId']==softmax_top5.indices[i][3], 'SignName'].values[0],
        softmax_top5.values[i][4], df_sign.loc[df_sign.loc[:, 'ClassId']==softmax_top5.indices[i][4], 'SignName'].values[0],
    )
            , size = 15)


# ### Project Writeup
# 
# Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

# ---
# 
# ## Step 4 (Optional): Visualize the Neural Network's State with Test Images
# 
#  This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.
# 
#  Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.
# 
# For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.
# 
# <figure>
#  <img src="visualize_cnn.png" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above)</p> 
#  </figcaption>
# </figure>
#  <p></p> 
# 

# In[34]:

### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    print(activation.shape)
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(16,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('Map ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")


# In[35]:

image_input = X_sample[0]
# Plot what we are passing to the network
plt.figure(figsize=(1,1))
plt.imshow(image_input)
print(image_input.shape)


# In[37]:

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    image_input = image_input.reshape(1,32,32,3)
    conv_layer_1_visual = sess.graph.get_tensor_by_name('conv1:0')
    outputFeatureMap(image_input,conv_layer_1_visual)
    
    conv_layer_2_visual = sess.graph.get_tensor_by_name('conv2:0')
    outputFeatureMap(image_input,conv_layer_2_visual)


# In[ ]:



