**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/readme_1.png "Train Image"
[image2]: ./examples/readme_2.png "Label Counts"
[image3]: ./examples/readme_3.png "Propability Distribution"
[image4]: ./examples/1_100.jpg "Traffic Sign 1"
[image5]: ./examples/2_100.jpg "Traffic Sign 2"
[image6]: ./examples/3_100.jpg "Traffic Sign 3"
[image7]: ./examples/4_100.jpg "Traffic Sign 4"
[image8]: ./examples/5_100.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
####1. Here is a link to my [project code](https://github.com/fengjihua/CarND-Traffic-Sign-Classifier-P2/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, I used the python and pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43


####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
* Train data is a 32*32*3 RGB image
* It is a bar chart showing the distribution of train labels class count

![alt text][image1]
![alt text][image2]

###Design and Test a Model Architecture

####1. Preprocess the data

* First step, augment the training data.

I augemented the trainging set because it might help improve model performace.I used keras.preprocessing.image.ImageDataGenerator to generate the fake data. I just change a little bit width and height shift range of the original image to generate fake image, I didn't use horizonal or vertical flip because it will change the rotation of some right, left, ahead traffic signs in training data. The augmentation worked very well. Finally, I generated 30000 training data from 34799 to 64799.

Before Augment:  (34799, 32, 32, 3) (34799,)

After Augment:  (64799, 32, 32, 3) (64799,)


* Second step, normalize the image data.

I normalized by (pixel - 128)/ 128

After normalization, all training valid and test data in range [-1,1], it is a good way to remove the effect of any intensity variations.

Before Normalize:  [ 28.  25.  24.]

After Normalize:  [-0.78125   -0.8046875 -0.8125   ]


####2. Build Model

My final model consisted of the following layers:

| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x3 RGB image                             | 
| Convolution 5x5x64    | 1x1 stride, valid padding, outputs 28x28x64   |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 14x14x6                  |
| Convolution 5x5x128   | 1x1 stride, valid padding, outputs 10x10x128  |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 5x5x16                   |
| Faltten               | Output = 3200                                 |
| Fully connected       | Input = 3200. Output = 1024                   |
| Fully connected       | Input = 1024. Output = 512                    |
| Fully connected       | Input = 512. Output = n_classes(43)           |
 
I do some changes on the base of LeNet model
* 1.Convolution Layer1 use 64 kernels instead of 6 kernels
* 2.Convolution Layer2 use 128 kernels instead of 16 kernels
* 3.Fully connected Layer1 use 1024 outputs instead of 120
* 4.Fully connected Layer2 use 512 outputs instead of 84

I increase convolution kernels and fully connected outputs to imrpove the performance of the model, though it take much more time to train. Finally i got a better reward.


####3. Train Model

To train the model, I used Cross Entropy to calculate my loss function and use AdamOptimizer to optimize my weights

Here are my hyperparameters:
* epochs = 30
* batch_size = 64
* learning_rate = 0.001

I also add L2 regularization to the Cross Entropy to prevent over-fitting, L2 factor is 5e-4


####4. My final model results were:
* training set accuracy of 0.993
* validation set accuracy of 0.963 
* test set accuracy of 0.951

At first time iteration, I use standard LeNet model to train. LeNet model include layers below:
* Convolution layer: to make image features deeper
* ReLu layer: to activate the convolution layer features
* Maxpooling Layer: to subsampling the features, make a lowers calcution for model. This step can be passed if we want a better result, but will train slower
* Fully connected layer: connect the features inputs to the final classes outputs

I got about 0.930 accuracy on LeNet Model. 

Then I do some changes as mentioned before to improve the performance of the model. 
* Increase kernels of convolution layer
* Increase outputs of fully connected layer
* add l2 regularization to prevent over-fitting

Finally I got a better result about 0.960 accuracy.



###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]


My model takes 32*32*3 RGB images, I download five high resolution images from web. I found that it lost many accuracy after resize from a high resolution to 32*32, and the result is not good enough. So i download some images with a lower resolution, then I got a better result than before.


####2. Here are the results of the prediction:

| Image                 |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Yield                 | Yield                                         | 
| No Entry              | No Entry                                      |
| Stop                  | Stop                                          |
| 30 km/h               | 30 km/h                                       |
| Ahead only            | Ahead only                                    |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 1.00. The result is better than the accuracy of test set 0.951 because I only predict 5 web images.

I Strongly recommeded small size to test, because resize image will lose resolution and model can not work very well.


####3. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

![alt text][image3]