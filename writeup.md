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

[image1]: ./examples/1.png "Visualization"
[image4]: ./examples/1.jpg "Traffic Sign 1"
[image5]: ./examples/2.jpg "Traffic Sign 2"
[image6]: ./examples/3.jpg "Traffic Sign 3"
[image7]: ./examples/4.jpg "Traffic Sign 4"
[image8]: ./examples/5.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
####1. Here is a link to my [project code](https://github.com/fengjihua/CarND-Traffic-Sign-Classifier-P2/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. I used the python library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Preprocess the data

Normalize the image data by: (pixel - 128)/ 128 

Before Normalize:  [ 28.  25.  24.]
After Normalize:  [-0.78125   -0.8046875 -0.8125   ]

####2. Build LeNet model

My final model consisted of the following layers:

| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x3 RGB image                             | 
| Convolution 3x3       | 1x1 stride, valid padding, outputs 28x28x6    |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 14x14x6                  |
| Convolution 3x3       | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 5x5x16                   |
| Faltten               | Output = 400                                  |
| Fully connected       | Input = 400. Output = 120                     |
| Fully connected       | Input = 120. Output = 84                      |
| Fully connected       | Input = 84. Output = n_classes(43)            |
 


####3. Train Model

To train the model, I used Cross Entropy to calculate my loss function and use AdamOptimizer to optimize my weights

Here are my hyperparameters:
* epochs = 20
* batch_size = 64
* learning_rate = 0.001

I also add L2 regularization to the Cross Entropy, L2 factor is 5e-4

####4. My final model results were:
* validation set accuracy of 0.925 
* test set accuracy of 0.916


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

####2. Here are the results of the prediction:

| Image                 |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Yield                 | Yield                                         | 
| Stop                  | Stop                                          |
| Turn Right            | Turn Right                                    |
| 60 km/h               | 50 km/h                                       |
| Go straight or left   | Go straight or left                           |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%.
Strongly recommeded small size to test, because resize image will lose resolution and model can not work very well.

####3. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

Sample Image 1

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 1.0                   | Yield                                         | 
| 0.0                   | 20 km/h                                       |
| 0.0                   | 30 km/h                                       |
| 0.0                   | 50 km/h                                       |
| 0.0                   | 60 km/h                                       |


Sample Image 2

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 1.0                   | No Entry                                      | 
| 0.0                   | Traffic signals                               |
| 0.0                   | Bumpy road                                    |
| 0.0                   | Yield                                         |
| 0.0                   | No vehicles                                   |

Sample Image 3

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 0.98                  | Turn right ahead                              | 
| 0.02                  | Roundabout mandatory                          |
| 0.0                   | Ahead only                                    |
| 0.0                   | Turn left ahead                               |
| 0.0                   | 100 km/h                                      |

Sample Image 4

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 0.999                 | 50 km/h                                       | 
| 0.001                 | 60 km/h                                       |
| 0.0                   | 80 km/h                                       |
| 0.0                   | Wild animals crossing                         |
| 0.0                   | 30 km/h                                       |

Sample Image 5

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 1.0                   | Go straight or lef                            | 
| 0.0                   | Roundabout mandatory                          |
| 0.0                   | Turn right ahead                              |
| 0.0                   | Keep left                                     |
| 0.0                   | Keep right                                    |