# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/example.jpg "Example Image"
[image2]: ./examples/recovery1.jpg "Recovery Image 1"
[image3]: ./examples/recovery2.jpg "Recovery Image 2"
[image4]: ./examples/normal.jpg "Normal Image"
[image5]: ./examples/flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolutional neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 57-74) 

The model includes RELU layers to introduce nonlinearity (code lines 63-67, 70-72), and the data is normalized in the model using a Keras lambda layer (code line 60). 

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (model.py line 68). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 15). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 74).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving and recovering from the left and right sides of the road by driving on the first track.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to 

My first step was to use a convolution neural network model similar to the one the autonomous vehicle team at Nvidia used that was shown in the lesson. I thought this model might be appropriate because it was already used to tackle the same problem of driving an autonomous vehicle.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it included a dropout layer.

Then I run the simulator to see how well the car was driving around track two. It drove fairly well but got stuck at an S curve after a while.

Then I captured data again but on the first track. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I increased the number of training epochs. It didn't help. So I decided to train on the data produced by Udacity.

The final step was to run the simulator to see how well the car was driving around track one. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 57-74) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image  							| 
| Cropping         		| 160x225x3 RGB image  							|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 78x111x3 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 37x54x3 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 17x25x3 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 15x23x3 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 13x21x3 	|
| RELU					|												|
| Dropout				| Keep probability of 50%						|
| Flatten				| outputs 819									|
| Fully connected		| outputs 100									|
| RELU					|												|
| Fully connected		| outputs 50									|
| RELU					|												|
| Fully connected		| outputs 10									|
| RELU					|												|
| Fully connected		| outputs 1 									|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded nine laps on track one using center lane driving. Here is an example image of center lane driving:

![Example Image][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct its trajectory in case it drifted from the center. These images show what a recovery looks like starting from the left and right side of the road respectively:

![Recovery Image 1][image2]
![Recovery Image 2][image3]

To augment the data set, I also flipped images and angles thinking that this would help generalize better. For example, here is an image that has then been flipped:

![Normal Image][image4]
![Flipped Image][image5]

After the collection process, I had 44974 number of data points. I then preprocessed this data by centering around zero with small standard deviation.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the validation error increasing after 5 epochs: 

Epoch 1/7
201/201 [==============================] - 106s 526ms/step - loss: 0.0116 - val_loss: 0.0109
Epoch 2/7
201/201 [==============================] - 78s 386ms/step - loss: 0.0102 - val_loss: 0.0102
Epoch 3/7
201/201 [==============================] - 78s 389ms/step - loss: 0.0099 - val_loss: 0.0102
Epoch 4/7
201/201 [==============================] - 78s 389ms/step - loss: 0.0099 - val_loss: 0.0101
Epoch 5/7
201/201 [==============================] - 78s 390ms/step - loss: 0.0098 - val_loss: 0.0100
Epoch 6/7
201/201 [==============================] - 79s 391ms/step - loss: 0.0096 - val_loss: 0.0101
Epoch 7/7
201/201 [==============================] - 79s 394ms/step - loss: 0.0096 - val_loss: 0.0104

I used an adam optimizer so that manually training the learning rate wasn't necessary.
