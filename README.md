# **Behavioral Cloning Project**

[//]: # (Image References)

[image1]: ./examples/left.jpg "left camera image"
[image2]: ./examples/left_cropped.jpg "cropped left camera image"
[image3]: ./examples/center.jpg "center camera image"
[image4]: ./examples/center_cropped.jpg "cropped center camera image"
[image5]: ./examples/right.jpg "right camera image"
[image6]: ./examples/right_cropped.jpg "cropped right camera image"

## Introduction
#### Goals
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

#### Files
* model.py -- the script to create and train the model
* drive.py -- the script that uses the mode to drive the car in autonomous mode
* model.h5 -- final trained Keras ConvNet 
* model2.h5 --  alternative trained Keras ConvNet
* README.md -- a writeup summarizing the results
* examples/video.mp4 -- a video of the car being driven in autonomous mode using the final model
* examples/corrections_video.mp4 -- a video of the car being driven in autonomous mode after manually moving the car to the side of the road (final model)
* examples/video2.mp4 -- a video of the car being driven in autonomous mode using the alternative model

#### HowTo
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### Data
The following camera and steering angle data was captured using Udacity's simulator with a Playstation3 controller for steering input:
 * **4** laps of smooth center-lane driving (2 laps in each direction on the track).
 * **2** additional laps of smooth center-lane driving (1 lap in each direction).
 * **6** passes of the tight corners (L-R-L if going counter-clockwise) at the "top" of the track (3 in each direction). These are the only corners on the track that require a steering angle greater than 5 degrees. Data was recorded only while the car was turning.
 * **1** lap of correction data. The car was allowed to drift near the inner and outer lane markers and then was recorded moving back into the center of the road.

Below are three sample images from the left, center and right cameras:

![alt text][image1]

![alt text][image3]

![alt text][image5]

## Approach

My first attempt was to train Nvidia's [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316.pdf) model (see below) using the 4 laps of the track. The model was implemented in Keras using a mean squared error loss function and the Adam optimizer. It was trained for 5 epochs.

| Layer                                                        | Output Shape  |  Params  |
|:-------------------------------------------------------------|:--------------|:---------|
| Input                                                        | (160, 320, 3) |  0       |
| Cropping2D((70, 25), (0, 0))                                 | (65, 320, 3)  |  0       |
| Lambda(x / 255.0 - 0.5)                                      | (65, 320, 3)  |  0       |
| Convolution2D(24, 5, 5, activation='relu', subsample=(2, 2)) | (31, 158, 24) |  1824    |
| Convolution2D(36, 5, 5, activation='relu', subsample=(2, 2)) | (14, 77, 36)  |  21636   |
| Convolution2D(48, 5, 5, activation='relu', subsample=(2, 2)) | (5, 37, 48)   |  43248   |
| Convolution2D(64, 3, 3, activation='relu')                   | (3, 35, 64)   |  27712   |
| Convolution2D(64, 3, 3, activation='relu')                   | (1, 33, 64)   |  36928   |
| Flatten                                                      | (2112)        |  0       |
| Dense(100, activation='linear')                              | (100)         |  211300  |
| Dense(50, activation='linear')                               | (50)          |  5050    |
| Dense(10, activation='linear')                               | (10)          |  510     |
| Dense                                                        | (1)           |  11      |

***Total params: 348,219

Before training, I reviewed the images and decided to crop 70 pixels from the top of the image and 25 pixels from the bottom. This removes the car hood and background and the resulting image only contains areas where features relevant to steering direction can be discovered by the model. Cropped samples from the left, center and right cameras can be seen below.

![alt text][image2]

![alt text][image4]

![alt text][image6]

The Lambda normalization layer zero centers the pixel values of the input image within a range of -0.5 to 0.5.

My first attempt used only the center camera images and a validation split of 0.2. It was obvious from the training and validation loss that the model was overfitting the training data. The car drove poorly and veered off at the first corner.

At this point, I added dropout regularization after each hidden fully-connected layer with a rate of 0.5. I also doubled the number of training and validation images by flipping each images over the y axis and adjusting the associated measurement appropriately. The model did not seem to be overfitting as much, but still had trouble with drifting and corners.

## Results

The most successful architecture that I trained to drive on the track is below. It is the model found in the Nvidia [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316.pdf) paper with an image cropping layer, normalization layer (Lambda) and dropout on the hidden fully-connected layers.

| Layer                |      Output Shape  |  Params  |
|:-------------------------------------------------------------|:--------------|:---------|
| Input                                                        | (160, 320, 3) |  0       |
| Cropping2D((70, 25), (0, 0))                                 | (65, 320, 3)  |  0       |
| Lambda(x / 255.0 - 0.5)                                      | (65, 320, 3)  |  0       |
| Convolution2D(24, 5, 5, activation='relu', subsample=(2, 2)) | (31, 158, 24) |  1824    |
| Convolution2D(36, 5, 5, activation='relu', subsample=(2, 2)) | (14, 77, 36)  |  21636   |
| Convolution2D(48, 5, 5, activation='relu', subsample=(2, 2)) | (5, 37, 48)   |  43248   |
| Convolution2D(64, 3, 3, activation='relu')                   | (3, 35, 64)   |  27712   |
| Convolution2D(64, 3, 3, activation='relu')                   | (1, 33, 64)   |  36928   |
| Flatten                                                      | (2112)        |  0       |
| Dense(100, activation='linear')                              | (100)         |  211300  |
| Dropout(0.5)                                                 | (100)         |  0       |
| Dense(50, activation='linear')                               | (50)          |  5050    |
| Dropout(0.5)                                                 | (50)          |  0       |
| Dense(10, activation='linear')                               | (10)          |  510     |
| Dropout(0.5)                                                 | (10)          |  0       |
| Dense                                                        | (1)           |  11      |

***Total params: 348,219

The Lambda normalization layer zero centers the pixel values of the input image withing a range of -0.5 to 0.5.

### Training the model



| Layer                |      Output Shape  |  Params  |
|:-------------------------------------------------------------|:--------------|:---------|
| Input                                                        | (160, 320, 3) |  0       |
| Cropping2D((70, 25), (0, 0))                                 | (65, 320, 3)  |  0       |
| Lambda(x / 255.0 - 0.5)                                      | (65, 320, 3)  |  0       |
| Convolution2D(24, 5, 5, activation='relu', subsample=(2, 2)) | (31, 158, 24) |  1824    |
| Convolution2D(36, 5, 5, activation='relu', subsample=(2, 2)) | (14, 77, 36)  |  21636   |
| Convolution2D(48, 5, 5, activation='relu', subsample=(2, 2)) | (5, 37, 48)   |  43248   |
| Convolution2D(64, 3, 3, activation='relu')                   | (3, 35, 64)   |  27712   |
| Convolution2D(64, 3, 3, activation='relu')                   | (1, 33, 64)   |  36928   |
| Flatten                                                      | (2112)        |  0       |
| Dense(100, activation='relu')                                | (100)         |  211300  |
| Dropout(0.5)                                                 | (100)         |  0       |
| Dense(50, activation='relu')                                 | (50)          |  5050    |
| Dropout(0.5)                                                 | (50)          |  0       |
| Dense(10, activation='relu')                                 | (10)          |  510     |
| Dropout(0.5)                                                 | (10)          |  0       |
| Dense                                                        | (1)           |  11      |

Total params: 348,219


####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
