# **Behavioral Cloning Project**

[//]: # (Image References)

[image1]: ./examples/left.jpg "left camera image"
[image2]: ./examples/left_cropped.jpg "cropped left camera image"
[image3]: ./examples/center.jpg "center camera image"
[image4]: ./examples/center_cropped.jpg "cropped center camera image"
[image5]: ./examples/right.jpg "right camera image"
[image6]: ./examples/right_cropped.jpg "cropped right camera image"

## Introduction
### Goals
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

### Files
* model.py -- a script to create and train the model
* drive.py -- a script that uses the model to drive the car in autonomous mode
* model.h5 -- final trained Keras ConvNet 
* model2.h5 --  alternative trained Keras ConvNet
* README.md -- a writeup summarizing the results
* examples/video.mp4 -- a video of the car being driven in autonomous mode using the final model
* examples/corrections_video.mp4 -- a video of the car being driven in autonomous mode after manually moving the car to the side of the road (final model)
* examples/video2.mp4 -- a video of the car being driven in autonomous mode using the alternative model

### HowTo
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### Data
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

My first attempt was to train Nvidia's [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316.pdf) model (see below) using the 4 laps of the track. The model was implemented in Keras using a mean squared error loss function and the Adam optimizer. It was trained for 5 epochs and the data was randomly shuffled.

As recommended in the CS231n lecture, I used a weight initialization that has been shown to perform better for deep CNNs with RELU activations (Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification, He et al.). This was accomplished by setting init='he_normal' in the Keras model layers.

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

***Total params: 348,219***

Before training, I reviewed the images and decided to crop 70 pixels from the top of the image and 25 pixels from the bottom. This removes the car hood and background and the resulting image only contains areas where features relevant to steering direction can be discovered by the model. Cropped samples from the left, center and right cameras can be seen below.

![alt text][image2]

![alt text][image4]

![alt text][image6]

The Lambda normalization layer zero-centers the pixel values of the input image within a range of -0.5 to 0.5.

My first attempt used only the center camera images and a validation split of 0.2. It was obvious from the training and validation loss that the model was overfitting the training data. The car drove poorly and veered off at the first corner.

At this point, I added dropout regularization after each hidden fully-connected layer with a probability of 0.5. I also doubled the number of training and validation images by flipping each image over the y-axis and negating the associated steering measurement. The model did not seem to be overfitting as much, but still had trouble with drifting and tight corners in testing. This model always drifted off onto the dirt driveway in the first tight corner.

Next, I collected more data from the tightest corners on the track. I made six passes, recording only when the car was turning. I collected even more data by allowing the car to drift to the inside and outside lane markers and recording the car moving back towards the center of the road. I also used the left and right camera images by applying a 0.25 steering correction for the left camera a -0.25 correction for the right. Training the model described above with this data yielded the best result.

At one point, I attempted to use an additional 2 laps of center driving data, but I found that the model performed worse and the car drifted off the road in tight corners. I am not sure why or if this is still the case with the final model. It is clear from the results that this additional data is not needed.

## Results

The most successful architecture that I trained to predict steering angles is below.

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

***Total params: 348,219***

The hyperparamaters are:
 * epochs = 6
 * validation split = 0.2
 * dropout probability = 0.5
 * left/right steering correction = 0.25
 
I let this model steer the car around the track for 10+ laps without going off the road. Of all the models I tested it is by far the smoothest at driving around the track.

A video of the car driving 1.5 laps around track (tight corners twice) using this model can be found at examples/video.mp4. I included a video at examples/correction_video.mp4 that shows the car steering away from the edges of the track after I manually move it there.

I also trained a model that used RELU activations for the fully-connected layers. It can be seen below.

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

***Total params: 348,219***

After experimentation, this model was trained for 10 epochs with a 0.2? validation split and a 0.0001 learning rate for the Adam optimizer.

While the car succesfully goes around the track in autonomous mode, this model does not perform as well as the one above. It makes sharper corrections and the car weaves more than the first model. I believe that the first model performs better because the fully-connected layers are just linear combinations of the final convolutional layer and it is not overfitting the training data as much.

A video of the car driving around the track using this model can be found at examples/video2.mp4.

### Note

This does seem to be a very large model for the relatively simple simulated environment that we are working with. I experimented with smaller models that used convolutional layers like VGG and max pooling, but far less deep and I did not get good results.
