# **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/left_full.png "full left camera image"
[image2]: ./examples/left_cropped.png "cropped left camera image"
[image3]: ./examples/right_full.png "full right camera image"
[image4]: ./examples/right_cropped.png "cropped right camera image"
[image5]: ./examples/center_full.png "full center camera image"
[image6]: ./examples/center_cropped.png "cropped center camera image"
[image7]: ./examples/center_full_2.png "full center camera image"
[image8]: ./examples/center_cropped_2.png "cropped center camera image"
[image9]: ./examples/center_full_3.png "full center camera image"
[image10]: ./examples/center_cropped_3.png "cropped center camera image"
[image11]: ./examples/left.jpg "sample left camera image"
[image12]: ./examples/center.jpg "sample center camera image"
[image13]: ./examples/right.jpg "sample right camera image"



### Files

This project includes the following files for submission:
* model.py -- the script to create and train the model
* drive.py -- the script that uses the mode to drive the car in autonomous mode
* model.h5 -- a trained convolution neural network 
* video.mp4 -- a video of the can beind driven in autonomous mode using the model
* README.md -- a writeup summarizing the results

### How to run the model
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### Driving Data

The following camera and steering angle data was captured using Udacity's simulator with a Playstation3 controller for steering input:
1. 4 laps of smooth center-lane driving (2 laps in each direction on the track).
2. 2 additional laps of smooth center-lane driving (1 lap in each direction).
3. 6 passes of the tight corners (L-R-L if going counter-clockwise) at the "top" of the track (3 in each direction). These are the only corners on the track that require a steering angle greater than 5 degrees. The recording only took place while the car was turning around these corners.
4. 1 lap of correction data. The car was allowed to drift near the inner and outer lane markers and then was recorded moving back into the center of the road.

Below are three image samples from the left, center and right "cameras":

![alt text][image11]

![alt text][image12]

![alt text][image13]

###Model Architecture and Training Strategy

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
