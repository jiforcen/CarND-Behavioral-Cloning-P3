#**Behavioral Cloning** 

##Writeup report
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./write_up_images/ErrorLoss.png "Error Loss"
[image2]: ./write_up_images/imagePrev.png ""
[image3]: ./write_up_images/imagePost.png ""
[image4]: ./write_up_images/imagePrevB.png ""
[image5]: ./write_up_images/imagePostB.png ""
[image6]: ./write_up_images/imagePrevM.png "Image"
[image7]: ./write_up_images/imagePostM.png "Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py 'model.h5'
```
Also to train the neural network model.py file can be execute like in the next line:
```sh
python model.py
```
It will output the file model.h5 with the trained model.

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In the begining models like LeNet were tested, but quite forward Nvidia network architecture was tested. It was necesary add Dropout layers (Also L2 regularization was tested) to prevent overfitting.

Car tend to go out off the track until good aproach was applied. Balanced data was one of the keys of the project, because in data aquisition are too much images with steering near to 0, if we train the model with this data we will have a biased model, probably with a good value in error loss, but bad in driving.

Also all images were equlized before fed to the neural network

Different images sets were used to train the model during the set-up proccess. 
First data acquired manually was used to train the model and data from udacity was used for validation. Later when data of track 2 was acquired, all data was split into train dataset and validation dataset because udacity data doesn´t contain data of track 2 so is not a good validation data for both tracks.

Lot of driving test were made to check how well the car was driving around both tracks. There were a few spots where the vehicle fell off the track to improve the driving behavior in these cases data of these areas were collected.

At the end of the process, the vehicle is able to drive autonomously around both tracks without leaving the road. Track one is completed at 30mph and track two at 20mph.

My model consists of a convolution neural network inspired in nvidia neural network [geometric transformations](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the next table al the layers are described:

(model.py lines !!-!!) 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 3@66x200 YUV image   							| 
| Lambda layer    	| Normalization betwen -0.5 and 0.5 |
| Convolution2D 5x5     	| 2×2 stride, 5×5 kernel, outputs 24@31x98 	|
| RELU					|	Activation layer			|
| Dropout					|	0.2	dropout 				|
| Convolution2D 5x5     	| 2×2 stride, 5×5 kernel, outputs 36@14x47 	|
| RELU					|	Activation layer			|
| Dropout					|	0.2	dropout 				|
| Convolution2D 5x5     	| 2×2 stride, 5×5 kernel, outputs 24@5x22 	|
| RELU					|	Activation layer			|
| Dropout					|	0.2	dropout 				|
| Convolution2D 3x3     	| 3×3 kernel, outputs 64@3x20 	|
| RELU					|	Activation layer			|
| Dropout					|	0.2	dropout 				|
| Convolution2D 3x3     	| 3×3 kernel, outputs 64@1x18 	|
| RELU					|	Activation layer			|
| Dropout					|	0.2	dropout 				|
| Flatten					|	 				|
| Fully connected	layer	| input 1152, output 100.        				|
| Dropout					|	0.2	dropout 				|
| Fully connected	layer	| input 100, output 50.        				|
| Dropout					|	0.2	dropout 				|
| Fully connected	layer	| input 50, output 10.        				|
| Dropout					|	0.2	dropout 				|
| Fully connected	layer	| input 10, output 1.        				|

Images are cropped and equalized before fed into the model, also data is normalized in the model using a Keras lambda layer (code line 18). 
After each convolution layer a RELU layer is included to introduce nonlinearity, also Dropout layer is introduced to prevent overfitting.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers after each layer to reduce overfitting (model.py lines 21). To ensure that model was not overfitting datataset is split in train dataset and validation dataset. Also in the next picture we can see a graph which represents loss vs epochs for train and validation datasets.

![alt text][image1]

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.
Without this dropout layers the model tends to overfit, also L2 regularization layers was tested but best resuts were obtained with dropout. After Convolution layers 0.2 dropout is used, and after "Dense" layer is used a Dropout value of 0.1. These values wer obtained experimentally

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from sides of the road, driving in counter sense. All in both tracks driving at speed similar to test speed.

For details about how I created the training data, see the next section. 

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
