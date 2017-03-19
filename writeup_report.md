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
[image8]: ./write_up_images/dataset1.png ""
[image9]: ./write_up_images/dataset2.png "Image"
[image10]: ./write_up_images/dataset3.png "Image"

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

Different images sets were used to train the model during the set-up process. 
First data acquired manually was used to train the model and data from udacity was used for validation. Later when data of track 2 was acquired, all data was split into train dataset and validation dataset because udacity data doesn´t contain data of track 2 so is not a good validation data for both tracks.

Lot of driving test were made to check how well the car was driving around both tracks. There were a few spots where the vehicle fell off the track to improve the driving behavior in these cases data of these areas were collected.

At the end of the process, the vehicle is able to drive autonomously around both tracks without leaving the road. Track one is completed at 30mph and track two at 20mph.

My model consists of a convolution neural network inspired in Nvidia neural network [Nvidia NN](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the next table al the layers are described:

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

Images are cropped and equalized before fed into the model, also data is normalized in the model using a Keras lambda layer (code line !!). 
After each convolution layer a RELU layer is included to introduce nonlinearity, also Dropout layer is introduced to prevent overfitting.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers after each layer to reduce overfitting (model.py lines 21). To ensure that model was not overfitting datataset is split in train dataset and validation dataset. Also in the next picture we can see a graph which represents loss vs epochs for train and validation datasets.

![alt text][image1]

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.
Without this dropout layers the model tends to overfit, also L2 regularization layers was tested but best resuts were obtained with dropout. After Convolution layers 0.2 dropout is used, and after "Dense" layer is used a Dropout value of 0.1. These values wer obtained experimentally

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25). But all parameters were tune experimentally like augmentation values (displacement, brightnes), number of images to feed the model, number of images per batch, epochs, dropout values.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from sides of the road and driving in counter sense.
Driving data was acquired in both tracks driving at speed similar to the desired speed in the test.

Data was preprocesed in generator to avoid use big amout of memory. So names of filenames with their measurements and aditional information is stored during the process of obtain a balanced dataset (code line !!). This info will be provided to the generator to give images to the model during the train process.

To augment data from laps, we use also side cameras adding 0.1 for left images and substracting for right images. Also flipping these images and changing the sign of its measuremet we obtain three images more. So we have 6 images per each position of the car during the learning process.

As I told before, one of the keys of this project is having a balanced dataset. When we drive the car we tend to stay in th center of the road so our dataset will have a lot of images with measurements near to 0. So we use images with measurement  higher than 0.001 and lower than -0.001. After this we have the next dataset:
![alt text][image8]

As we can see our dataset after removing center images is not balanced still. At this point two strategies were tested, one augmentate data to have balanced dataset and remove some center images and augmentate data to have a balanced dataset. In both we have a balanced dataset but in the second we have less images. Second solution was selected because these amount of images was enough to train the model and the trainning process was much faster.

In the next figure we can see the histogram after remove images when exceded 500 images per bin.
![alt text][image9]

Finally after augmentate data in bins which have less images we obtain the next histogram:
![alt text][image10]

To augment the dataset, we use three different methods, firs move images, second augmentate brightness and third add random shadows, this third method was inpired in this posts:
[Vivek Yadav](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.lnrrf0vcb)
-
[Jeremy Shannon](https://medium.com/udacity/udacity-self-driving-car-nanodegree-project-3-behavioral-cloning-446461b7c7f9#.mh6z0fpez)
.

First method consist in move image with a displacement. It can be vertical and horizontal. When horizontal displacement is added is necesary to compensate the measuremt. Experimentaly a value of 0.0001 per pixel was used. In the next images we can see an image displaced compared with the previous one. (code line !!)

![alt text][image6]
![alt text][image7]

Second method is commonly used in image augmentation and consists in vary the brightness of the image. We can see an image with different brightnes compared with the previous one. (code line !!)

![alt text][image4]
![alt text][image5]

This third method consists in add to the image random shadows these is also useful to help the model to generalize. We can see an image with different brightnes compared with the previous one. (code line !!)

![alt text][image2]
![alt text][image3]

This third methods are applied in each aditional image of image augmentation. Variation in each step is randomly selected.

After this process 49973 images are fed tho the neural network. Images are randomly shuffled, 80% for the data set and 20% to the validation set. 

In the trainning process 50 epochs were executed and 256 images per batch to feed the model. As we can see in the image of "model mean squared error loss", error loss in model with 50 epochs is stablished. Also in the videos atached we can see how car drives in both tracks.
