# Self Driving Car Engineer Nanodegree Project 2
## Traffic Sign Recognition
The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./downloadedgermansigns/50SpeedLimit.jpg "50 Speed Limit"
[image2]: ./downloadedgermansigns/GeneralCaution.jpg "General Caution"
[image3]: ./downloadedgermansigns/TurnRightAhead.jpg "Turn Right Ahead"
[image4]: ./downloadedgermansigns/BumpyRoad.jpg "Bumpy Road"
[image5]: ./downloadedgermansigns/RightOfWayEnds.jpg "Right Of Way Ends"
[image6]: ./downloadedgermansigns/StopSign.jpg "Stop Sign"
[image7]: ./downloadedgermansigns/PriorityRoad.jpg "Priority Road"
[image8]: ./downloadedgermansigns/StopSign4.jpg "Stop Sign 4"
[image9]: ./visualizations/TrainingData.png "Training Data"
[image10]: ./visualizations/ValidationData.png "Validation Data"
[image11]: ./visualizations/TestData.png "Test Data"
[image12]: ./visualizations/ProcessedData.png "Processed Data"
[image13]: ./visualizations/AugmentedData.png "Augmented Data"
[image14]: ./visualizations/top5probsearly.png "Early Probabilities"
[image15]: ./visualizations/top5probs.png "Probabilities"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

Here is a link to my [project code](https://github.com/alangordon258/SelfDrivingCar-Term1-Proj2/blob/master/Traffic_Sign_Classifier.ipynb) . As you can see I achieved an accuracy of 97% on the test data set and 87.5% (7/8) on an additional 8 images that I found on the Internet.

## 1. Data Set Summary & Exploration
I used the numpy and pandas libraries to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43


## 2. An exploratory visualization of the dataset

Here is an exploratory visualization of the data set. The following bar chart is a histogram of the training data set showing how many images there are of each type.
![alt text][image9]
The following bar chart is a histogram of the validation data sset
![alt text][image10]
The following bar chart is a histogram of the test data sset
![alt text][image11]
The charts show that although some sign types have more representation in the data sets than others. The training, test, and validation data sets are similar.

## 3. Design and Test a Model Architecture

#### I preprocessed the images as follows: 
1. Convert to grayscale. Accuracy was a 2 or 3 percent higher with grayscale and ran faster also because there is only a single channel. I assume the acuracy was higher because shape and other factors were more distinct than color between the sign types and eliminating color allowed the network to focus on these other factors. 
2. Improve the contrast of the images using the OpenCV equalizeHist method.
3. Normalized the images using (x-avg)/stddev.
Example before and after images are shown below.

![alt text][image12]

After reading a few research papers from people who had worked on the same problem, such as [this](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), it was clear that one of the techniques for improving accuracy was to increase the size of the training set by perturbing the available samples with small, random changes of translation, rotation and scale. The Keras library includes a method called ImageDataGenerator that can be used for this purpose. I called the method with the following parameters. These values were arrive at largely by trial and error. 

datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        dim_ordering='tf',
        fill_mode='nearest')

Here is an example of an original image and an augmented image:

![alt text][image13]

This data augmentation was used to double the size of the training set from 34799 to 69598


### Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x64   |
| RELU					| 	        									|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 					|
| Fully connected		| input 1600  output 120 			        	|
| RELU 					| 	        									|
| Dropout				| 0.6        									|
| Fully connected		| input 120  output 84 			        		|
| RELU 					| 	        									|	
| Dropout				| 0.6        									|
| Fully connected		| input 84  output 43	 			        	|
| Softmax				| Output		       							|
															

### Model Training Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam Optimzer in Tensor Flow with a learning rate of 0.0005. My batch size was 128 and I used 80 epochs. I largely arrived at these values via an iterative approach. I eventually achieved an accuracy of 0.969. I essentially stuck with the basic Lenet architecture. I played around a little bit with an architecture where outputs from earlier, convolutional layers were fed forward into the fully-connected layers. But I was not able to see an imrpovement with this approach. 

Initially, I was seeing an accuracy of approx 90% with the Lenet architecture as it was created in our lectures. Early on I did see that increasing the number of features in the convolutional layer had a significant positive effect. This was particularly true before I converted the images to grayscale. This makes sense as the additional features are needed to capture the additional complexity of the traffic sign images relative to the character dataset that was used previously. After increasing the number of weights, augmenting the dataset and adding dropout had the next largest effects. Again, this makes sense because as I added additional weights I started to see a problem with overfitting begin to manifest itself. The training accuracy was high, but the validation numbers were coming in lower. When I added the dropout this improved. I tried a view runs with different values for the dropout starting at 0.5. 0.6 improved the final validation accuracy and 0.7 was better than 0.5 but not as good as 0.6, so I settled on 0.6. I next started varying the learning rate and number of epochs. I found that I reached a point of diminishing returns around 100 epoches and the best learning rate seemed to be between .0004 and .0005. I eventually settled on a value of 0.0005 for the learning rate with 80 epochs. This yielded an accuracy on the test set of 96.9%. I also tried both increasing and decreasing the batch size to 64 and 256 from 128, but neither offered an improvement.


My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 98.8% 
* test set accuracy of 96.9%

A lot of research has already been done on this data set with some researchers achieving accuracies above 99% which actually exceeds human recognition accuracy of 98.81%. See [here](http://benchmark.ini.rub.de/?section=gtsrb&subsection=results#3). Yann Lec=Cun himself achieved accuracy of 99.17% as part of the GTSRB competition [here](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). Hence I did not see any reason to change the basic architecture as it had been shown to be very accurate at solving this particular problem. The only thing I felt was necessary was to do  some tuning of the number of weights in the convolution layer, which I found from experimentation to have a strong effect and tuning some of the other parameters such as the learning rate and keep probability for the dropout. I did notice that in manner of the architectures used by the researchers who got the best results, the output of the 1st convolutional layer is fed directly to the classifier (the fully-connected layers) as higher-resolution features. I decided to leave an investigation of this until the end. Unfortunately, I did not have enough time to explore this architecture in depth, but this will be a subject of future investigation for me. The results as to the efficacy of this architecture speak for themselves. I achieved 97% accuracy on the test data and 100% images that were pulled off the Internet.

 

## 4. Test a Model on New Images
I searched for and found eight images of German traffic signs on the web. The eight images are shown below.

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]


## 5. Predictions on New Images
The model was able to correctly guess 8 of 8 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set.

Here are the initial results of the prediction:

| Image			        	|     Prediction	        					| 
|:-------------------------:|:---------------------------------------------:| 
| Stop Sign      			| Stop sign 	 								| 
| Stop Sign   				| Stop Sign 									|
| Bumpy road   				| Bumpy road 									|
| Right of way intersection	| Right of way intersection						|
| Speed limit 50 km/h		| Speed limit 50 km/h  			 				|
| Turn right ahead			| Turn right ahead      						|
| Priority road 			| Priority Road  	     						|
| General caution			| General General       						|

You may wonder why there are two stop sign images in my data. This is because at first, one of the stop signs was being recognized incorrectly as Priority Road. I looked at the stop sign image and noticed that it was shot at an angle that created a keystoneing effect that altered the perceived shape of the image. See below.

![alt text][image6]

I suspect that the neural network has learned the shapes of the signs and in this image because of the distortion, it does not match the shape of a stop sign. To confirm this I decided to find a second stop sign image that didn't have this. This image is shown below.

![alt text][image8]

When I added this image to the dataset is was recognized  immediately. However, as I continued to train my neural network and make improvements, particularly when I increased the number of epochs from 40 to 80, I noticed that my classifier actually started to recognize the first stop sign correctly albeit with a lower confidence than the other images.

## 6. Describe how certain the model is when predicting on each of the new images.
My model was very certain of its results with the exception of the stop sign described above. The Softmax probabilities for all of the images with the exception of the stop sign image described above was very near 1.0 (about 0.99) for the selected sign type with all of the other probabilities near zero. See the image below.
![alt text][image15]
For the problematic stop sign image, the top five soft max probabilities were as follows:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .73         			| Stop sign   									| 
| .24     				| Priority Road									|
| .02					| Speed Limit 100 km/h							|
| .01	      			| Speed Limit 120 km/h					 		|
| .00				    | Keep Right      								|

The HTML output from my Ipython notebook showing my results can be found [here](https://github.com/alangordon258/SelfDrivingCar-Term1-Proj2/blob/master/Traffic_Sign_Classifier.html). When I tried viewing it on github.com it was complaining that it was too large, I can email this file if necessary. My code can be found [here](https://github.com/alangordon258/SelfDrivingCar-Term1-Proj2/blob/master/Traffic_Sign_Classifier.ipynb) This was a fun and challenging assignment. I wish I had more time to spend on it.




