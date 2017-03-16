import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import gc

def n_completed(n_Count):
    completed = True
    for count in n_Count:
        if (count > 0):
            completed = False
            break
    return completed

def keep_prob(n,bins,steering):
    for i in range(len(n)):
        act_n = n[i]
        act_bin = bins[i+1]
        if steering <= act_bin:
            break
    keep_p = (np.amax(n)-abs(act_n))/(np.amax(n))
    return keep_p

def act_n_hist(n,bins,steering):
    for i in range(len(n)):
        act_n = n[i]
        act_bin = bins[i+1]
        if steering <= act_bin:
            break
    return act_n,i

def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    #image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image[:,:,2]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            #image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
            image[:,:,2][cond1] = image[:,:,2][cond1]*random_bright
        else:
            #image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright  
            image[:,:,2][cond0] = image[:,:,2][cond0]*random_bright    

    #image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image

def augment_image_brightness(image):
    image = np.array(image, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image[:,:,2] = image[:,:,2]*random_bright
    image[:,:,2][image[:,:,2]>255]  = 255
    image = np.array(image, dtype = np.uint8)
    return image


def move_image(image,measurement):
    trans_range = 10;
    tr_y = trans_range*np.random.uniform()-trans_range/2
    tr_x = trans_range*np.random.uniform()-trans_range/2 
    rows,cols,ch = image.shape
    M = np.float32([[1,0,tr_x],[0,1,tr_y]]) 
    image = cv2.warpAffine(image,M,(cols,rows))
    measurement = measurement + tr_x*0.001

    return image,measurement


#En todas ecualizar */shadow

#Brightness augmentation

#Desplazar
#for image in images:
#    keep_prob = abs(measurement)
#    if keep_prob > np.random.uniform():
        
import os
import csv

lines= []
first_line=True;

base_dir = os.path.dirname(os.path.abspath(__file__))
filename_train_log = base_dir + '/data2/driving_log.csv'
filename_valid_log = base_dir + '/data2/driving_log.csv'
filename_train_path = base_dir + '/data2/IMG/'
filename_valid_path = base_dir + '/data2/IMG/'


with open(filename_train_log) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if first_line:
            first_line = False
        else:
            lines.append(line)
print(len(lines))

file_names = []
measurements = []
correction = 0.1 # Estimate angle correction for side cameras

#print(len(images))

for line in lines:
    current_path = filename_train_path 
    filename_center = line[0].split('/')[-1]
    filename_left = line[1].split('/')[-1]
    filename_right = line[2].split('/')[-1]
    
    #img_center = cv2.cvtColor(img_center,cv2.COLOR_BGR2YUV)
    #img_center[:,:,0] = cv2.equalizeHist(img_center[:,:,0])
    #img_left = cv2.cvtColor(img_left,cv2.COLOR_BGR2YUV)
    #img_left[:,:,0] = cv2.equalizeHist(img_left[:,:,0])
    #img_right = cv2.cvtColor(img_right,cv2.COLOR_BGR2YUV)
    #img_right[:,:,0] = cv2.equalizeHist(img_right[:,:,0])
    
    #flip_img_center = cv2.flip(img_center,1)
    #flip_img_left = cv2.flip(img_left,1)
    #flip_img_right = cv2.flip(img_right,1)
    
    steering_center = float(line[3])   
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    #flip_steering_center = -steering_center   
    #flip_steering_left = -steering_left
    #flip_steering_right = -steering_right
    if ((steering_center>0.001)or(steering_center<-0.001)):
        file_names.append(filename_center)
        file_names.append(filename_left)
        file_names.append(filename_right)
        #images.append(flip_img_center)
        #images.append(flip_img_left)
        #images.append(flip_img_right)
        
        measurements.append(steering_center)
        measurements.append(steering_left)
        measurements.append(steering_right)
        #measurements.append(flip_steering_center)
        #measurements.append(flip_steering_left)
        #measurements.append(flip_steering_right) 

#images = np.array(images)
#measurements = np.array(measurements)


'''
plt.figure(figsize=(10,5))
plt.title("Distribution of images per class")
(n, bins, patches) = plt.hist(measurements, 100)
plt.show()
'''
n, bins = np.histogram(measurements, 100)

#for i in range(len(images)):
#    X = images[i]
#    y = measurements[i]
#    act_n = act_n_hist(n,bins,y)
#    if act_n > 0:
#        act_n = ((100*(1-abs(y))+50-act_n)/act_n)
#        act_n = math.floor(act_n)
#        if act_n > 0:
#            for j in range(abs(act_n)):   
#                if (np.random.uniform())>0.5:
#                    X = augment_image_brightness(X)
#                if (np.random.uniform())>0.5:
#                    X = add_random_shadow(X)
#                X,y = move_image(X,y)
#                images.append(X)    
#                measurements.append(y)  


#a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
#index = [2, 3, 6]

#new_a = np.delete(a, index)

#print(n,'n')
#print(bins,'bins')
                
#X_train = np.array(images)
#y_train = np.array(measurements)

#im = []
#me = []
#for i in range(len(images)):
#    X = images[i]
#    y = measurements[i]
#    prob = keep_prob(n,bins,y)
#    if prob > (np.random.uniform()+0.2):
#        im.append(X)
#        me.append(y)
        
#images = im
#measurements = me

#del im
#del me
#gc.collect()


### OP_0
aditional_file_names = []
aditional_measurements = []

n_Count = [max(n) for i in range(len(n))]
for i in range(len(n_Count)):
    if n[i]>0:
        n_Count[i] = n_Count[i] - n[i]
    else:
        n_Count[i] = 0

while not(n_completed(n_Count)): 
    for i in range(len(file_names)):
        f = file_names[i]
        y = measurements[i]
        act_n,ind = act_n_hist(n,bins,y)
        if n_Count[ind] > 1:           
            n_times = ((n_Count[ind]-act_n)/act_n)
            n_times = math.floor(n_times)+10
            for j in range(n_times): 
                aditional_file_names.append(f)    
                aditional_measurements.append(y)
            n_Count[ind] = n_Count[ind] - n_times
        else: 
            if n_Count[ind] > 0:
                n_Count[ind] = n_Count[ind] - 1
                aditional_file_names.append(f)    
                aditional_measurements.append(y)
    #print(n_Count)


        
####                
### OP_1
#aditional_file_names = []
#aditional_measurements = []

#for i in range(len(file_names)):
#    f = file_names[i]
#    y = measurements[i]
#    act_n = act_n_hist(n,bins,y)
#    if act_n > 0:
#        #act_n = ((50*(1-abs(y))+100-act_n)/act_n)
#        #act_n = (max(n)/act_n)-1
#        act_n = ((max(n)-150-act_n)/act_n)
#        act_n = math.floor(act_n)
#        #act_n = math.ceil(act_n)
#
#        if act_n > 0:
#            for j in range(abs(act_n)):   
#                aditional_file_names.append(f)    
#                aditional_measurements.append(y)  

####                
                
                
#for X, y in zip(X_train,y_train):
#    prob = keep_prob(n,bins,y)
#    if prob > 0.5:#np.random.uniform():
#        trans_range = 5;
#        tr_y = trans_range*np.random.uniform()-trans_range/2
#        rows,cols,ch = X.shape    
#        M = np.float32([[1,0,0],[0,1,tr_y]]) 
#        images.append(cv2.warpAffine(X,M,(cols,rows)))    
#        measurements.append(y) 


#print(n,'n')
#print(bins,'bins')
                
#X_train = np.array(images)
#y_train = np.array(measurements)

size_original = len(file_names)
print(size_original)
file_names, measurements = shuffle(file_names, measurements)
aditional_file_names, aditional_measurements = shuffle(aditional_file_names, aditional_measurements)

file_names = file_names + aditional_file_names
measurements = measurements + aditional_measurements

'''
plt.figure(figsize=(10,5))
plt.title("Distribution of images per class")
(n, bins, patches) = plt.hist(measurements, 100)
plt.show()
'''
n, bins = np.histogram(measurements, 100)

#images = np.array(images)

#from sklearn.model_selection import train_test_split
#train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#images, measurements = shuffle(images, measurements)
#X_train, X_valid, y_train, y_valid = train_test_split(images, measurements, test_size = 0.20, random_state = 100) 

#X_train, y_train = shuffle(images, measurements)

import cv2
import numpy as np
import sklearn

def generator_train(file_names, measurements, size_original, batch_size):
    current_path = filename_train_path
    num_samples = len(file_names)
    while 1: # Loop forever so the generator never terminates

        for offset in range(0, num_samples, batch_size):
            batch_file_names = file_names[offset:offset+batch_size]
            batch_measurements = measurements[offset:offset+batch_size]
            im = []
            me = []
            for filename,measurement in zip(batch_file_names,batch_measurements):
                image = cv2.imread(current_path + filename)
                #image[:,:,0] = cv2.equalizeHist(image[:,:,0])       
                image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
                                   
                if offset > size_original:
                    if (np.random.uniform())>0.5:
                        image = augment_image_brightness(image)
                    if (np.random.uniform())>0.6:
                        image = add_random_shadow(image)
                    #image,measurement = move_image(image,measurement)              
                #else:
                    #if (np.random.uniform())>0.8:
                        #image = augment_image_brightness(image)
                    #if (np.random.uniform())>0.8:
                        #image = add_random_shadow(image)    
                image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
                image = cv2.cvtColor(image,cv2.COLOR_RGB2YUV)
                image[:,:,0] = cv2.equalizeHist(image[:,:,0])
                image = image[70:136,:,:]
                image = cv2.resize(image,(200, 66), interpolation = cv2.INTER_AREA)
                im.append(image)
                me.append(measurement)

            X = np.array(im)
            y = np.array(me)
            yield sklearn.utils.shuffle(X, y)

            
def generator_valid(batch_size):
    lines= []
    first_line=True;
    with open(filename_valid_log) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if first_line:
                first_line = False
            else:
                lines.append(line)

    current_path = filename_valid_path 
    num_samples = len(lines)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_lines = lines[offset:offset+batch_size]
            im = []
            me = [] 
            for line in batch_lines:
                filename = line[0].split('/')[-1]
                img_center = cv2.imread(current_path + filename)
                steering_center = float(line[3])   
                #img_center = cv2.cvtColor(img_center,cv2.COLOR_BGR2RGB)
                img_center = cv2.cvtColor(img_center,cv2.COLOR_BGR2YUV)
                img_center[:,:,0] = cv2.equalizeHist(img_center[:,:,0])
                img_center = img_center[70:136,:,:]
                img_center = cv2.resize(img_center,(200, 66), interpolation = cv2.INTER_AREA)
                im.append(img_center)   
                me.append(steering_center)

            X = np.array(im)
            y = np.array(me)

            yield sklearn.utils.shuffle(X, y)
        
# compile and train the model using the generator function

train_generator = generator_train(file_names, measurements, size_original, batch_size=512)
validation_generator = generator_valid(batch_size=512)


from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.regularizers import l2
#from keras.regularizers import l2, activity_l2


model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 0.5, input_shape=(66,200,3)))
#model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dropout(0.2))
model.add(Dense(1))

'''
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24, 5, 5,subsample=(2,2), W_regularizer = l2(.005),activation="relu"))
model.add(Convolution2D(36, 5, 5,subsample=(2,2), W_regularizer = l2(.005),activation="relu"))
model.add(Convolution2D(48, 5, 5,subsample=(2,2), W_regularizer = l2(.005),activation="relu"))
model.add(Convolution2D(64, 3, 3, W_regularizer = l2(.005),activation="relu"))
model.add(Convolution2D(64, 3, 3, W_regularizer = l2(.005),activation="relu"))
model.add(Flatten())
model.add(Dense(100, W_regularizer = l2(.005)))
model.add(Dropout(0.2))
model.add(Dense(50, W_regularizer = l2(.005)))
model.add(Dropout(0.2))
model.add(Dense(10, W_regularizer = l2(.005)))
model.add(Dropout(0.2))
model.add(Dense(1))
'''

model.compile(loss = 'mse', optimizer = 'adam')

# For a mean squared error regression problem
#model.compile(optimizer='rmsprop',loss='mse')

#model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 3)
#history_object = model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 3, verbose=1)

history_object = model.fit_generator(train_generator, samples_per_epoch=
            len(measurements), validation_data=validation_generator,
            nb_val_samples=8030, nb_epoch=5, verbose=1)
    #len(X_valid)
model.save('model.h5')
print('Finished correctly')


#history_object = model.fit_generator(train_generator, samples_per_epoch =
#    len(train_samples), validation_data = 
#    validation_generator,
#    nb_val_samples = len(validation_samples), 
#    nb_epoch=5, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()


