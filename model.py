import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import gc
import os

#Function to check if any count in an array is great than zero
def n_completed(n_Count):
    completed = True
    for count in n_Count:
        if (count > 0):
            completed = False
            break
    return completed

#Function which returns the number of images in a bin of a given measurement
def act_n_hist(n,bins,steering):
    for i in range(len(n)):
        act_n = n[i]
        act_bin = bins[i+1]
        if steering <= act_bin:
            break
    return act_n,i

#add random shadow to an image
def random_shadow(image):
    col_1 = 320*np.random.uniform()
    col_2 = 320*np.random.uniform()
    shadow = 50
    mask = 0*image[:,:,2]
    image = np.array(image, dtype = np.float64)

    if np.random.randint(2)==1:
        pts = np.array([[0,0],[0,160],[col_1,160],[col_2,0]], np.int32)
    else:
        pts = np.array([[320,0],[320,160],[col_1,160],[col_2,0]], np.int32)

    cv2.fillPoly(mask,[pts],(shadow))
    '''
    imp = np.array(image, dtype = np.uint8)
    imp = cv2.cvtColor(imp,cv2.COLOR_HSV2BGR)
    cv2.imwrite('imagePrev.png',imp)
    '''
    im_mask = 0*image
    im_mask[:,:,2] = mask
    image = image - im_mask
    image[:,:,2][image[:,:,2]<0]  = 0
    image = np.array(image, dtype = np.uint8)
    ''' 
    imp = np.array(image, dtype = np.uint8)
    imp = cv2.cvtColor(imp,cv2.COLOR_HSV2BGR)
    cv2.imwrite('imagePost.png',imp)
    '''
    return image

def change_brightness(image):
    image = np.array(image, dtype = np.float64)
    image[:,:,2] = image[:,:,2]*(0.25+1.5*np.random.uniform())
    image[:,:,2][image[:,:,2]>255]  = 255
    image = np.array(image, dtype = np.uint8)
    return image

def random_move(image,measurement):
    tr_y = 10*np.random.uniform()-10/2
    tr_x = 10*np.random.uniform()-10/2 
    rows,cols,ch = image.shape
    M = np.float32([[1,0,tr_x],[0,1,tr_y]]) 
    image = cv2.warpAffine(image,M,(cols,rows))
    measurement = measurement + tr_x*0.0001
    return image,measurement
        
lines= []
base_dir = os.path.dirname(os.path.abspath(__file__))
#filename_train_log = base_dir + '/data copia/driving_log.csv'
#ilename_train_path = base_dir + '/data copia/IMG/'
filename_train_log = base_dir + '/data/driving_log.csv'
filename_train_path = base_dir + '/data/IMG/'

with open(filename_train_log) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

file_names = [] # Array with filenames
measurements = [] # Array with measuremets
info = [] # Array with additional info of an image: 1 - Original / 2 - Original flipped / 3 - Original augmented / 4 - Original flipped augmented
correction = 0.1 # Estimate angle correction for side cameras

print(len(lines))

# Fill arrays file_names, measurements, and info with the information of images an its measurements.
for line in lines:
    current_path = filename_train_path 
    filename_center = line[0].split('/')[-1]
    filename_left = line[1].split('/')[-1]
    filename_right = line[2].split('/')[-1]
      
    steering_center = float(line[3])   
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    if ((steering_center>0.001)or(steering_center<-0.001)):
        file_names.append(filename_center)
        file_names.append(filename_left)
        file_names.append(filename_right)
        file_names.append(filename_center)
        file_names.append(filename_left)
        file_names.append(filename_right)
        
        info.append(1)
        info.append(1)
        info.append(1)
        info.append(2)
        info.append(2)
        info.append(2)
               
        measurements.append(steering_center)
        measurements.append(steering_left)
        measurements.append(steering_right)
        measurements.append(-steering_center)
        measurements.append(-steering_left)
        measurements.append(-steering_right)

# Show histogram of measurements
'''
plt.figure(figsize=(10,5))
plt.title("Distribution of images per class")
(n, bins, patches) = plt.hist(measurements, 100)
plt.show()
'''
n, bins = np.histogram(measurements, 100)


# Shuffle image before select data
file_names, measurements, info = shuffle(file_names, measurements, info)

file_names2 = []
measurements2 = []
info2 = [] # 1 - Original / 2 - Original flipped / 3 - Original augmented / 4 - Original flipped augmented

# Remove images to have same number of images in all the bins.
n_Count = [500 for i in range(len(n))]
for i in range(len(n_Count)):
    bin_i = bins[i]
    bin_s = bins[i+1]
    for fn,me,inf in zip(file_names,measurements,info):    
        if n_Count[i]>0:
            if ((me>bin_i)and(me<=bin_s)):
                file_names2.append(fn)        
                measurements2.append(me)        
                info2.append(inf)
                n_Count[i] = n_Count[i] - 1
        else:
            break

file_names = file_names2
measurements = measurements2
info = info2

# Show histogram of measurements
'''
plt.figure(figsize=(10,5))
plt.title("Distribution of images per class")
(n, bins, patches) = plt.hist(measurements, 100)
plt.show()
'''
n, bins = np.histogram(measurements, 100)

# Arrays with info of augmented images
aditional_file_names = []
aditional_measurements = []
aditional_info = [] # 1 - Original / 2 - Original flipped / 3 - Original augmented / 4 - Original flipped augmented


# Augment images
print(0.9*max(n))
n_Count = [0.9*max(n) for i in range(len(n))]
while not(n_completed(n_Count)): 
    for i in range(len(file_names)):
        f = file_names[i]
        y = measurements[i]
        inf = info[i]+2
        act_n,ind = act_n_hist(n,bins,y)
        if n_Count[ind] > 1:           
            n_times = ((n_Count[ind]-act_n)/act_n)
            n_times = math.floor(n_times)+5
            for j in range(n_times): 
                aditional_file_names.append(f)    
                aditional_measurements.append(y)
                aditional_info.append(inf)
            n_Count[ind] = n_Count[ind] - n_times
        else: 
            if n_Count[ind] > 0:
                n_Count[ind] = n_Count[ind] - 1
                aditional_file_names.append(f)    
                aditional_measurements.append(y)
                aditional_info.append(inf)


file_names = file_names + aditional_file_names
measurements = measurements + aditional_measurements
info = info + aditional_info

file_names2 = []
measurements2 = []
info2 = [] # 1 - Original / 2 - Original flipped / 3 - Original augmented / 4 - Original flipped augmented

# Remove images to have same number of images in all the bins.
n_Count = [500 for i in range(len(n))]
for i in range(len(n_Count)):
    bin_i = bins[i]
    bin_s = bins[i+1]
    for fn,me,inf in zip(file_names,measurements,info):    
        if n_Count[i]>0:
            if ((me>=bin_i)and(me<bin_s)):
                file_names2.append(fn)        
                measurements2.append(me)        
                info2.append(inf)
                n_Count[i] = n_Count[i] - 1
        else:
            break

file_names = file_names2
measurements = measurements2
info = info2


# Show histogram of measurements

'''
plt.figure(figsize=(10,5))
plt.title("Distribution of images per class")
(n, bins, patches) = plt.hist(measurements, 100)
plt.show()
'''
n, bins = np.histogram(measurements, 100)

from sklearn.model_selection import train_test_split
#Shufle images before split
file_names, measurements, info = shuffle(file_names, measurements, info)
#Split images in train an validation.
fnames_train, fnames_valid, meas_train, meas_valid, inf_train, inf_valid = train_test_split(file_names, measurements, info, test_size = 0.20, random_state = 100) 

import cv2
import numpy as np
import sklearn

# Train generator to feed the neural network
def generator_train(file_names, measurements, info, batch_size):
    current_path = filename_train_path
    num_samples = len(file_names)
    while 1: # Loop forever so the generator never terminates

        for offset in range(0, num_samples, batch_size):
            batch_file_names = file_names[offset:offset+batch_size]
            batch_measurements = measurements[offset:offset+batch_size]
            batch_info = info[offset:offset+batch_size]

            im = []
            me = []
            for filename,measurement,inf in zip(batch_file_names,batch_measurements,batch_info):
                image = cv2.imread(current_path + filename)
                image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
                
                if ((inf==2)or(inf==4)): # Flag 2 and 4 indicates flip image
                    image = cv2.flip(image,1)
               
                if ((inf==3)or(inf==4)): # Flag 3 and 4 indicates augmentate data
                    '''
                    if (np.random.uniform())>0.5:
                        image = change_brightness(image)
                    if (np.random.uniform())>0.6:
                        image = random_shadow(image)
                    '''
                    image = change_brightness(image)
                    image = random_shadow(image)

                image[:,:,2] = cv2.equalizeHist(image[:,:,2])

                if ((inf==3)or(inf==4)):
                    image,measurement = random_move(image,measurement)   
                    
                image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
                image = cv2.cvtColor(image,cv2.COLOR_RGB2YUV) # Convert images to YUV
                image = image[70:136,:,:] # Cropping images
                image = cv2.resize(image,(200, 66), interpolation = cv2.INTER_AREA) # Resize images
                im.append(image)
                me.append(measurement)

            X = np.array(im)
            y = np.array(me)
            yield sklearn.utils.shuffle(X, y)
            
# Valid generator to feed the neural network
def generator_valid(file_names, measurements, info, batch_size):
    current_path = filename_train_path
    num_samples = len(file_names)
    while 1: # Loop forever so the generator never terminates

        for offset in range(0, num_samples, batch_size):
            batch_file_names = file_names[offset:offset+batch_size]
            batch_measurements = measurements[offset:offset+batch_size]
            batch_info = info[offset:offset+batch_size]

            im = []
            me = []
            for filename,measurement,inf in zip(batch_file_names,batch_measurements,batch_info):
                image = cv2.imread(current_path + filename)
                image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
                
                if ((inf==2)or(inf==4)): # Flag 2 and 4 indicates flip image
                    image = cv2.flip(image,1)
               
                if ((inf==3)or(inf==4)): # Flag 3 and 4 indicates augmentate data
                    if (np.random.uniform())>0.5:
                        image = change_brightness(image)
                    if (np.random.uniform())>0.6:
                        image = random_shadow(image)
                
                image[:,:,2] = cv2.equalizeHist(image[:,:,2])
                '''
                if ((inf==3)or(inf==4)):
                    image,measurement = random_move(image,measurement)              
                '''
                
                image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
                image = cv2.cvtColor(image,cv2.COLOR_RGB2YUV) # Convert images to YUV
                image = image[70:136,:,:] # Cropping images
                image = cv2.resize(image,(200, 66), interpolation = cv2.INTER_AREA) # Resize images
                im.append(image)
                me.append(measurement)

            X = np.array(im)
            y = np.array(me)
            yield sklearn.utils.shuffle(X, y)
            
# compile and train the model using the generator function
train_generator = generator_train(file_names, measurements, info, batch_size=256)
validation_generator = generator_train(file_names, measurements, info, batch_size=256)

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.regularizers import l2

# Create model
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 0.5, input_shape=(66,200,3)))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.1))
model.add(Dense(50))
model.add(Dropout(0.1))
model.add(Dense(10))
model.add(Dropout(0.1))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')

# Train model
history_object = model.fit_generator(train_generator, samples_per_epoch=
            len(measurements), validation_data=validation_generator,
            nb_val_samples=8030, nb_epoch=50, verbose=1)

# Save model
model.save('model5.h5')
print('Finished correctly')

### print the keys contained in the history object
print(history_object.history.keys())
'''
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

'''
