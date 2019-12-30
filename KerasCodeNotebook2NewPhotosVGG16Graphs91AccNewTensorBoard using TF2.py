#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Tensorflow (TF) 2.0


# In[1]:


import tensorflow as tf


# In[2]:


tf.__version__


# In[3]:


# import all the packages that are needed


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import PIL
from PIL import Image
#import cv2
from os import listdir
import skimage
from skimage import transform
from skimage import data
from tensorflow.python.keras import layers
from tensorflow.python.keras import Model
from tensorflow.python.keras import models
from IPython.display import SVG
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


from time import time
from tensorflow.python.keras.callbacks import TensorBoard


# In[5]:


def image_flipper():
    normal_photo_base_path = r'C:\CDriveDataSet\AI4GenetXDataset\Normal'
    down_syndrome_photo_base_path = r'C:\CDriveDataSet\AI4GenetXDataset\Down'
    williams_syndrome_photo_base_path = r'C:\CDriveDataSet\AI4GenetXDataset\Williams'
    

    normal_filenames = []
    for f in listdir(normal_photo_base_path):
        normal_filenames.append(normal_photo_base_path + '\\' + f) # appends the path of the picture to a list
        
        if len(normal_filenames) <= 53:
            image = cv2.imread(normal_photo_base_path + '\\' + f)
            image = cv2.resize(image, (244,244))
            flipped_image = cv2.flip(image, 1)
        #    path = r'Z:\Share\DownSyndrome\4. All Photo\\Normal_Flipped\\' + f + '_flipped' + '.jpg' 
            cv2.imwrite(path, flipped_image)
            

    down_filenames = []
    for f in listdir(down_syndrome_photo_base_path):
        down_filenames.append(down_syndrome_photo_base_path + '\\' + f) # appends the path of the picture to a list
        
        if len(down_filenames) <= 2270:
            image = cv2.imread(down_syndrome_photo_base_path + '\\' + f)
        try:
            image = cv2.resize(image, (244,244))
        except:
            print(path)
        flipped_image = cv2.flip(image, 1)
        #path = r'Z:\\Share\DownSyndrome\\4. All Photo\\Down_Flipped\\' + f[:-4] + '_flipped' + '.jpg' 
        cv2.imwrite(path, flipped_image)
        
        

    williams_filenames = []
    for f in listdir(williams_syndrome_photo_base_path):
        if f == 'desktop.ini':
            pass
        else:
            williams_filenames.append(williams_syndrome_photo_base_path + '\\' + f) # appends the path of the picture to a list
        
        #path = r'Z:\\Share\\DownSyndrome\\4. All Photo\\Williams_Flipped\\' + f[:-4] + '_flipped' + '.jpg' 
        image = cv2.imread(williams_syndrome_photo_base_path + '\\' + f)
        try:
            image = cv2.resize(image, (244,244))
        except:
            print(path)
        flipped_image = cv2.flip(image, 1)
        
        cv2.imwrite(path, flipped_image)


# In[6]:


def CNN_Preprocess():
    '''
    This functions runs the preprocessing for the images (resizes and converts to a numpy array)
    '''
    
    normal_photo_base_path = r'C:\CDriveDataSet\AI4GenetXDataset\Normal'
    down_syndrome_photo_base_path = r'C:\CDriveDataSet\AI4GenetXDataset\Down'
    williams_syndrome_photo_base_path = r'C:\CDriveDataSet\AI4GenetXDataset\Williams'
    

    normal_filenames = []
    for f in listdir(normal_photo_base_path):
        if f == "Thumbs.db":
            pass
        else:
            normal_filenames.append(normal_photo_base_path + '\\' + f) # appends the path of the picture to a list
            

    down_filenames = []
    for f in listdir(down_syndrome_photo_base_path):
        if f == "Thumbs.db":
            pass
        else:
            down_filenames.append(down_syndrome_photo_base_path + '\\' + f) # appends the path of the picture to a list
        

    williams_filenames = []
    for f in listdir(williams_syndrome_photo_base_path):
        if f == 'desktop.ini' or f == "Thumbs.db":
            pass
        else:
            williams_filenames.append(williams_syndrome_photo_base_path + '\\' + f) # appends the path of the picture to a list


    image_filenames = normal_filenames + down_filenames + williams_filenames # combines all the image paths together

    normal_label = 0 # labels
    down_label = 1 # labels
    williams_label = 2 # labels

    global number_of_images_each
    number_of_images_each = {'total':0, 'normal':0, 'down':0, 'williams':0}
    number_of_images_each['total'] = len(image_filenames)
    number_of_images_each['normal'] = len(normal_filenames)
    number_of_images_each['down'] = len(down_filenames)
    number_of_images_each['williams'] = len(williams_filenames)
    
    global images
    images = []

    normal_labels = [normal_label] * len(normal_filenames)          # generates labels for the normal images
    down_labels = [down_label] * len(down_filenames)                # generates labels for the down images
    williams_labels = [williams_label] * len(williams_filenames)    # generates labels for the williams images

    labels = normal_labels + down_labels + williams_labels          # adds all the labels together into a single list
    labels = np.array(labels)                                       # converts the list into a numpy array
        
    counter = 1
    for image in image_filenames:
        image_photo = Image.open(image)                                      # reads the image into memory
        resized_image = image_photo.resize((244,244), PIL.Image.ANTIALIAS)   # resizes the image to 244 by 244
        resized_image.save(image,"PNG")                                      # saves the resized image

        images.append(skimage.data.load(image))                              # loads the image as a numpy array
        print('Preprocessing Images for CNN...on image ' + str(counter))     # prints this line for debugging
        
        if skimage.data.load(image).shape == (244, 244, 3):                  # prints this for debugging
            print('true')                                                    # prints this for debugging
        else:
            print('false')                                                   # prints this for debugging
            print(image)                                                     # prints this for debugging
            print(skimage.data.load(image).shape)                            # prints this for debugging

        counter = counter + 1

    images = np.stack(images, axis=0)                              # conbines all the numpy arrays into a single one
    
    #images.dump('X_newphotos.npy')                                # saves the images array
    #labels.dump('y_newphotos.npy')                                # saves the labels array
    
    return images, labels


# In[7]:


X, y = CNN_Preprocess()
#X = np.load('X_newphotos.npy')            # loads the saved images numpy array
#y = np.load('y_newphotos.npy')            # loads the saved labels numpy array


# In[8]:


plt.figure(figsize = (8,5))

x_graph = ['Down Syndrome', 'Normal', 'Williams Syndrome']
y_graph = [number_of_images_each['normal'], number_of_images_each['down'], number_of_images_each['williams']]
image_count_graph = sns.barplot(x_graph, y_graph, orient = 'v', palette = 'muted')
image_count_graph.set_title('Number of Images by Classification')
image_count_graph.set_xlabel('Classifications')
image_count_graph.set_ylabel('Number of Images')


# In[9]:


plt.figure(figsize = (8,5))

sns.barplot(x = ['Images'], y = [number_of_images_each['total']], color = '#D98B5F')
sns.barplot(x = ['Images'], y = [number_of_images_each['total']*.8], palette = 'Blues')

topbar = plt.Rectangle((0,0),1,1,fc="#D98B5F", edgecolor = 'none')
bottombar = plt.Rectangle((0,0),1,1,fc='#78AAC8',  edgecolor = 'none')
plt.legend([topbar, bottombar], ['Testing Images', 'Training Images'])
plt.title('Training/Testing Breakup of Total')
plt.ylabel('Number of Images')


# In[10]:


X


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) # splits the data into 75% training and 25% testing


# In[12]:


y_cat_train = tf.keras.utils.to_categorical(y_train, 3) # converts the y_train to categorical valules (one-hot encoding)
y_cat_test = tf.keras.utils.to_categorical(y_test, 3) # converts the y_test to categorical valules (one-hot encoding)


# In[13]:


X_train = X_train / X_train.max() # converts all the pixel values from between 0 and 255 to between 0 and 1
X_test = X_test / X_test.max() # converts all the pixel values from between 0 and 255 to between 0 and 1


# In[14]:


model = tf.keras.Sequential() # generates a model instance

model.add(layers.ZeroPadding2D((1,1),input_shape=(244, 244, 3)))
model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Convolution2D(128, (3, 3), activation='relu'))
model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Convolution2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Convolution2D(256, (3, 3), activation='relu'))
model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Convolution2D(256, (3, 3), activation='relu'))
model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Convolution2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))  # last classification layer

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

model.compile(loss = 'categorical_crossentropy', # functions
             optimizer = 'sgd',
             metrics = ['accuracy'])


# In[15]:


model.summary() # provides a summary of the model


# In[ ]:


results = model.fit(X_train, y_cat_train, epochs = 100) # trains the model to the training data for 100 epochs


# In[ ]:


model.metrics_names


# In[22]:


model.evaluate(X_test, y_cat_test) # tests the model and finds accuracy


# In[23]:


predictions = model.predict_classes(X_test) # predicts classes on the testing data


# In[24]:


print(classification_report(y_test, predictions)) # prints a classification report


# In[25]:


cm = confusion_matrix(y_test, predictions) # prints a confusion matrix
print(cm)


# In[26]:


df_cm = pd.DataFrame(cm, index = [i for i in ['Normal', 'Down', 'Williams']], 
                     columns = [i for i in ['Normal', 'Down', 'Williams']])
plt.figure(figsize = (8,5))
sns.set(font_scale=1)
sns.heatmap(df_cm, annot=True, annot_kws={"size": 14}, cmap = 'RdBu', linecolor = 'black', square = True, fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')


# In[27]:


sns.set(font_scale=1)
plt.figure(figsize = (8,5))
plt.plot(results.history['acc'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy') # displays a graph of accuracy vs epoch
plt.title('Accuracy vs. Epoch')


# In[28]:


plt.figure(figsize = (8,5))
plt.plot(results.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss') # displays a graph of loss vs epoch
plt.title('Loss vs. Epoch')


# In[29]:


plt.figure(figsize = (8,5))
plt.plot(results.history['acc'], color = 'blue')
plt.plot(results.history['loss'], color = 'red')
plt.title('Accuracy vs. Loss')
plt.xlabel('Epoch')
plt.ylabel('Value')


# In[30]:


model.save('Model.h5') # saves the model

