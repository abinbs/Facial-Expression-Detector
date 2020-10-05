import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import os

num_classes= 5
img_rows,img_cols= 48,48
batch_size= 8

train_data_dir= r'D:\Facial Recognition\images\train'  #since python takes forward slash r reverses the slash of the directory path
validation_data_dir= r'D:\Facial Recognition\images\validation'

train_datagen= ImageDataGenerator(      #images are generated for training
    rescale= 1./255, #reduce image size
    rotation_range= 30,
    shear_range= 0.3,
    zoom_range= 0.3,
    width_shift_range= 0.4,
    height_shift_range= 0.4,   #create a new image by shifting accordingly
    horizontal_flip= True,      #Mirror image
    vertical_flip= True
)

validation_datagen= ImageDataGenerator(
    rescale= 1./255
)

train_generator= train_datagen.flow_from_directory(       #Training the data
    train_data_dir,
    color= 'grayscale',
    target_size= (img_rows,img_cols),
    batch_size= batch_size,
    class_mode= 'categorical',
    shuffle= True       #Shuffling all the images for better training, so it would memorize all the category images
)

validation_generator= validation_datagen.flow_from_directory(       #Validating the data
    validation_data_dir,
    color= 'grayscale',
    target_size= (img_rows,img_cols),
    batch_size= batch_size,
    class_mode= 'categorical',
    shuffle= True  
)

#TODO- cnn model