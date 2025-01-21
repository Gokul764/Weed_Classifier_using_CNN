
import os
base_dir = './weedandcrops'
os.makedirs(base_dir, exist_ok=True)


train_dir = os.path.join(base_dir, 'train')
os.makedirs(train_dir, exist_ok=True)

validation_dir = os.path.join(base_dir, 'validation')
os.makedirs(validation_dir, exist_ok=True)

test_dir = os.path.join(base_dir, 'test')
os.makedirs(test_dir, exist_ok=True)

train_grass_dir = os.path.join(train_dir, 'grass')
os.makedirs(train_grass_dir, exist_ok=True)

train_soil_dir = os.path.join(train_dir, 'soil')
os.makedirs(train_soil_dir, exist_ok=True)

train_soybean_dir = os.path.join(train_dir, 'soybean')
os.makedirs(train_soybean_dir, exist_ok=True)

train_weed_dir = os.path.join(train_dir, 'weed')
os.makedirs(train_weed_dir, exist_ok=True)

validation_grass_dir = os.path.join(validation_dir, 'grass')
os.makedirs(validation_grass_dir, exist_ok=True)

validation_soil_dir = os.path.join(validation_dir, 'soil')
os.makedirs(validation_soil_dir, exist_ok=True)

validation_soybean_dir = os.path.join(validation_dir, 'soybean')
os.makedirs(validation_soybean_dir, exist_ok=True)

validation_weed_dir = os.path.join(validation_dir, 'weed')
os.makedirs(validation_weed_dir, exist_ok=True)

test_grass_dir = os.path.join(test_dir, 'grass')
os.makedirs(test_grass_dir, exist_ok=True)

test_soil_dir = os.path.join(test_dir, 'soil')
os.makedirs(test_soil_dir, exist_ok=True)

test_soybean_dir = os.path.join(test_dir, 'soybean')
os.makedirs(test_soybean_dir, exist_ok=True)

test_weed_dir = os.path.join(test_dir, 'weed')
os.makedirs(test_weed_dir, exist_ok=True)




import shutil


original_dataset_dir_soil = './dataset/soil'
original_dataset_dir_soybean = './dataset/soybean'
original_dataset_dir_grass = './dataset/grass'
original_dataset_dir_weed = './dataset/broadleaf'


fnames = ['{}.tif'.format(i) for i in range(1,1950)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_soil, fname)
    dst = os.path.join(train_soil_dir, fname)
    shutil.copyfile(src, dst)


fnames = ['{}.tif'.format(i) for i in range(1950, 2600)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_soil, fname)
    dst = os.path.join(validation_soil_dir, fname)
    shutil.copyfile(src, dst)
    

fnames = ['{}.tif'.format(i) for i in range(2600, 3250)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_soil, fname)
    dst = os.path.join(test_soil_dir, fname)
    shutil.copyfile(src, dst)
    

fnames = ['{}.tif'.format(i) for i in range(1,4427)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_soybean, fname)
    dst = os.path.join(train_soybean_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.tif'.format(i) for i in range(4427, 5902)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_soybean, fname)
    dst = os.path.join(validation_soybean_dir, fname)
    shutil.copyfile(src, dst)
    

fnames = ['{}.tif'.format(i) for i in range(5902, 7377)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_soybean, fname)
    dst = os.path.join(test_soybean_dir, fname)
    shutil.copyfile(src, dst)
    

fnames = ['{}.tif'.format(i) for i in range(1,2113)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_grass, fname)
    dst = os.path.join(train_grass_dir, fname)
    shutil.copyfile(src, dst)


fnames = ['{}.tif'.format(i) for i in range(2113, 2817)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_grass, fname)
    dst = os.path.join(validation_grass_dir, fname)
    shutil.copyfile(src, dst)
    

fnames = ['{}.tif'.format(i) for i in range(2817, 3521)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_grass, fname)
    dst = os.path.join(test_grass_dir, fname)
    shutil.copyfile(src, dst)
    

fnames = ['{}.tif'.format(i) for i in range(1,716)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_weed, fname)
    dst = os.path.join(train_weed_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.tif'.format(i) for i in range(716, 954)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_weed, fname)
    dst = os.path.join(validation_weed_dir, fname)
    shutil.copyfile(src, dst)
    

fnames = ['{}.tif'.format(i) for i in range(954, 1192)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_weed, fname)
    dst = os.path.join(test_weed_dir, fname)
    shutil.copyfile(src, dst)
    

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150,150,3))) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu')) 
model.add(layers.Dense(4, activation='softmax')) 
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])


from keras.preprocessing import image


from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)


test_datagen = ImageDataGenerator(rescale=1./255)




train_generator = train_datagen.flow_from_directory(

        train_dir,

        target_size=(150, 150),
        color_mode="rgb",
       
        batch_size=92,
      
        class_mode='categorical')


validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        color_mode="rgb",
        batch_size=31,
        class_mode='categorical')


test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        color_mode="rgb",
        batch_size=31,
        class_mode='categorical')



import math
training_samples =9202
batch_size_training_generator=92
validation_samples =3067
batch_size_validation_generator=31

history = model.fit_generator(
      train_generator,
     
      steps_per_epoch=math.ceil(training_samples/batch_size_training_generator),
      epochs=15,
      validation_data=validation_generator,
      validation_steps=math.ceil(validation_samples/batch_size_validation_generator))


model.save('model_weedcrops.h5')