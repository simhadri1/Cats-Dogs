
# use below command if using on colab
# if working on local computer, go to the below url to download
# data and park it in the folder that you are working

# !wget --no-check-certificate https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip


import os
import zipfile

pwd()

local_zip = './cats_and_dogs_filtered.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall()
zip_ref.close()

base_dir = './cats_and_dogs_filtered'

train_dir =os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')


#directory with training pictures

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

#directory with validation picutres

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# to check the file names on how they are assigned

train_cat_fnames = os.listdir(train_cats_dir)
train_dog_fnames = os.listdir(train_dogs_dir)

print(train_cat_fnames[:10])
print(train_dog_fnames[:10])

# to find out the total number of cat and dog images in train and validation directories

print ( 'Total training cat images :', len(os.listdir( train_cats_dir)))
print ( 'Total training dog images :', len(os.listdir( train_dogs_dir)))

print ( 'Total validation cat images :', len(os.listdir( validation_cats_dir)))
print ( 'Total validation dog images :', len(os.listdir( validation_dogs_dir)))


%matplotlib inline

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

pic_index = 0 # Index for iterating over images

fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)

pic_index+=8

next_cat_pix = [os.path.join(train_cats_dir, fname)
                for fname in train_cat_fnames[ pic_index-8:pic_index]
               ]

next_dog_pix = [os.path.join(train_dogs_dir, fname)
                for fname in train_dog_fnames[ pic_index-8:pic_index]
               ]

for i, img_path in enumerate(next_cat_pix+next_dog_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

import tensorflow as tf


model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16,(3,3), activation ='relu' , input_shape=(150,150,3)),
    tf.keras.layers.MaxPool2D(2,2) ,
    tf.keras.layers.Conv2D(32,(3,3), activation ='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'),
    tf.keras.layers.MaxPool2D(2,2),

    # flatten the resutls to feed to a DNN

    tf.keras.layers.Flatten(),
    # dense layer with 512 neurons
    tf.keras.layers.Dense(512, activation='relu'),
    # output only 1 neuran , with values between 0 to 1, 0 = cats. 1 = dogs
    tf.keras.layers.Dense(1,activation ='sigmoid')

    ])

model.summary()

from tensorflow.keras.optimizers import RMSprop

model.compile (optimizer=RMSprop(learning_rate=0.001),
               loss='binary_crossentropy',
               metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# reslcaling images all pixels range from 0 to 255  , we would normalize

train_datagen = ImageDataGenerator( rescale = 1.0/255.0)
valid_datagen = ImageDataGenerator( rescale = 1.0/255.0)

#flow training images

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150,150))

#flow validation images

validation_generator = valid_datagen.flow_from_directory(validation_dir,
                                                          batch_size = 20,
                                                          class_mode= 'binary',
                                                          target_size = (150,150) )


#fitting model

history = model.fit(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=100,
                    epochs =10,
                    validation_steps=50,
                    verbose=2)