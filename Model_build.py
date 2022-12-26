import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, GlobalMaxPooling2D, Dropout, MaxPooling2D
 
#- setting file path
no_mask_data_dir = './data/no_mask/'
mask_data_dir = './data/mask/'
 
no_mask_file = os.listdir(no_mask_data_dir)
mask_file = os.listdir(mask_data_dir)
 
file_num = len(no_mask_file) + len(mask_file) 
 
#- image preprocessing
num = 0
all_img = np.float32(np.zeros((file_num, 224, 224, 3))) 
all_label = np.float64(np.zeros((file_num, 1)))

#- no_mask
for img_name in no_mask_file:
    img_path = no_mask_data_dir+img_name
    img = load_img(img_path, target_size=(224, 224))
    
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    all_img[num, :, :, :] = x
    
    all_label[num] = 0
    num = num + 1

#- mask
for img_name in mask_file:
    img_path = mask_data_dir+img_name
    img = load_img(img_path, target_size=(224, 224))
    
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    all_img[num, :, :, :] = x
    
    all_label[num] = 1 # mask
    num = num + 1
 
 
#- dataset shuffle
n_elem = all_label.shape[0]
indices = np.random.choice(n_elem, size=n_elem, replace=False)
 
all_label = all_label[indices]
all_img = all_img[indices]
 
 
#- slpit dataset
num_train = int(np.round(all_label.shape[0]*0.8))
num_test = int(np.round(all_label.shape[0]*0.2))
 
train_img = all_img[0:num_train, :, :, :]
test_img = all_img[num_train:, :, :, :] 
 
train_label = all_label[0:num_train]
test_label = all_label[num_train:]
 

#- model build
img_input = (224, 224, 3)

model = Sequential()
model.add(Conv2D(16, (4,4), activation='relu', input_shape=img_input)) 
model.add(MaxPooling2D(2,2))
model.add(Conv2D(32, (4,4), activation='relu')) 
model.add(MaxPooling2D(2,2))
model.add(Conv2D(32, (4,4), activation='relu')) 
model.add(MaxPooling2D(2,2))
#model.add(GlobalMaxPooling2D())
#model.add(Dropout(0.3))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(512, activation = 'relu'))
model.add(Dense(1, activation='sigmoid'))


opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#- train model 
model.fit(train_img, train_label, epochs=10, batch_size=32, validation_split = 0.2)
 
 
#- save model
model.save("mask_wearing_detection_model.h5")