import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

dataset = pd.read_csv('dataset/final_data.csv')
dataset = pd.get_dummies(dataset, columns=['label'])

train_dataset = dataset.sample(frac=0.8, random_state=8) #train = 80%,  random_state = any int value means every time when you run your program you will get the same output for train and test dataset, random_state is None by default which means every time when you run your program you will get different output because of splitting between train and test varies within 
test_dataset = dataset.drop(train_dataset.index) #remove train_dataset from dataframe to get test_dataset

train_labels = pd.DataFrame([train_dataset.pop(x) for x in ['label_Red', 'label_Green', 'label_Blue', 'label_Yellow', 'label_Orange', 'label_Pink', 'label_Purple', 'label_Brown', 'label_Grey', 'label_Black', 'label_White']]).T
test_labels = pd.DataFrame([test_dataset.pop(x) for x in ['label_Red', 'label_Green', 'label_Blue', 'label_Yellow', 'label_Orange', 'label_Pink', 'label_Purple', 'label_Brown', 'label_Grey', 'label_Black', 'label_White']]).T

train_dataset_tf=tf.convert_to_tensor(train_dataset)
train_labels_tf=tf.convert_to_tensor(train_labels)
test_dataset_tf=tf.convert_to_tensor(test_dataset)
test_labels_tf=tf.convert_to_tensor(test_labels)

train_dataset_tf = np.asarray(train_dataset_tf).astype('float32')
test_dataset_tf = np.asarray(test_dataset_tf).astype('float32')

model = keras.Sequential([
    layers.Dense(64, activation = "relu"),
    layers.Dense(64, activation = "relu"),
    layers.Dense(11, activation = "softmax"),
])

model.compile(optimizer="rmsprop",
             loss="categorical_crossentropy",
             metrics=["accuracy"])

history = model.fit(x=train_dataset_tf, y=train_labels_tf,
                    epochs=512,
                    batch_size=2048,
                    validation_data=(test_dataset_tf, test_labels_tf))

test_loss, test_acc = model.evaluate(test_dataset_tf, test_labels_tf)
print(f"Test accuracy: {test_acc:.3f}")

model.save("model/pickcolor.keras")