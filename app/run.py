# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import pathlib
import mlflow
import os
import matplotlib.pyplot as plt

data_dir = pathlib.Path('Data/')

image_count = len(list(data_dir.glob('*/*.jpg')))

batch_size = 10
img_height = 180
img_width = 180
epochs_to_run = 3

mlflow.set_tracking_uri('https://dagshub.com/jsprovoke/PigANoteOnDAGS.mlflow')
os.environ['MLFLOW_TRACKING_USERNAME']='jsprovoke' 
os.environ['MLFLOW_TRACKING_PASSWORD']='2cf70b9982f050dcf6dd9dc04731a1d39fa48988'

mlflow.start_run()

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

mlflow.log_param("Image_Height", img_height)
mlflow.log_param("Image_Width", img_width)
mlflow.log_param("Batch Size", batch_size)
mlflow.log_param("Epochs", epochs_to_run)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

num_classes = 3

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

hist = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)
mlflow.log_metrics({"loss": hist.history["loss"][-1],
                   "accuracy":hist.history["accuracy"][-1],
                   "val_loss":hist.history["val_loss"][-1],
                   "val_accuracy":hist.history["val_accuracy"][-1]})

acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = range(1, len(acc) + 1)

fig1 = plt.figure()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

fig1.savefig('junk.jpg', bbox_inches='tight', dpi=150)
mlflow.log_figure(fig1, "figure1.jpg")

fig2 = plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

fig2.savefig('line plot.jpg', bbox_inches='tight', dpi=150)
mlflow.log_figure(fig2,"figure2.jpg")

plt.show()

mlflow.end_run()