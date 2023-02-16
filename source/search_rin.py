# %%
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='InceptionV3')
parser.add_argument('--weights', type=str, default='radimagenet')
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--n_neurons', type=int, default=512)
parser.add_argument('--n_dropout', type=float, default=0.2)
parser.add_argument('--lr_1', type=float, default=3e-4)
parser.add_argument('--lr_2', type=float, default=3e-6)
parser.add_argument('--image_size', type=int, default=512, required=False)
parser.add_argument('--batch_size', type=int, default=16, required=False)

args = parser.parse_args()

model_name = args.model_name
weights = args.weights
n_layers = args.n_layers
n_neurons = args.n_neurons
n_dropout = args.n_dropout
lr_1 = args.lr_1
lr_2 = args.lr_2
img_size = args.image_size
batch_size = args.batch_size

# %%
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.metrics import roc_curve, roc_auc_score, recall_score
from tensorflow.keras.applications import InceptionResNetV2, ResNet50, InceptionV3, DenseNet121, Xception
import tensorflow_hub as hub
import keras_cv


# %%
train_dir = '../data/split_1/train'
val_dir = '../data/split_1/val'
test_dir = '../data/split_1/test'


# %%
train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_dir, label_mode='binary', seed=0, image_size=(img_size, img_size), batch_size=batch_size, color_mode='rgb')
val_ds = tf.keras.preprocessing.image_dataset_from_directory(val_dir, label_mode='binary', seed=0, image_size=(img_size, img_size), batch_size=batch_size, color_mode='rgb')
test_ds = tf.keras.preprocessing.image_dataset_from_directory(test_dir, label_mode='binary', seed=0, image_size=(img_size, img_size), batch_size=1, color_mode='rgb')


# %%
#Apply data augmentation
preprocessing_model = tf.keras.Sequential()
preprocessing_model.add(
    tf.keras.layers.experimental.preprocessing.RandomRotation(40))
preprocessing_model.add(
    tf.keras.layers.experimental.preprocessing.RandomTranslation(0.2, 0.2))
preprocessing_model.add(
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.2, 0.2))
preprocessing_model.add(
    tf.keras.layers.experimental.preprocessing.RandomFlip(mode="horizontal"))
preprocessing_model.add(
    tf.keras.layers.experimental.preprocessing.RandomFlip(mode="vertical"))
#add cutmix augmentation
preprocessing_model.add(keras_cv.augmentations.CutMix(batch_size=batch_size, img_size=img_size, n_classes=2, alpha=1.0, seed=0))
#add random cutout augmentation
preprocessing_model.add(keras_cv.augmentations.RandomCutout(img_size=img_size, n_holes=1, length=0.2, seed=0))


# %%
train_ds = train_ds.map(lambda images, labels:
                        (preprocessing_model(images), labels))


# %%
if model_name == 'InceptionResNetV2':
    preprocess_fx = tf.keras.applications.inception_resnet_v2.preprocess_input
    model_dir = "../RadImageNet/models/RadImageNet-IRV2_notop.h5"
    if weights == 'imagenet':
        base_model = InceptionResNetV2(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet', pooling='avg')
    elif weights == 'radimagenet':
        base_model = InceptionResNetV2(input_shape=(img_size, img_size, 3), include_top=False, weights=model_dir, pooling='avg')
elif model_name == 'ResNet50':
    preprocess_fx = tf.keras.applications.resnet50.preprocess_input
    model_dir = "../RadImageNet/models/RadImageNet-ResNet50_notop.h5"
    if weights == 'imagenet':
        base_model = ResNet50(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet', pooling='avg')
    elif weights == 'radimagenet':
        base_model = ResNet50(input_shape=(img_size, img_size, 3), include_top=False, weights=model_dir, pooling='avg')
elif model_name == 'InceptionV3':
    preprocess_fx = tf.keras.applications.inception_v3.preprocess_input
    model_dir = "../RadImageNet/models/RadImageNet-InceptionV3_notop.h5"
    if weights == 'imagenet':
        base_model = InceptionV3(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet', pooling='avg')
    elif weights == 'radimagenet':
        base_model = InceptionV3(input_shape=(img_size, img_size, 3), include_top=False, weights=model_dir, pooling='avg')
elif model_name == 'DenseNet121':
    preprocess_fx = tf.keras.applications.densenet.preprocess_input
    model_dir = "../RadImageNet/models/RadImageNet-DenseNet121_notop.h5"
    if weights == 'imagenet':
        base_model = DenseNet121(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet', pooling='avg')
    elif weights == 'radimagenet':
        base_model = DenseNet121(input_shape=(img_size, img_size, 3), include_top=False, weights=model_dir, pooling='avg')
elif model_name == 'Xception':
    preprocess_fx = tf.keras.applications.xception.preprocess_input
    if weights == 'imagenet':
        base_model = tf.keras.applications.Xception(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet', pooling='avg')
elif model_name == 'BiT':
    base_model = hub.KerasLayer("https://tfhub.dev/google/bit/m-r50x1/1", trainable=False)
    preprocess_fx = tf.keras.applications.resnet50.preprocess_input
    


# %%


inputs = keras.Input(shape=(img_size, img_size, 3))
y = preprocess_fx(inputs)
y = base_model(y, training=False)
for i in range(n_layers):
    y = keras.layers.Dense(n_neurons, activation='relu')(y)
    y = keras.layers.Dropout(n_dropout)(y)
outputs = keras.layers.Dense(1, activation='sigmoid')(y)
model = keras.Model(inputs, outputs)

# %%
early_stopping = keras.callbacks.EarlyStopping(patience=100, min_delta=1e-15, restore_best_weights=True)

# %%

model.compile(
    optimizer=keras.optimizers.Adam(lr_1), 
    loss=keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.AUC()],
)

epochs = 1000
model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=early_stopping, verbose=1)
print('phase 1 complete')
# %%
#unfreeze all layers and train at lower learning rate
base_model.trainable = True
model.compile(
    optimizer=keras.optimizers.Adam(lr_2), 
    loss=keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.AUC()],
)
model.fit(train_ds, epochs=1000, validation_data=val_ds, callbacks=early_stopping, verbose=1)
print('phase 2 complete')

#save model to h5
model.save('models/{}_{}_{}_{}.h5'.format(model_name, weights, n_layers, n_neurons))

# %%
#load the entire dataset and make predictions
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_dir, label_mode='binary', seed=0, shuffle=False, image_size=(img_size, img_size), batch_size=batch_size, color_mode='rgb')
val_ds = tf.keras.preprocessing.image_dataset_from_directory(val_dir, label_mode='binary', seed=0, shuffle=False, image_size=(img_size, img_size), batch_size=batch_size, color_mode='rgb')
test_ds = tf.keras.preprocessing.image_dataset_from_directory(test_dir, label_mode='binary', seed=0, shuffle=False, image_size=(img_size, img_size), batch_size=1, color_mode='rgb')

y_train_pred = np.array([])
y_train = np.array([])
for images, labels in train_ds:
    y_train = np.append(y_train, labels.numpy())
    y_train_pred = np.append(y_train_pred, model.predict(images))
train_file_paths = train_ds.file_paths

y_val_pred = np.array([])
y_val = np.array([])
for images, labels in val_ds:
    y_val = np.append(y_val, labels.numpy())
    y_val_pred = np.append(y_val_pred, model.predict(images))
val_file_paths = val_ds.file_paths

y_test_pred = np.array([])
y_test = np.array([])
for images, labels in test_ds:
    y_test = np.append(y_test, labels.numpy())
    y_test_pred = np.append(y_test_pred, model.predict(images))
test_file_paths = test_ds.file_paths

y_all_pred = np.append(y_train_pred, y_val_pred)
y_all_pred = np.append(y_all_pred, y_test_pred)
y_all = np.append(y_train, y_val)
y_all = np.append(y_all, y_test)
file_paths = np.append(train_file_paths, val_file_paths)
file_paths = np.append(file_paths, test_file_paths)
pred_df = pd.DataFrame({'file_path': file_paths, 'y_true': y_all, 'y_pred': y_all_pred})
pred_df.to_csv('preds/preds_{}_{}_{}_{}.csv'.format(model_name, weights, n_layers, n_neurons), index=False)

auroc_val = roc_auc_score(y_val, y_val_pred)
auroc_test = roc_auc_score(y_test, y_test_pred)

#save the results to a csv file
results_df = pd.DataFrame({'model': [model_name], 'weights': [weights], 'image_size': [img_size], 'batch_size': [batch_size], 'lr_1': [lr_1], 'lr_2': [lr_2], 'n_layers': [n_layers], 'n_neurons': [n_neurons], 'n_dropout': [n_dropout], 'auroc_val': auroc_val, 'auroc_test': auroc_test})
#if results file exists, append to it without removing the header, otherwise create it
try:
    results_df.to_csv('results/results_rin.csv', mode='a', header=False, index=False)
except:
    results_df.to_csv('results/results_rin.csv', mode='w', header=True, index=True)