import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.models import Model, load_model
from keras.layers import Input
from keras.preprocessing.image import load_img
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from Architechture import build_model
from IOU import iou_metric
from Utils import DataAugment, RLE

def cov_to_class(val):    
    for i in range(0, 11):
        if val * 10 <= i :
            return i

img_size_ori = 101
img_size_target = 101
im_width = 101
im_height = 101
im_chan = 1

train_df = pd.read_csv("train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]
train_df = train_df[:50]
test_df = test_df[:10]

len(train_df)

train_df["images"] = [np.array(load_img("train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in train_df.index]
train_df["masks"] = [np.array(load_img("train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in train_df.index]

train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
train_df["coverage_class"] = train_df.coverage.map(cov_to_class)

ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_valid, depth_train, depth_valid = train_test_split(
                                        train_df.index.values,
                                        np.array(train_df.images.tolist()).reshape(-1, img_size_target, img_size_target, 1), 
                                        np.array(train_df.masks.tolist()).reshape(-1, img_size_target, img_size_target, 1), 
                                        train_df.coverage.values,
                                        train_df.z.values,
                                        test_size=0.2,
                                        # stratify=train_df.coverage_class, 
                                        random_state= 1234)

#Data augmentation
x_train, y_train = DataAugment(x_train, y_train)
print(x_train.shape)
print(y_valid.shape)

input_layer = Input((img_size_target, img_size_target, 1))
output_layer = build_model(input_layer, 16,0.5)

model = Model(input_layer, output_layer)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

early_stopping = EarlyStopping(monitor='val_acc', mode='max', patience=20, verbose=1)
model_checkpoint = ModelCheckpoint("./unet_best1.model", monitor='val_acc', mode='max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', mode='max', factor=0.2, patience=5, min_lr=0.00001, verbose=1)
#reduce_lr = ReduceLROnPlateau(factor=0.2, patience=5, min_lr=0.00001, verbose=1)

epochs = 10
batch_size = 8

history = model.fit(x_train, y_train,
                    validation_data=[x_valid, y_valid], 
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping, model_checkpoint, reduce_lr], 
                    verbose=1)

plt.plot(history.history['acc'][1:])
plt.plot(history.history['val_acc'][1:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','Validation'], loc='upper left')
plt.show()

model = load_model("./unet_best1.model")

def predict_result(model, x_test, img_size_target): # predict both orginal and reflect x
    preds_test = model.predict(x_test).reshape(-1, img_size_target, img_size_target)
    preds_test += np.array([ np.fliplr(a) for a in model.predict(np.array([np.fliplr(x) for x in x_test])).reshape(-1, img_size_target, img_size_target)])
    return preds_test/2.0

preds_valid = predict_result(model,x_valid,img_size_target)

def filter_image(img):
    if img.sum() < 100:
        return np.zeros(img.shape)
    else:
        return img

## Scoring for last model
thresholds = np.linspace(0.3, 0.7, 31)
ious = np.array([iou_metric(y_valid.reshape((-1, img_size_target, img_size_target)), [filter_image(img) for img in preds_valid > threshold]) for threshold in thresholds])

threshold_best_index = np.argmax(ious) 
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]

plt.plot(thresholds, ious)
plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
plt.xlabel("Threshold")
plt.ylabel("IoU")
plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
plt.legend()

import gc

del x_train, x_valid, y_train, y_valid, preds_valid
gc.collect()

x_test = np.array([(np.array(load_img("../input/test/images/{}.png".format(idx), grayscale = True))) / 255 for idx in test_df.index]).reshape(-1, img_size_target, img_size_target, 1)

preds_test = predict_result(model,x_test,img_size_target)

pred_dict = {idx: RLE(filter_image(preds_test[i] > threshold_best)) for i, idx in enumerate(test_df.index.values)}

sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('submission.csv')





