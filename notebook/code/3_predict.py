import keras.backend as K
from keras.models import load_model
import xml.etree.ElementTree as ET
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.optimizers import Adam
# from keras.utils import multi_gpu_model
import numpy as np
import tensorflow as tf
from datetime import datetime
import os
from settings import setting
import cv2
from keras.utils.np_utils import to_categorical
img_count = 0
############################################
import Util_V2 as U
if setting['model'] == "DecayByBatch":
    optimizer = Adam(lr=setting["lr"])
elif setting['model'] == "DecayByEpoch":
    optimizer = Adam(lr=setting["lr"])
if setting['loss'] == 'Loss_v2':
    lossFunction = U.Loss_v2
elif setting['loss'] == 'Loss_v3':
    lossFunction = U.Loss_v3
#############################################

def LOAD_IMAGE(path):
    image1 = load_img(path)
    train_example = img_to_array(image1, data_format='channels_first')
    img = array_to_img(train_example, data_format='channels_first')
    train_example = train_example.transpose()
    return train_example

def readXML(f):
    mapped = {}
    tree = ET.parse(f)
    root = tree.getroot()
    for elem in root:
        if len(elem) == 0:
            mapped[elem.tag] = elem.text
        for subelem in elem:
            if len(subelem) == 0:
                mapped[subelem.tag] = subelem.text
            for sub2elem in subelem:
                if len(sub2elem) == 0:
                    mapped[sub2elem.tag] = sub2elem.text
    xmax = int(mapped['xmax'])
    xmin = int(mapped['xmin'])
    ymax = int(mapped['ymax'])
    ymin = int(mapped['ymin'])
    C_index = int(mapped['name'][1])
    yTrue = np.zeros((19, 14, 7), dtype='float32')
    yTrue[:, :] = [xmax, xmin, ymax, ymin, 0, 0, C_index]
    return yTrue

def load_DATA(srcDir):
    fileCount = len([name for name in os.listdir(srcDir) if name.endswith(".xml")])
    train_data = np.empty(shape=[fileCount, 640, 480, 3], dtype='float32')
    train_label = np.empty(shape=[fileCount, 19, 14, 7], dtype='float32')
    count = 0
    for filename in os.listdir(srcDir):
        if not filename.endswith(".xml"): continue
        count += 1
        xmlFile = srcDir + "/" + filename
        print(xmlFile)
        imgFile = xmlFile.replace(".xml", ".jpg")
        print(imgFile)
        train_data[count - 1] = LOAD_IMAGE(imgFile)
        train_label[count - 1] = readXML(xmlFile)
    train_label[:, :, :, 4] = np.arange(0, 19, 1).reshape(19, 1)
    train_label[:, :, :, 5] = np.arange(0, 14, 1).reshape(1, 14)
    return [train_data, train_label]

def draw(img, boxes, scores, classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    global img_count
    num_indices = tf.image.non_max_suppression(boxes, scores, max_output_size=3, iou_threshold=0.8)
    boxes = tf.gather(params=boxes, indices=num_indices)
    scores = tf.gather(params=scores, indices=num_indices)
    classes = tf.gather(params=classes, indices=num_indices)
    boxes = K.eval(boxes)
    scores = K.eval(scores)
    classes = K.eval(classes)
    i = 0
    color = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 127, 80), (0, 128, 128), (255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 128, 0), (0, 0, 0)]
    for box, score, cl in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), color[cl], 2)
        i+=1
    img = img.transpose()
    img = array_to_img(img, data_format='channels_first')
    img.save('result'+str(img_count)+'.png')
    img_count += 1
    
if __name__ == '__main__':
    # loading an existed model
    model = load_model(str(setting["weight_file"]+'.h5'), custom_objects={setting["loss"]: lossFunction})
#     model = multi_gpu_model(model, gpus=4)
    srcDir = "group2/images/1"
    print("+++ run: " + srcDir + " " + str(datetime.now()) + "+++")    
    # loading the data
    train_data, train_label = load_DATA(srcDir=srcDir)
    one_hot_encoding = to_categorical(y=train_label[:, :, :, 6], num_classes=10)
    train_label = np.concatenate((train_label, one_hot_encoding), axis = -1)
    
    # predict
    results = model.predict(x=train_data, batch_size=setting['batch_size'])
    setting['batch_size'] = results.shape[0]
    results = tf.convert_to_tensor(results)
    Boxes, Classes, Scores = U.generating_consequences(results)
    x1, y1, x2, y2 = U.transform_to_coordinate(Boxes[:, :, 0], Boxes[:, :, 1], Boxes[:, :, 2], Boxes[:, :, 3])
    x1 = K.reshape(x=x1, shape=(results.shape[0], -1, 1))
    x2 = K.reshape(x=x2, shape=(results.shape[0], -1, 1))
    y1 = K.reshape(x=y1, shape=(results.shape[0], -1, 1))
    y2 = K.reshape(x=y2, shape=(results.shape[0], -1, 1))
    Boxes = K.concatenate([x1, y1, x2, y2])
    for i in range(setting['batch_size']):
        draw(train_data[i], boxes=Boxes[i], scores=Scores[i], classes=Classes[i])