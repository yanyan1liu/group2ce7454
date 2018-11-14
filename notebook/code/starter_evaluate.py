import keras.backend as K
from keras.models import load_model
import xml.etree.ElementTree as ET
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf
from datetime import datetime
import os
from settings import setting
import cv2
from keras.utils.np_utils import to_categorical


img_count = 0

############################################
if setting["architecture"] == "Yolo_V1":
    import Util_V1 as U
elif setting["architecture"] == "Yolo_V2":
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
    for box, score, cl in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, 'C'+str(cl), (int((x2+x1)/2.0), int((y1+y2)/2.0) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
        print('cl:', cl)
    img = img.transpose()
    img = array_to_img(img, data_format='channels_first')
    img.show()
    img.save('result'+str(img_count)+'.png')
    img_count += 1

def Matching(x1_pred, y1_pred, x2_pred, y2_pred, x_max_true, x_min_true, y_max_true, y_min_true):
    intersect_x1 = K.maximum(x1_pred, x_min_true)
    intersect_y1 = K.maximum(y1_pred, y_min_true)
    intersect_x2 = K.minimum(x2_pred, x_max_true)
    intersect_y2 = K.minimum(y2_pred, y_max_true)

    area_1 = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    area_2 = (x_max_true - x_min_true) * (y_max_true - y_min_true)

    intersect_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)

    IOU = intersect_area / (area_1 + area_2 - intersect_area)
    matching = K.greater_equal(IOU, 0.7)
    return matching

def LOAD_IMAGE(path = 'C:/Users/ZJZ20\Downloads/DeepLearnData/img_0.jpg'):
    image1 = load_img(path)
    train_example = img_to_array(image1, data_format='channels_first')
    img = array_to_img(train_example, data_format='channels_first')
#    img.show()
    train_example = train_example.transpose()
    return train_example


def readXML(f='C:/Users/ZJZ20\Downloads/DeepLearnData/img_0.xml'):
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

def Testing_Performance_location(y_true, y_pred):
    boxes, classes, scores = U.generating_consequences(y_pred)
    x1, y1, x2, y2 = U.transform_to_coordinate(boxes[:, :, 0], boxes[:, :, 1], boxes[:, :, 2], boxes[:, :, 3])
    x1 = K.reshape(x=x1, shape=(setting['batch_size'], -1, 1))
    x2 = K.reshape(x=x2, shape=(setting['batch_size'], -1, 1))
    y1 = K.reshape(x=y1, shape=(setting['batch_size'], -1, 1))
    y2 = K.reshape(x=y2, shape=(setting['batch_size'], -1, 1))
    Boxes = K.concatenate([x1, y1, x2, y2])
    x1_pred = Boxes[:, :, 0]
    y1_pred = Boxes[:, :, 1]
    x2_pred = Boxes[:, :, 2]
    y2_pred = Boxes[:, :, 3]

    x_max_true = y_true[:, 0, 0, 0]
    x_min_true = y_true[:, 0, 0, 1]
    y_max_true = y_true[:, 0, 0, 2]
    y_min_true = y_true[:, 0, 0, 3]
    x_max_true = K.reshape(x_max_true, (setting['batch_size'], 1))
    x_min_true = K.reshape(x_min_true, (setting['batch_size'], 1))
    y_max_true = K.reshape(y_max_true, (setting['batch_size'], 1))
    y_min_true = K.reshape(y_min_true, (setting['batch_size'], 1))
    IOU = Matching(x1_pred=x1_pred, y1_pred=y1_pred, x2_pred=x2_pred, y2_pred=y2_pred, x_max_true=x_max_true,
                   x_min_true=x_min_true, y_max_true=y_max_true, y_min_true=y_min_true)
    IOU = K.cast(IOU, 'float32')
    mat = K.max(IOU, axis=-1)
    return K.cast(tf.count_nonzero(input_tensor=mat), 'float32') / setting['batch_size']


def Testing_Performance_classification(y_true, y_pred):
    boxes, classes, scores = U.generating_consequences(y_pred)
    groundtrue_class = y_true[:, 0, 0, 6]
    groundtrue_class = K.cast(groundtrue_class, 'int64')
    groundtrue_class = K.reshape(groundtrue_class, (setting['batch_size'], 1))
    classification_loss = K.equal(classes, groundtrue_class)
    classification_loss = K.cast(x=classification_loss, dtype='float32')
    class_match = K.max(classification_loss, axis=-1)
    return K.cast(tf.count_nonzero(input_tensor=class_match), 'float32') / setting['batch_size']

if __name__ == '__main__':
    model = load_model('weight-setting5.h5', custom_objects={setting["loss"]: lossFunction})
    srcDir = "D:/Dataset/YoLo/4"
    print("+++ run: " + srcDir + " " + str(datetime.now()) + "+++")
    train_data, train_label = load_DATA(srcDir=srcDir)
    one_hot_encoding = to_categorical(y=train_label[:, :, :, 6], num_classes=10)
    train_label = np.concatenate((train_label, one_hot_encoding), axis = -1)
    model.compile(optimizer=optimizer, loss=lossFunction, metrics=[Testing_Performance_location, Testing_Performance_classification])
    loss_and_metrics = model.evaluate(x=train_data, y=train_label, batch_size=setting['batch_size'])
    print('location_loss', loss_and_metrics[1])
    print('classification_loss', loss_and_metrics[2])

    model.predct()



