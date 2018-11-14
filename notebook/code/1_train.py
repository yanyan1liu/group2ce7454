import keras
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.models import load_model
import xml.etree.ElementTree as ET
from keras.preprocessing.image import load_img, img_to_array, array_to_img
# from keras.utils import multi_gpu_model
import pydot
import numpy as np
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from datetime import datetime
import os
from settings import setting

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


# initModel is a sign for if we already have a trained model stored in .h5 file.
def main(iniModel, setting):      
    # Choosing different architecture
    if setting["architecture"] == "Yolo_V1":
        import Yolo_V1 as yolo
        import Util_V1 as U     
    elif setting["architecture"] == "Yolo_V2":
        import Yolo_V2 as yolo
        import Util_V2 as U
        
        
    # Choosing different decay mechanism
    if setting['model'] == "DecayByBatch":
        optimizer = Adam(lr=setting["lr"])
        CallBackFun = U.lr_minimum()
    elif setting['model'] == "DecayByEpoch":
        optimizer = Adam(lr=setting["lr"])
        CallBackFun = U.DecayByEpoch()
    
    # Choosing different loss function
    if setting['loss'] == 'Loss_v2':
        lossFunction = U.Loss_v2
    elif setting['loss'] == 'Loss_v3':
        lossFunction = U.Loss_v3  
    
    # folderCount is a variable whose value represents how many folders the model are loading
    # each folder has 1000 images with .xml files - deleted
    # folder 0 has 20,000 images with .xml files
    # folder 1 has 2,410 images with .xml files 
    # isDone is a variable that represents if the model has been loaded, to avoid repeatedly loading
    folderCount = 1
    isDone = False
    for count in range(0, folderCount, 1):
        srcDir = "src/" + count 
        print("+++ run: "+ srcDir + " " + str(datetime.now()) + "+++")
        train_data, train_label = load_DATA(srcDir)
        
        # to transform the class representation into one-hot encoding for cross-entropy loss  
        one_hot_encoding = to_categorical(y=train_label[:, :, :, 6], num_classes=10)
        train_label = np.concatenate((train_label, one_hot_encoding), axis = -1)

        if iniModel:
            model = yolo.network_architecture(input_data=[640, 480, 3])
#             model = multi_gpu_model(model, gpus=4)
            model.compile(optimizer=optimizer, loss=lossFunction)
            iniModel = False
            plot_model(model=model, to_file='Architecture.png', show_layer_names=False)
            SVG(model_to_dot(model).create(prog='dot', format='svg'))
        elif not isDone:
            model = load_model(setting["weight_file"], custom_objects={setting["loss"]: lossFunction})
#             model = multi_gpu_model(model, gpus=4)
        isDone = True
        
        checkpoint = keras.callbacks.ModelCheckpoint(setting["weight_file"]+'-{epoch:08d}.h5', save_weights_only=True, period=1)
        history = model.fit(x=train_data, y=train_label, validation_split=0.20, batch_size=setting["batch_size"], 
                            epochs=setting["epochs"], callbacks=[CallBackFun, checkpoint])
        
        # saving a log in case the exception occurs
        f = open(setting["loss_file"], "a+")
        f.write("\n")
        description = "+++ run: "+ srcDir + " " + str(datetime.now()) + "+++"
        f.write(description)
        f.write("\n")
        f.write(str(history.history))
        f.close()
        print("+++ saved: " + str(datetime.now()) + "+++")
        model.save(setting["weight_file"])
        
print("+++ start: " + str(datetime.now()) + "+++")
iniModel = True
if os.path.isfile(setting["weight_file"]):
    iniModel = False
main(iniModel, setting)
print("+++ finished: " + str(datetime.now()) + "+++")