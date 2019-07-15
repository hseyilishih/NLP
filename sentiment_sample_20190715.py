# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 00:30:20 2019
https://blog.csdn.net/LieQueov/article/details/80568387
@author: USER
"""
#keras使用流程分析（模型搭建、模型保存、模型加載、模型使用、訓練過程可視化、模型可視化等）
'''
【一】本文內容綜述

1. keras使用流程分析（模型搭建、模型保存、模型加載、模型使用、訓練過程可視化、模型可視化等）

2. 利用keras做文本數據預處理

【二】環境準備

1. 數據集下載：http://ai.stanford.edu/~amaas/data/sentiment/

2.安裝Graphviz ，keras進行模型可視化時，會用到該組件： https://graphviz.gitlab.io/_pages/Download/Download_windows.html

【三】數據預處理

將imdb壓縮包解壓後，進行數據預處理。

1. 將每條影評中的部分詞去掉

2. 將影評與label對應起來

3. 將影評映射爲int id，同時將每條影評的長度固定，好作爲定長輸入數據
'''

#!pip install keras
import tarfile

import os, urllib, logging  
import tarfile  
from urllib.request import urlretrieve  
import  keras
 
import numpy as np
import re
from keras.preprocessing import text
from keras.preprocessing import sequence
from keras.utils import plot_model
import matplotlib.pyplot as plt
# XXXimport tokenize
from keras.preprocessing.text import Tokenizer

from keras.models import Sequential
from keras.layers import Dense,Flatten,Embedding,SimpleRNN

###################  
# Step0: Global setting  
####################  
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'  
logging.basicConfig(format=LOG_FORMAT)  
logger = logging.getLogger('IMDBb')  
logger.setLevel(logging.DEBUG)  
  
###################  
# Step1: Download IMDB  
####################  


def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('',text)

def read_files(filetype):
    path = "./aclImdb/"
    file_list=[]
    positive_path=path + filetype+"/pos/"
    for f in os.listdir(positive_path):
        file_list+=[positive_path+f]   
    negative_path=path + filetype+"/neg/"
    for f in os.listdir(negative_path):
        file_list+=[negative_path+f]      
    print('read',filetype, 'files:',len(file_list))      
    all_labels = ([1] * 12500 + [0] * 12500)    
    all_texts  = []
    for fi in file_list:
        with open(fi,encoding='utf8') as file_input:
            filelines = file_input.readlines()       
            all_texts += [rm_tags(filelines[0])]         
    return all_labels,all_texts
 
url="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filepath="aclImdb_v1.tar.gz"
if not os.path.isfile(filepath):
    result=urllib.request.urlretrieve(url,filepath)
    print('downloaded:',result)
if not os.path.exists("aclImdb"):
    tfile = tarfile.open("aclImdb_v1.tar.gz", 'r:gz')
    result=tfile.extractall('.')
    
    
y_train, x_train = read_files('train')
y_test, x_test = read_files('test')


token = Tokenizer(num_words=2000)
token.fit_on_texts(x_train)

x_train_seq = token.texts_to_sequences(x_train)
x_test_seq = token.texts_to_sequences(x_test)

x_train_v = sequence.pad_sequences(x_train_seq,maxlen=100)
x_test_v =  sequence.pad_sequences(x_test_seq,maxlen=100)
 


model = Sequential()
model.add(Embedding(input_dim=2000,output_dim=32,input_length=100))
model.add(Flatten())

#model.add(SimpleRNN(units=32))
model.add(Dense(units=256,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
train_his = model.fit(x_train_v,y_train,batch_size=128,epochs=10,verbose=2,validation_split=0.1)
scores = model.evaluate(x_test_v,y_test,verbose=1)
scores[1]


'''
Reg = re.compile(r'[A-Za-z]*')
stop_words = ['is','the','a']

max_features = 5000
word_embedding_size = 50
maxlen = 400
filters = 250
kernel_size = 3
hidden_dims = 250

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def prepross(file):
    with open(file,encoding='utf-8') as f:
        data = f.readlines()
        data = Reg.findall(data[0])
        # 將句子中的每個單詞轉化爲小寫
        data = [x.lower() for x in data]
        # 將句子中的部分詞從停用詞表中剔除
        data = [x for x in data if x!='' and x not in stop_words]
        # 返回值必須是個句子，不能是單詞列表
        return ' '.join(data)

def imdb_load(type):
    root_path = "E:/nlp_data/aclImdb_v1/aclImdb/"
    # 遍歷所有文件
    file_lists = []
    pos_path = root_path + type + "/pos/"
    for f in os.listdir(pos_path):
        file_lists.append(pos_path + f)
    neg_path = root_path + type + "/neg/"
    for f in os.listdir(neg_path):
        file_lists.append(neg_path + f)
    # file_lists中前12500個爲pos，後面爲neg，labels與其保持一致
    labels = [1 for i in range(12500)]
    labels.extend([0 for i in range(12500)])
    # 將文件隨機打亂，注意file與label打亂後依舊要通過下標一一對應。
    # 否則會導致 file與label不一致
    index = np.arange(len(labels))
    np.random.shuffle(index)
    # 轉化爲numpy格式
    labels = np.array(labels)
    file_lists = np.array(file_lists)
    labels[index]
    file_lists[index]
    # 逐個處理文件
    sentenses = []
    for file in file_lists:
        #print(file)
        sentenses.append(prepross(file))
    return sentenses,labels

def imdb_load_data():
    x_train,y_train = imdb_load("train")
    x_test,y_test = imdb_load("test")
    # 建立單詞和數字映射的詞典
    token = text.Tokenizer(num_words=max_features)
    token.fit_on_texts(x_train)
    # 將影評映射到數字
    x_train = token.texts_to_sequences(x_train)
    x_test = token.texts_to_sequences(x_test)
    # 讓所有影評保持固定長度的詞數目
    x_train = sequence.pad_sequences(x_train,maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test,maxlen=maxlen)
    return (x_train,y_train),(x_test,y_test)

def train():
    (x_train, y_train), (x_test, y_test) = imdb_load_data()
    model = keras.Sequential()
    # 構造詞嵌入層
    model.add(keras.layers.Embedding(input_dim=max_features,output_dim=word_embedding_size,name="embedding"))
    # 通過layer名字獲取layer的信息
    print(model.get_layer(name="embedding").input_shape)
    # 基於詞向量的堆疊方式做卷積
    model.add(keras.layers.Conv1D(filters=filters,kernel_size=kernel_size,strides=1
                                  ,activation=keras.activations.relu,name="conv1d"))
    # 對每一個卷積出的特徵向量做最大池化
    model.add(keras.layers.GlobalAvgPool1D(name="maxpool1d"))
    # fc,輸入是250維，輸出是hidden_dims
    model.add(keras.layers.Dense(units=hidden_dims,name="dense1"))
    # 添加激活層
    model.add(keras.layers.Activation(activation=keras.activations.relu,name="relu1"))
    # fc，二分類問題，輸出維度爲1
    model.add(keras.layers.Dense(units=1,name="dense2"))
    # 二分類問題，使用sigmod函數做分類器
    model.add(keras.layers.Activation(activation=keras.activations.sigmoid,name="sigmoe"))
    # 打印模型各層layer信息
    model.summary()
    # 模型編譯，配置loss，optimization
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.binary_crossentropy,
                  metrics=['accuracy'])
    # 模型訓練
'''
    
# 如果想保存每一個batch的loss等數據，需要傳遞一個callback
'''
    history = LossHistory()
    train_history = model.fit(x=x_train,
                              y=y_train,
                              batch_size=128,
                              epochs=1,
                              validation_data=(x_test,y_test),
                              callbacks=[history])
    show_train_history2(history)
    # 結果可視化
    
    '''
    # fit 返回的log中，有 epochs 組數據，即只保存每個epoch的最後一次的loss等值
    '''
    train_history = model.fit(x=x_train,
                              y=y_train,
                              batch_size=128,
                              epochs=10,
                              validation_data=(x_test,y_test))
    show_train_history(train_history)

    # 模型保存
    model.save(filepath="./models/demo_imdb_rnn.h5")
    # 模型保存一份圖片
    plot_model(model=model,to_file="./models/demo_imdb_rnn.png",
               show_layer_names=True,show_shapes=True)

def show_train_history2(history):
    plt.plot(history.losses)
    plt.title("model losses")
    plt.xlabel('batch')
    plt.ylabel('losses')
    plt.legend()
    # 先保存圖片，後顯示，不然保存的圖片是空白
    plt.savefig("./models/demo_imdb_rnn_train.png")
    plt.show()
    
def show_train_history(train_history):
    print(train_history.history.keys())
    print(train_history.epoch)
    plt.plot(train_history.history['acc'])
    plt.plot(train_history.history['val_acc'])
    plt.title("model accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])
    plt.title("model loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

def gen_predict_data(path):
    sent = prepross(path)
    x_train,t_train = imdb_load("train")
    token = text.Tokenizer(num_words=max_features)
    token.fit_on_texts(x_train)
    x = token.texts_to_sequences([sent])
    x = sequence.pad_sequences(x,maxlen=maxlen)
    return x

RESULT = {1:'pos',0:'neg'}

def predict(path):
    x = gen_predict_data(path)
    model = keras.models.load_model("./models/demo_imdb_rnn.h5")
    y = model.predict(x)
    print(y)
    y= model.predict_classes(x)
    print(y)
    print(RESULT[y[0][0]])

#train()
predict(r"E:\nlp_data\aclImdb_v1\aclImdb\test\neg\0_2.txt")
predict(r"E:\nlp_data\aclImdb_v1\aclImdb\test\pos\0_10.txt")
'''