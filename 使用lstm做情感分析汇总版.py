#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:23:57 2019

@author: zhouxj
"""
import pandas as pd
import numpy as np
import jieba
import json
from tqdm import tqdm
import re
import gensim
from gensim.models import word2vec
#from gensim.models.word2vec import LineSentence
from gensim.corpora import Dictionary
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout,Activation
np.random.seed(1337)  # For Reproducibility
from gensim.models.word2vec import Word2Vec
from keras.preprocessing import sequence
import time
import keras
from keras.models import load_model

# In[] 参数
vocab_dim = 300 #词向量的维度
n_iterations = 1  # ideally more..
n_exposures = 10 # 所有频数超过10的词语
window_size = 7
n_epoch = 6
input_length = 1000 
maxlen = 1000 #每个新闻的最大长度
batch_size = 32

# In[]
with open('stop_words.txt', 'r') as f:
    stop_words = [line.strip() for line in f.readlines()]
train = pd.read_table('coreEntityEmotion_train.txt', encoding = 'utf-8', names = ['content'])
emotion_dict = {'POS': 1, 'NEG': -1, 'NORM': 0, 'POS ': 1}
news_frame = pd.DataFrame()
for i in tqdm(range(len(train))):
    title = json.loads(train['content'][i])['title']
    content = json.loads(train['content'][i])['content']
    emotion = sum([emotion_dict[i] for i in [i['emotion'] for i in json.loads(train['content'][i])['coreEntityEmotions']]])
    frame = pd.DataFrame([emotion, title, content]).T
    news_frame = pd.concat([news_frame, frame])
news_frame.columns = ['emotion', 'title', 'content']
news_frame = news_frame[(news_frame['emotion'] == 0) | (news_frame['emotion'] == 1)| (news_frame['emotion'] == 2)| (news_frame['emotion'] == 3)| (news_frame['emotion'] == -1)| (news_frame['emotion'] == -2)|(news_frame['emotion'] == -3)]
news_frame = news_frame.reset_index(drop = True)
#查看y值得计数频率情况
news_frame['emotion'].value_counts()

# In[]构建word2vec
r = '[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>（）?@，。?★、…【】\xa0《》？“”‘’！[\\]^_`{|}~•；：·® \n]+'
news_frame['切词']  = news_frame.content.apply(lambda x: [i for i in jieba.cut(re.sub(r, '', str(x)))])
news_frame['模型词'] = news_frame['切词'].apply(lambda x: str(' '.join(x)))
fileSegWordDonePath ='corpus_lstm.txt'
with open(fileSegWordDonePath,'w') as fW:
    for i in range(0, len(news_frame)):
        fW.write(news_frame['模型词'][i])
        fW.write('\n')
        
start = time.time()
inp = 'corpus_lstm.txt'
sentences = word2vec.Text8Corpus(inp)
yuqing_model = word2vec.Word2Vec(sentences, size=vocab_dim) #vocab_dim = 300
end = time.time()
running_time = end-start
print(running_time)
yuqing_model.save('./lstm_0507/Word2vec_model.pkl') #构建完成，存成pkl格式

# In[] 构建x值
def create_dictionaries(model=None,combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries
    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        #  freqxiao10->0 所以k+1
        w2indx = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过10的词语的索引,(k->v)=>(v->k)
        w2vec = {word: model[word] for word in w2indx.keys()}#所有频数超过10的词语的词向量, (word->model(word))
        def parse_dataset(combined): # 闭包-->临时使用
            ''' Words become integers
            '''
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0) # freqxiao10->0
                data.append(new_txt)
            return data # word=>index
        combined=parse_dataset(combined)
        combined= sequence.pad_sequences(combined, maxlen=maxlen)#每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec,combined
    else:
        print ('No data provided...')
        
combined = news_frame['模型词']
index_dict, word_vectors, combined = create_dictionaries(model=yuqing_model,combined=combined)

# In[] 构建y值
def translate(x):
    if x > 0:
        result_tran = 2 #如果是正面问题 则为2
    elif x < 0:
        result_tran = 1 #如果是负面问题 则为1
    else:
        result_tran = 0 #如果是中性问题 则为0
    return result_tran
news_frame['label'] = news_frame['emotion'].apply(lambda x: translate(x))

# In[] 将数据集准备好
def get_data(index_dict,word_vectors,combined,y):
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim)) # 初始化 索引为0的词语，词向量全为0
    for word, index in index_dict.items(): # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    y_train = keras.utils.to_categorical(y_train,num_classes=3) 
    y_test = keras.utils.to_categorical(y_test,num_classes=3)
    # print x_train.shape,y_train.shape
    return n_symbols,embedding_weights,x_train,y_train,x_test,y_test
n_symbols,embedding_weights,x_train,y_train,x_test,y_test=get_data(index_dict, word_vectors,combined, news_frame['label'])

# In[] train model
def train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test):
    print ('Defining a Simple Keras Model...')
    model = Sequential()  # or Graph or whatever
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))  # Adding Input Length
    model.add(LSTM(output_dim=50, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax')) # Dense=>全连接层,输出维度=7
    model.add(Activation('softmax'))
    print ('Compiling the Model...')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    print ("Train...") # batch_size=32
    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch,verbose=1)
    print ("Evaluate...")
    score = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    '''
    yaml_string = model.to_yaml()
    with open('../舆情/lstm/lstm.yml', 'w') as outfile:
        outfile.write(yaml_string)
    model.save_weights('../舆情/lstm/lstm.h5')
    '''
    model.save('lstm_model.h5')
    print ('Test score:', score)
train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test)

# In[] predict via model
lstm_model = load_model('./model/lstm_model.h5')
word2vec_model = Word2Vec.load('./model/Word2vec_model.pkl')

def create_dictionaries(model=None,
                        combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries
    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        #  freqxiao10->0 所以k+1
        w2indx = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过10的词语的索引,(k->v)=>(v->k)
        w2vec = {word: model[word] for word in w2indx.keys()}#所有频数超过10的词语的词向量, (word->model(word))
        def parse_dataset(combined): # 闭包-->临时使用
            ''' Words become integers
            '''
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0) # freqxiao10->0
                data.append(new_txt)
            return data # word=>index
        combined=parse_dataset(combined)
        combined= sequence.pad_sequences(combined, maxlen=maxlen)#每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec, combined
    else:
        print ('No data provided...')

def get_emotion_class(content_text, word2vec_model=word2vec_model, lstm_model=lstm_model):
    '''apply to each content_text'''
    words=jieba.lcut(content_text)
    words=np.array(words).reshape(1,-1)
    _,_,combined = create_dictionaries(model=word2vec_model, combined=words)
    emotion_class = lstm_model.predict_classes(combined)
    # try:
    #     _,_,combined = create_dictionaries(model=word2vec_model, combined=words)
    #     emotion_class = lstm_model.predict_classes(combined)
    # except:
    #     emotion_class = 99 #未知情绪类别
    return emotion_class
score = get_emotion_class(content_text)[0]