#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 09:44:57 2019

@author: zhouxj
"""
import yaml
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout,Activation
from keras.models import model_from_yaml
from gensim.models.word2vec import Word2Vec
from keras.preprocessing import sequence
import collections
import time
import keras
from keras.models import load_model
import jieba
import numpy as np
import pandas as pd
import jieba
from tqdm import tqdm
import re
import gensim
from gensim.models import word2vec
from gensim.models.word2vec import LineSentence
from gensim.corpora.dictionary import Dictionary

np.random.seed(1337)  # For Reproducibility
maxlen = 100
string = "小米生态链，须眉便携式涡轮三叶剃须刀报价149元，限时限量40元券，实付109元包邮，领券并购买。之前最低129元，新低价了；有白色、金色和黑色三种款式可选，三红新店，特价促销。天猫小米生态链，须眉便携式涡轮三叶剃须刀券后109元领40元券淘宝小米生态链，无染天然抑菌竹纤维抽纸130抽×12包第二件0元【拍2件】券后29.9元领20元券天猫小米生态链，直白旗舰店沙龙级负离子吹风机券后149元领50元券须眉的这款剃须刀主打轻巧便携和简约设计，长度11.8cm，重量仅为136g，直径与一枚1元硬币相当。机身为纯色金属材质，金色和黑色款筒身采用铝合金阳极氧化工艺+细喷砂（上盖），白色款的筒身为铝合金烤漆工艺，精美养眼。须眉涡轮三叶剃须刀的刀网薄而大，刀网仅厚0.09mm，柔韧可形变，而且网孔以回旋排列从内到外逐渐增大，这种设计增大了剃须刀与皮肤的接触面积，剃须体验温柔舒适。支持IPX7级防水——真正的防水，可以整机浸没水中清洗，也可以开盖直接冲洗刀头。此外，须眉涡轮三叶剃须刀还支持2小时快充（Micro USB接口），充满电后每天剃须2分钟，续航可达30天，省去经常充电的烦恼，尤其适合差旅一族随行携带。• 点此享受小米生态链，须眉便携式涡轮三叶剃须刀109元：领券并购买。• 火辣的价格、火辣的商品，更多促销优惠下载App专享。• 搜索“辣品”可关注辣品官方微博、微信公众号账号。天猫“他买了5盒”，耐时旗舰店8节5号3000mAh锂铁电池一共有8节，采用卷绕结构，不漏液、不短路券后19.9元领20元券天猫微软Office 2019家庭和学生版 永久激活券后288元领300元券天猫微软Office 2016家庭和学生版 电子密钥 永久激活券后198元领460元券天猫微软Win10专业版 电子密钥 永久激活券后349元领50元券淘宝暖暖一杯，江中猴姑早餐米稀米糊450克15天装物美料足，早上起来喝一杯，“胃暖暖的，很舒服”。券后48元领70元券天猫58款可选！凡客诚品旗舰店男女纯棉短袖T恤第二件半价券后29元领20元券淘宝小米生态链，无染天然抑菌竹纤维抽纸130抽×12包第二件0元【拍2件】券后29.9元领20元券"
words=jieba.lcut(string)
words=np.array(words).reshape(1,-1)
word2vec_model = Word2Vec.load('Word2vec_model.pkl')


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
        return w2indx, w2vec,combined
    else:
        print ('No data provided...')
_,_,combined=create_dictionaries(word2vec_model,words)

f = open('lstm.yml')
#data = yaml.load(f)
lstm_model = load_model(f)
lstm_model.predict_classes(combined)