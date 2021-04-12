# -*- coding: utf-8 -*-
# Author    ：Yang Ming
# Create at ：2021/4/12
# tool      ：PyCharm

import jieba
from opencc import OpenCC
from gensim.models import word2vec



# Settings
seed = 666
sg = 0
window_size = 10
vector_size = 100
min_count = 1
workers = 8
epochs = 30
batch_words = 10000

def tokenize(data_path, data_save_path):
    cc = OpenCC('t2s')
    with open(data_save_path, 'w', encoding='utf-8') as new_f:
        with open(data_path, 'r', encoding='utf-8') as f:
            for times, data in enumerate(f, 1):
                print('data num:', times)
                data = cc.convert(data)
                data = jieba.cut(data)
                data = [word for word in data if word != ' ']
                data = ' '.join(data)
                new_f.write(data)

def train_w2v_zh(data_path, model_save_path):
    train_data = word2vec.LineSentence(data_path)
    model = word2vec.Word2Vec(
    train_data,
    min_count=min_count,
    vector_size=vector_size,
    workers=workers,
    epochs=epochs,
    window=window_size,
    sg=sg,
    seed=seed,
    batch_words=batch_words
)
    model.build_vocab(train_data)
    model.train(train_data,total_examples = model.corpus_count,epochs = model.epochs)

    model.save(model_save_path)

def test_w2v_zh(model_path, word):
    model = word2vec.Word2Vec.load(model_path)
    return model.wv[word]

if __name__ == '__main__':
    data_path = './../data/training-data-1/死亡通知书'
    model_path = './../model/w2v_model_exp_1.model'
    new_data_path = f'./../data/training-data-1/seg_{data_path.split("/")[-1]}'
    tokenize(data_path,new_data_path)
    train_w2v_zh(new_data_path,model_path)
    print(test_w2v_zh(model_path,'通知'))
