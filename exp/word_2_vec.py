# -*- coding: utf-8 -*-
# Author    ：Yang Ming
# Create at ：2021/4/9
# tool      ：PyCharm

import word2vec

# word2vec.word2phrase('../data/text8', '../data/text8-phrases', verbose=True)
# word2vec.word2vec('../data/text8-phrases', '../data/text8.bin', size=100, binary=True, verbose=True)
# word2vec.word2clusters('../data/text8', '../data/text8-clusters.txt', 100, verbose=True)
# model = word2vec.load('../data/text8.bin')


def train(path):
    word2vec.word2phrase(path, path+'-phrases', verbose=True)
    word2vec.word2vec(path+'-phrases', path+'.bin', size=100, binary=True, verbose=True)
    word2vec.word2clusters(path, path + '.clusters.txt', 100, verbose=True)
    model = word2vec.load(path+'.bin')
    return model

def predict(model,word):
    return model[word]

if __name__ == '__main__':
    model = train('../data/training-data-1/死亡通知书')

    print('done')