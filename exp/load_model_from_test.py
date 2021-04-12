# -*- coding: utf-8 -*-
# Author    ：Yang Ming
# Create at ：2021/4/12
# tool      ：PyCharm

from gensim.models.word2vec import KeyedVectors


def load_model(path,save=None):
    wv_from_text = KeyedVectors.load_word2vec_format(path, binary=False)
    if save:
        wv_from_text.save(save)
    return wv_from_text


if __name__ == '__main__':
    file = './../model/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'
    load_model(file,save='./../model/dynwin5_model.model')