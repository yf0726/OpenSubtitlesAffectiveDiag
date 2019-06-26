import re,gensim,os,chardet
import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
from smart_open import smart_open

ps = PorterStemmer() 
tokenizer = RegexpTokenizer(r'\w+')

def sentence_extract(line):
    encode_type = chardet.detect(line)  
    try:
        line = line.decode(encode_type['encoding']) 
        line = line.replace('<GO>','').replace('<EOS>','')
        line = [line.split('|')[1]]
    except:
        line = []
    return line

class MyCorpus(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname)[:1000]: # take 500 files for test
            for line in smart_open(os.path.join(self.dirname, fname), 'rb'):
                yield from sentence_extract(line)
                
def corpus_clean(s):
    s = s.replace('\'','')
    s = tokenizer.tokenize(s.lower())
#     s = ' '.join([ps.stem(x) for x in s])
    s = ' '.join(s)
    return s

