import os
import json
import jieba
import logging
import numpy as np
from tqdm import tqdm

jieba.setLogLevel(logging.INFO)

class Corpus():
    def __init__(self, row_documents=None):
        self.stop_words = self.__GetStopWords()
        self.row_document, self.documents, self.documents_len = self.__GetDocuments(row_documents)
        self.documents_num = len(self.documents)
        self.vocabulary = self.__GetVocabulary()
    
    def __GetStopWords(self):
        #读取停用词表
        with open('../lab1/stopwords(new).txt', 'r', encoding='utf-8') as file:
            stop_words = set([word.split()[0] for word in file.readlines()])
        return stop_words
    
    def __GetDocuments(self, row_documents):
        flag = row_documents == None
        if flag:
            with open('./data/passages_multi_sentences.json', 'r', encoding='utf-8') as file:
                lines = [json.loads(line) for line in file.readlines()]
            row_documents = [line['document'] for line in lines]

            if os.path.exists('./documents.json'):
                with open('./documents.json', 'r', encoding='utf-8') as file:
                    doc = json.load(file)
                documents = doc['documents']
                documents_length = np.array(doc['documents_length'])
            else:
                documents = [[[word for word in jieba.cut(sentence) if word not in self.stop_words]
                            for sentence in document] for document in tqdm(row_documents, desc='processing row documents')]
                documents_length = np.array([sum([len(sentence) for sentence in document]) for document in documents])
                with open('./documents.json', 'w', encoding='utf-8') as file:
                    json.dump({'documents':documents, 'documents_length':documents_length.tolist()}, file)
        else:
            documents = row_documents
            documents_length = np.array([sum([len(sentence) for sentence in document]) for document in documents])
        return row_documents, documents, documents_length
    
    def ChangeDoc(self, documnets):
        self.row_documents = documnets
        self.documents = [[[word for word in jieba.cut(sentence) if word not in self.stop_words]
                         for sentence in document] for document in tqdm(self.row_documents, desc='processing row documents')]
        self.documents_length = np.array([sum([len(sentence) for sentence in document]) for document in self.documents])
        self.documents_num = len(self.documents)

    def __GetVocabulary(self):
        """
        获取字典
        """
        vocabulary = {}
        for document in self.documents:
            for sentence in document:
                for word in sentence:
                    vocabulary[word] = len(vocabulary)
        
        return vocabulary

class BM25():
    def __init__(self, corpus:Corpus, tqdm=True, k1=1.2, b=0.75) -> None:
        self.__tqdm = tqdm
        self.__corpus = corpus

        #计算tf, idf的值
        self.__tf, self.__idf = self.__GetTfIdf(self.__corpus.documents)

        #提前计算所需参数
        self.__alpha = 1 - b + b * self.__corpus.documents_len / np.average(self.__corpus.documents_len)
        
        self.__k1 = k1

    def __GetTfIdf(self, documents:list):
        """
        计算tf和idf的值
        """
        #初始化tf与df
        tf = [{} for _ in range(self.__corpus.documents_num)]
        idf = dict.fromkeys(self.__corpus.vocabulary, 0)

        #遍历计算tf
        for pid in tqdm(range(self.__corpus.documents_num), desc='calculating tf and df') if self.__tqdm else range(self.__corpus.documents_num):
            for sentence in documents[pid]:
                for word in sentence:
                    if word not in tf[pid]:
                        tf[pid][word] = 0
                    tf[pid][word] += 1
                for word in set(sentence):
                    idf[word] += 1
        
        #计算idf
        for word in tqdm(self.__corpus.vocabulary, desc='calculating idf') if self.__tqdm else self.__corpus.vocabulary:
            idf[word] = np.log10(self.__corpus.documents_num) - np.log10(idf[word])
        
        return tf, idf

    def Search(self, query:str) -> list:
        #问题分词并去除不在词表中的词
        query = [word for word in jieba.cut(query) if word in self.__corpus.vocabulary]
        
        #计算rsv
        rsv = np.zeros(self.__corpus.documents_num)
        tf = {}
        for word in query:
            tf = np.zeros(self.__corpus.documents_num)
            for pid in range(self.__corpus.documents_num):
                tf[pid] = self.__tf[pid][word] if word in self.__tf[pid] else 0
            rsv += self.__idf[word] * (self.__k1 + 1) * tf / (self.__k1 * self.__alpha + tf)
        
        return np.argsort(-rsv).tolist()

if __name__ == '__main__':
    #获取语料
    corpus = Corpus()

    #创建索引
    bm25 = BM25(corpus, k1=1.2)

    #读取测试样例
    with open('./data/train.json', 'r', encoding='utf-8') as file:
        lines = [json.loads(line) for line in file.readlines()]
    test_X = [line['question'] for line in lines]
    test_Y = [line['pid'] for line in lines]

    #测试
    top = [1, 3, 5, 10]
    r = [0] * len(top)
    for i in tqdm(range(len(test_X)), desc='testing'):
        ans = bm25.Search(test_X[i])
        for j in range(len(top)):
            r[j] += test_Y[i] in ans[:top[j]]
    for i in range(len(top)):
        print('r: top{0}:{1}'.format(top[i], r[i] / len(test_X)))
