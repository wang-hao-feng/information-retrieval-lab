from ast import Str
import re
import json
import jieba
import logging
from tqdm import tqdm
from question_classification import QuestionClassifier
from pyltp import NamedEntityRecognizer, Postagger, Segmentor
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

jieba.setLogLevel(logging.INFO)

smooth = SmoothingFunction()

class AnswerExtractor():
    def __init__(self, stop_words=set()) -> None:
        self.__ltpseg = Segmentor('../model/cws.model')
        self.__ltppos = Postagger('../model/pos.model')
        self.__ltpner = NamedEntityRecognizer('../model/ner.model')
        self.__extractor_dict = {'HUM':self.__ExtractHUM, 'LOC':self.__ExtractLOC, 'NUM':self.__ExtractNUM,
                                'TIM':self.__ExtractTIME, 'OBJ':self.__ExtractOBJ, 'DES':self.__ExtractDES}
        self.__classifier = QuestionClassifier()
        self.__classifier.train('./question_classification/trian_questions.txt')
        self.__stop_word = stop_words

    def Extract(self, query, sentence):
        sentence = self.__Shrink(query, sentence)
        query_type = self.__GetQueryType(query)
        extractor = self.__extractor_dict[query_type]
        ans = extractor(query, sentence)
        return ans

    def __GetQueryType(self, query):
        for word in ['哪个人', '谁']:
            if word in query:
                return 'HUM'
        for word in ['什么地方', '哪里', '哪儿']:
            if word in query:
                return 'LOC'
        for word in ['多少', '几', '多重', '多长', '多大', '多久', '多宽', '多深', '多远']:
            if word in query:
                return 'NUM'
        for word in ['什么时候', '哪一年', '哪一天', '哪年', '何时', '多久', '时间', '哪天']:
            if word in query:
                return 'TIM'
        return self.__classifier.Sort([query])[0]

    def __ExtractHUM(self, query, sentence) -> str:
        #取第一个出现的人名或机构名作为答案
        words, _, nertags = self.__SentenceAnalyse(sentence)
        ans = ''
        for idx, nertag in enumerate(nertags):
            if nertag[-2:] == 'Nh' or nertag[-2:] == 'Ni':
                ans += words[idx]
        return ans

    def __ExtractLOC(self, query, sentence) -> str:
        #取第一个出现地名作为答案
        words, _, nertags = self.__SentenceAnalyse(sentence)
        ans = ''
        for idx, nertag in enumerate(nertags):
            if nertag.endswith('Ns'):  # 地点
                ans += words[idx]
        return ans

    def __ExtractNUM(self, query, sentence) -> str:
        words, postags = self.__SentenceAnalyse(sentence, deep=2)
        result = ''
        i = 0
        while i < len(postags):
            if postags[i] == 'm':
                result += words[i]
                i += 1
                while i < len(postags) and (postags[i] == 'm' or postags[i] == 'c'):
                    result += ' ' + words[i]
                    i += 1
            else:
                i += 1
        return result

    def __ExtractTIME(self, query, sentence) -> str:
        words, postags = self.__SentenceAnalyse(sentence, deep=2)
        result = ''
        for idx, pos_tag in enumerate(postags):
            if pos_tag == 'nt':
                result += words[idx]
        return result

    def __ExtractOBJ(self, query, sentence) -> str:
        #考虑书名
        if "《" in sentence:
            ans = ''.join(re.findall(r'《*》', sentence))
        else:
            query_words = self.__SentenceAnalyse(query, deep=1)
            sentence_words = self.__SentenceAnalyse(sentence, deep=1)
            ans = ''.join([word for word in sentence_words if word not in query_words])

        return ans

    def __ExtractDES(self, query, sentence) -> str:
        query_words = self.__SentenceAnalyse(query, deep=1)
        sentence_words = self.__SentenceAnalyse(sentence, deep=1)
        ans = ''.join([word for word in sentence_words if word not in query_words])
        return ans

    def __SentenceAnalyse(self, sentence, deep=3):
        words = [word for word in self.__ltpseg.segment(sentence) if word not in self.__stop_word]
        if deep == 1:
            return list(words)
        postags = self.__ltppos.postag(words)
        if deep == 2:
            return list(words), list(postags)
        nertags = self.__ltpner.recognize(words, postags)
        return list(words), list(postags), list(nertags)

    def __Shrink(self, query:Str, sentence:str):
        return sentence

def BLEU1(answer, ground_truth):
    answer == ' '.join(jieba.cut(answer))
    ground_truth = ' '.join(jieba.cut(ground_truth))
    bleu1 = sentence_bleu(ground_truth, answer, weights=(1, 0, 0, 0), smoothing_function=smooth.method1)
    return bleu1

if __name__ == '__main__':
    with open('./data/train.json', 'r', encoding='utf-8') as file:
        lines = [json.loads(line) for line in file.readlines()]

    extractor = AnswerExtractor()
    bleu1 = 0
    for line in tqdm(lines, desc='testing'):
        answer = extractor.Extract(line['question'], line['answer_sentence'][0])
        bleu1 += BLEU1(answer, line['answer'])
    print('bleu1:{0}'.format(bleu1 / len(lines)))

    """ s = Segmentor('../model/cws.model')
    p = Postagger('../model/pos.model')
    n = NamedEntityRecognizer('../model/ner.model')
    words = s.segment('墓主人为晋穆侯夫人。')
    postags = p.postag(words)
    nertags = n.recognize(words, postags)
    print(['{0}/{1}/{2}'.format(words[i], postags[i], nertags[i]) for i in range(len(words))]) """