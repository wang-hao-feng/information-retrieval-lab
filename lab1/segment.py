import os
import json
import tqdm
from pyltp import Segmentor

class Segment():
    def __init__(self, stopwords=[]):
        self.stop_words = set()
        for file_path in stopwords:
            self.__ReadStopWords(file_path)

    def __ReadStopWords(self, file_path:str) -> None:
        """
        读取停用词表
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            words = file.read().split()
            self.stop_words = self.stop_words.union(set(words))

    def Segment(self, dir_path='./craw', prefix='./preprocess_all.json') -> list:
        """
        分词，将结果存入prefix中并返回
        """
        segmentor = Segmentor('./cws.model')
        page_descriptions = []

        #遍历爬取的网页描述文件提取
        for _, dirs, _ in os.walk(dir_path):
            #初始化进度条
            bar = tqdm.tqdm(desc='分词', total=len(dirs))
            for dir in dirs:
                #读取文件
                with open(os.path.join(os.path.join(dir_path, dir), 'page_description.json'), 'r') as file:
                    page_description = json.load(file)
                
                #分词并去除停用词
                for key in ['title', 'parapraghs']:
                    page_description[key] = [word for word in segmentor.segment(page_description[key]) 
                                            if word not in self.stop_words]
                page_descriptions.append(page_description)
                #更新进度条
                bar.update()
            break
                            
        segmentor.release()

        #存入磁盘
        with open(os.path.join('./', prefix), 'w', encoding='utf-8') as file:
            json.dump(page_descriptions, file)
        
        return page_descriptions

if __name__ == '__main__':
    segmentor = Segment(stopwords=['stopwords.txt', 'stopwords(new).txt'])
    segment = segmentor.Segment()
    with open('./preprocess.json', 'w', encoding='utf-8') as file:
        for i in range(10):
            file.write(json.dumps(segment[i], ensure_ascii=False) + '\n')