import os
import json
import time
import queue
import shutil
from urllib import response
import requests
import urllib3
import numpy as np
from tqdm import tqdm
from urllib.request import urlopen
from bs4 import BeautifulSoup
from threading import Lock, Thread

class Crawler():
    def __init__(self, thread_num=10, prefix='./craw', sleep_time=0, time_out=5) -> None:
        self.thread_num = thread_num
        self.prefix = prefix
        self.__condition_lock = Lock()
        self.__queue_lock = Lock()
        self.__stop_condition = np.array([0, 0])
        self.__url_queue = queue.Queue(0)
        self.__flag = True
        self.__bar = [None, None]
        self.__sleep_time = sleep_time
        self.__time_out = time_out
        self.__http = urllib3.PoolManager()
        
    
    def run(self) -> None:
        """
        运行爬虫
        """
        #定义退出标志
        self.__flag = True

        #初始化进度条
        self.__bar[0] = tqdm(total=1000, desc='网页')
        self.__bar[1] = tqdm(total=100, desc='带附件的网页')

        #创建存储爬取的网页的文件夹
        if os.path.exists(self.prefix):
            shutil.rmtree(self.prefix)
        os.mkdir(self.prefix)

        #创建守护线程，其不断向url队列中填充新的url
        thread0 = Thread(target=self.__GetURLFromPage, daemon=True)
        thread0.start()

        #创建爬取线程
        threads = []
        for _ in range(self.thread_num):
            threads.append(Thread(target=self.__Crawl))
            threads[-1].start()
            time.sleep(0.5)
        
        #等待爬取线程结束
        for i in range(self.thread_num):
            threads[i].join()
        
        self.__flag = False

    def __GetURLFromPage(self) -> None:
        """
        当队列为空时，读取页面中所有的url存入队列
        """
        page = 0
        todayHIT_url = 'http://today.hit.edu.cn/category/10?page='
        url_head = 'http://today.hit.edu.cn'
        session = requests.session()
        session.keep_alive = False
        while self.__flag:
            while True:
                self.__queue_lock.acquire()
                if self.__url_queue.empty():
                    break
                else:
                    self.__queue_lock.release()
                    time.sleep(self.__sleep_time)

            try:
                #读取该url下的页面
                try:
                    #response = urlopen(todayHIT_url + str(page), timeout=self.__time_out)
                    #html = response.read().decode('utf-8')
                    #response = self.__http.request('GET', todayHIT_url + str(page))
                    #html = response.data.decode('utf-8')
                    response = session.get(todayHIT_url + str(page), timeout=self.__time_out)
                    html = response.content.decode('utf-8')
                except Exception:
                    page += 1
                    continue
                page += 1

                #解析该html下可供爬取的网页
                url_list = []
                soup = BeautifulSoup(html, 'html.parser')
                for a in soup.select('span > span > a'):
                    url_list.append(a.get('href'))
                
                #将url添加到url队列中
                for url in url_list:
                    self.__url_queue.put(url_head + url)
            except Exception:
                page += 1
            
            self.__queue_lock.release()

    def __Crawl(self) -> None:
        """
        根据url队列中的url爬取网页并进行预处理
        """
        url = None
        session = requests.session()
        session.keep_alive = False
        while(1):
            #从队列中获取url
            self.__queue_lock.acquire()
            try:
                url = self.__url_queue.get()
            except queue.Empty:
                self.__queue_lock.release()
                time.sleep(self.__sleep_time)
                continue
            self.__queue_lock.release()

            #请求对应的页面
            try:
                #response = urlopen(url, timeout=self.__time_out)
                #html = response.read().decode('utf-8')
                #response = self.__http.request('GET', url)
                #html = response.data.decode('utf-8')
                response = session.get(url, timeout=self.__time_out)
                html = response.content.decode('utf-8')
            except Exception:
                continue

            #判断是否是登陆页面，是则跳过当前页面
            if self.__IsLogPage(html):
                time.sleep(self.__sleep_time)
                continue

            #解析页面并将信息等存入磁盘
            have_appendix = self.__Extract_text(url, html)

            #修改计数
            self.__condition_lock.acquire()
            #更新进度条
            if self.__stop_condition[0] < 1000:
                self.__bar[0].update()
            if have_appendix and self.__stop_condition[1] < 100:
                self.__bar[1].update()
            self.__stop_condition += [1, 1 if have_appendix else 0]
            #满足退出条件后线程退出
            if self.__stop_condition[0] >= 1000 and self.__stop_condition[1] >= 100:
                self.__condition_lock.release()
                return
            self.__condition_lock.release()
            time.sleep(self.__sleep_time)

    def __Extract_text(self, url:str, html:str) -> bool:
        """
        从html中抽取正文、标题、附件并将信息存入一个字典中，字典以及附件
        网页相关信息存入由url生成的文件夹名的文件夹中
        """
        session = requests.session()
        session.keep_alive = False
        #网页描述信息
        page_description = {'url':url, 'title':None, 'parapraghs':'', 'file_name':[]}

        #创建用于存储网页描述信息的文件夹，文件夹由url生成
        dir_path = os.path.join(self.prefix, self.__GetDirNameFromURL(url))
        #如果文件夹已存在则跳过当前url
        if os.path.exists(dir_path):
            return
        os.mkdir(dir_path)

        #解析网页
        soup = BeautifulSoup(html, 'html.parser')

        #查找标题
        page_description['title'] = soup.find_all('h3')[0].get_text()

        #查找正文
        for text in soup.find_all('p'):
            page_description['parapraghs'] += text.get_text().strip()
        
        #如果有附件，则下载附件
        appendixs = soup.find_all('span', {'class':'file--x-office-document'})
        for appendix in appendixs:
            a = appendix.find_next('a')
            #获取文件名
            page_description['file_name'].append(a.get_text().strip())
            #建立文件存储路径
            appendix_path = os.path.join(dir_path, page_description['file_name'][-1])
            #下载文件
            try:
                with open(appendix_path, 'wb') as file:
                    #response = urlopen(a.get('href'), timeout=self.__time_out)
                    #file.write(response.read())
                    #response = self.__http.request('GET', a.get('href'))
                    #file.write(response.data)
                    response = session.get(a.get('href'), timeout=self.__time_out)
                    file.write(response.content)
                time.sleep(self.__sleep_time)
            except Exception:
                page_description['file_name'].pop()

        #将网页描述信息存入磁盘
        with open(os.path.join(dir_path, 'page_description.json'), 'w') as file:
            json.dump(page_description, file)

        #返回是否存在附件
        return len(appendixs) > 0

    def __IsLogPage(self, html) -> bool:
        """
        根据标题判断当前页是否是登陆页面
        """
        soup = BeautifulSoup(html, 'html.parser')
        return len(soup.find_all('h3')) == 0

    def __GetDirNameFromURL(self, url:str) -> str:
        """
        根据url中文章的日期和编号生成文件名
        """
        idx = url.split('/')[-4:]
        return idx[0] + '_' + idx[1] + '_' + idx[2] + '_' + idx[3]

if __name__ == '__main__':
    crawler = Crawler(thread_num=1, time_out=5)
    crawler.run()