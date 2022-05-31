import json
from tqdm import tqdm
from preprocessed import BM25, Corpus

class Selector():
    def __init__(self, corpus:Corpus, k1=1.2, b=0.75) -> None:
        self.__corpus = corpus
        self.__k1 = k1
        self.__b = b
    
    def Select(self, query, pids:list):
        document = []
        idxs = []
        for i in range(len(pids)):
            document += [[sentence] for sentence in self.__corpus.documents[pids[i]]]
            idxs += [(pids[i], j) for j in range(len(self.__corpus.documents[pids[i]]))]
        temp_corpus = Corpus(document)
        bm25 = BM25(temp_corpus, tqdm=False, k1=self.__k1, b=self.__b)
        pid, sid = idxs[bm25.Search(query)[0]]
        return self.__corpus.row_document[pid][sid]
    
if __name__ == '__main__':
    with open('./data/train.json', 'r', encoding='utf-8') as file:
        lines = file.readlines()
    lines = [json.loads(line) for line in lines]

    corpus = Corpus()

    selector = Selector(corpus, k1=1.9)

    r = 0
    for line in tqdm(lines, desc='testing'):
        query = line['question']
        pid = line['pid']
        r += (selector.Select(query, [pid]) == line['answer_sentence'][0])
    print('r: {0}'.format(r / len(lines)))