import json
from tqdm import tqdm
from preprocessed import Corpus, BM25
from answer_sentence_selection import Selector
from answer_span_selection import AnswerExtractor

corpus = Corpus()
bm25 = BM25(corpus)
selector = Selector(corpus)
extractor = AnswerExtractor()

with open('./data/test.json', 'r', encoding='utf-8') as file:
    lines = [json.loads(line) for line in file.readlines()]

with open('test_answer.json', 'w', encoding='utf-8') as file:
    for line in tqdm(lines, desc='running'):
        query = line['question']
        pid = bm25.Search(query)[:3]
        sentence = selector.Select(query, pid)
        answer = extractor.Extract(query, sentence)
        file.write(json.dumps({'qid':line['qid'], 'question':query, 'answer_pid':[pid], 'answer':answer}, ensure_ascii=False))
        file.write('\n')