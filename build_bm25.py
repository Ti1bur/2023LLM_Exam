import pickle as pkl
import math
from tqdm.auto import tqdm
from six import iteritems
from six.moves import range
import numpy as np
import heapq
from collections.abc import Iterable


PARAM_K1 = 1.5
PARAM_B = 0.75
EPSILON = 0.25


class fastbm25(object):
    """Implementation of Best Matching 25 ranking function.

    Attributes
    ----------
    corpus_size : int
        Size of corpus (number of documents).
    avgdl : float
        Average length of document in `corpus`.
    doc_freqs : list of dicts of int
        Dictionary with terms frequencies for each document in `corpus`. Words used as keys and frequencies as values.
    idf : dict
        Dictionary with inversed documents frequencies for whole `corpus`. Words used as keys and frequencies as values.
    doc_len : list of int
        List of document lengths.
    """

    def __init__(self, corpus):
        """
        Parameters
        ----------
        corpus : list of list of str
            Given corpus.

        """
        self.corpus_size = len(corpus)
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = {}
        self._initialize(corpus)
        self.get_score_by_reversed_index_all_documents(corpus)

    def _initialize(self, corpus):
        """Calculates frequencies of terms in documents and in corpus. Also computes inverse document frequencies."""
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for j, document in tqdm(enumerate(corpus),total=len(corpus)):
            self.doc_len[j] = len(document)
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in iteritems(frequencies):
                if word not in nd:
                    nd[word] = 0
                nd[word] += 1

        self.avgdl = float(num_doc) / self.corpus_size
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        self.nd = nd
        for word, freq in tqdm(iteritems(nd), total=len(nd)):
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = float(idf_sum) / len(self.idf)

        eps = EPSILON * self.average_idf
        for word in tqdm(negative_idfs,total=len(negative_idfs)):
            self.idf[word] = eps

    def similarity_bm25(self, document_a, document_b):
        """Computes BM25 score of given `document A` in relation to given `document B` .

        Parameters
        ----------
        document_a : list of str
            Document to be scored.
        document_b : list of str
            Document to be scored.
        Returns
        -------
        float
            BM25 score.

        """
        assert isinstance(document_a,Iterable),'document a is not iterable'
        assert isinstance(document_b,Iterable),'document b is not iterable'
        score = 0
        doc_freqs = {}
        for word in document_b:
            if word not in doc_freqs:
                doc_freqs[word] = 0
            doc_freqs[word] += 1
        freq = 1
        default_idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
        for word in document_a:
            if word not in doc_freqs:
                continue
            score += (self.idf.get(word,default_idf) * doc_freqs[word] * (PARAM_K1 + 1)
                      / (doc_freqs[word] + PARAM_K1 * (1 - PARAM_B + PARAM_B * len(document_b) / self.avgdl)))
        return score


    def get_score_by_reversed_index_all_documents(self, corpus):
        """
        Build reverted index for documents like {word:{index:grades}}
        """
        document_score = {}
        for index, document in tqdm(enumerate(corpus),total=len(corpus)):
            q_id =  index
            doc_freqs = self.doc_freqs[index]
            for word in document:
                if word not in doc_freqs:
                    continue
                score = (self.idf[word] * doc_freqs[word] * (PARAM_K1 + 1)
                          / (doc_freqs[word] + PARAM_K1 * (1 - PARAM_B + PARAM_B * self.doc_len[index] / self.avgdl)))
                if word not in document_score:
                    document_score[word] = {q_id: round(score, 2)}
                else:
                    document_score[word].update({q_id: round(score, 2)})
        self.document_score = document_score

    def top_k_sentence(self,document,k=1):
        """
        document: Iterable, to be retrieved
        Returns
        -------
        float
            List of [(nearest sentence,index,score)].
        """
        assert isinstance(document,Iterable),'document is not iterable'
        score_overall = {}
        for word in document:
            if word not in self.document_score:
                continue
            for key, value in self.document_score[word].items():
                if key not in score_overall:
                  # print(score_overall)
                    score_overall[key] = value
                else:
                    score_overall[key] += value
        k_keys_sorted = heapq.nlargest(k, score_overall,key=score_overall.__getitem__)
        return [(item,score_overall.get(item,None)) for item in k_keys_sorted]

tokenize_text = []
import pickle as pkl
for i in tqdm(range(61)):
    with open(f'./tmp/token_small/{i}.pkl','rb') as f:
            tokenize_text.extend(pkl.load(f))
bm25 = fastbm25(tokenize_text)
with open('./tmp/bm25.pkl','wb') as f:
    pkl.dump(bm25,f)