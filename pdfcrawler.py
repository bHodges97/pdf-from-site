from urllib.request import urlopen
from urllib.error import HTTPError
from urllib.parse import urlparse
from collections import Counter
from os import path
from hashlib import sha256
from operator import itemgetter
from nltk.util import bigrams as nltk_bigrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import json
import subprocess
import csv
import nltk
import numpy as np
import scipy.sparse as sp
from pdffinder import PDFFinder

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading resources")
    nltk.download('averaged_perceptron_tagger',quiet=True)
    nltk.download('wordnet',quiet=True)
    nltk.download('stopwords',quiet=True)
    nltk.download('punkt',quiet=True)


class PDFFreq():
    def __init__(self, exclude = [], find_termfreq = True, find_collocations = False):
        self.pdf_stopwords = set(stopwords.words('english'))
        self.pdf_stopwords.update(stopwords.words('german'))
        self.pdf_stopwords.update(exclude)#add additional stop words here
        self.find_collocations = find_collocations
        self.find_termfreq = find_termfreq

        self.vocab = dict()
        self.stemmer = WordNetLemmatizer()
        self.pdfs = []
        self.hashes = set()
        self._nextid = 0
        self._nextvocab = 0

        #Count vector
        self.j_indices = []
        self.indptr = [0]
        self.values = []

    def add_word(self, word):
        if word not in self.vocab:
            self.vocab[word] = self._nextvocab
            self._nextvocab+=1
        return self.vocab[word]


    def count_vectorize(self,low=2,limit=10000):
        """
        Adapted from sklearn's count_vectorizer
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#
        """
        #construct sparse matrix
        j_indices = np.asarray(self.j_indices, dtype=np.int_)
        indptr = np.asarray(self.indptr, dtype=np.int_)
        values = np.asarray(self.values, dtype=np.int_)
        X = sp.csr_matrix((values, j_indices, indptr),
                          shape=(len(indptr) - 1, len(self.vocab)),
                          dtype=np.int_)
        X.sort_indices()
        #calc docucment and term frequencies
        dfs = np.bincount(X.indices, minlength=X.shape[1])#TODO: Considering replacing min count with tfs and skip calculating dfs
        tfs = np.asarray(X.sum(axis=0)).ravel()

        #mask elements
        mask = np.ones(len(tfs), dtype=bool)
        mask &= dfs >= low
        if mask.sum() > limit:
            mask_inds = (-tfs[mask]).argsort()[:limit]
            new_mask = np.zeros(len(tfs), dtype=bool)
            new_mask[np.where(mask)[0][mask_inds]] = True
            mask = new_mask

        new_indices = np.cumsum(mask) - 1  # maps old indices to new
        for term, old_index in list(self.vocab.items()):
            if mask[old_index]:
                self.vocab[term] = new_indices[old_index]
            else:
                del self.vocab[term]
        kept_indices = np.where(mask)[0]
        self.X = X[:, kept_indices]
        self.tfs = tfs[kept_indices]
        return self.X

    def crawl_html(self, url):
        with urlopen(url) as responce:
            html = responce.read().decode('utf-8')
        parsed_uri = urlparse(url)
        root = '{uri.scheme}://{uri.netloc}'.format(uri=parsed_uri)
        return PDFFinder(root).feed(html)


    def download(url):
        if url == "":
            return ""

        file_path = url.replace("https://hps.vi4io.org/_", "../../data/")
        if not path.isfile(file_path):
            file_path = url.split('/')[-1]
            if not path.isfile(file_path):
                try:
                    pdf = urlopen(url).read()
                    with open(file_path, "bw") as fp:
                        fp.write(pdf)
                except HTTPError as err:
                    print("Error", err.code)
                    return ""
        return file_path

    def add_pdf(self, url, html):
        print("Adding:", url)
        file_path = PDFFreq.download(url)
        if file_path == "":
            print("Does not exist")
            return

        #check if it is valid pdf
        filetype = subprocess.run(["file","-i",file_path], stdout=subprocess.PIPE).stdout.decode('utf-8').split(":",1)[1]
        if "pdf" not in filetype:
            print("Not pdf",filetype,end='')
            return

        #hash
        with open(file_path, 'rb') as f:
            phash = sha256(f.read()).hexdigest()
            if phash in self.hashes:
                c = self.get_conflict(phash)
                print("File hash collision,",c," skipping")

        #pdf to t_freq
        result = subprocess.run(["pdftotext",file_path,"-"], stdout=subprocess.PIPE).stdout.decode('utf-8')
        words = word_tokenize(result)
        term_freq = Counter()

        if self.find_termfreq:
            term_freq = self.word_freq(words)

        if self.find_collocations:
            term_freq += self.bigram_freq(words)

        #save
        idx = self._nextid
        self._nextid+=1
        self.hashes.add(phash)
        self.pdfs.append([idx,html,phash])

        self.j_indices.extend(term_freq.keys())
        self.values.extend(term_freq.values())
        self.indptr.append(len(self.j_indices))
        del term_freq

        print("Success")

    def word_freq(self,words):
        t_freq = Counter()
        #strip garbage
        words = filter(lambda x:x.isalpha(), words)
        words = map(lambda x:x.lower(), words)
        tagged = nltk.pos_tag(words)
        for word,tag in tagged:
            if tag == 'NNS': #If plural , make singular
                word = self.stemmer.lemmatize(word)
            if tag[:2] == 'NN' and len(word) > 1 and word not in self.pdf_stopwords: # word is noun and not stop word
                idx = self.add_word(word)
                t_freq[idx] += 1
        return t_freq

    def bigram_freq(self,words):
        bigrams = [(x.lower(),y.lower()) for (x,y) in nltk_bigrams(words)]
        bigram_freq = Counter()
        filtered_bigrams = []
        for bigram in bigrams:
            if all(len(x) > 1 and x not in self.pdf_stopwords and x.isalpha() for x in bigram):
                w1,w2=bigram
                tagged = nltk.pos_tag(bigram)
                if tagged[1][1] == 'NNS':
                    w2 = self.stemmer.lemmatize(w2)
                bigram_str =  w1+" "+w2
                idx = self.add_word(bigram_str)
                bigram_freq[idx]+=1
        return bigram_freq

    def load_csv(self):
        try:
            self.vocab = np.load("vocab.npy",allow_pickle=True).item()
            self._nextvocab = ax(self.vocab.values()) + 1

            X = sp.load_npz("tfs.npz")
            self.j_indices = X.indices.tolist()
            self.values = X.data.tolist()
            self.indptr = X.indptr.tolist()

            with open("papers.csv","r",encoding='utf-8') as f:
                reader = csv.reader(f)
                for idx,html,phash in reader:
                    idx = int(idx)
                    self.pdfs.append([idx,html,phash])
                    self.hashes.add(phash)
                self._nextid = self.pdfs[-1][0] + 1
        except FileNotFoundError:
            print("Load failed")
            return
        print("Load success")


    def save_csv(self, max_count = 10000,minfreq=2):
        X = self.count_vectorize(limit=max_count,low=minfreq)
        inv_map= {v: k for k,v in self.vocab.items()}

        print("Writing numpy array")
        np.save('vocab.npy',self.vocab)
        sp.save_npz('tfs.npz',self.X)

        print("Writing freq_data.csv")
        with open("freq_data.csv","w",encoding='utf-8') as f:
            c = csv.writer(f, quoting=csv.QUOTE_NONE)
            for idx,count in enumerate(self.tfs):
                c.writerow([inv_map[idx],count])

        print("Writing papers.csv")
        with open("papers.csv","w",encoding='utf-8') as f:
            c = csv.writer(f,quoting=csv.QUOTE_NONNUMERIC)
            for idx,html,phash in self.pdfs:
                c.writerow([idx,html,phash])

        print("Writing related_papers.csv")
        with open("related_papers.csv","w",encoding='utf-8') as f:
            c = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            for i in range(self.X.shape[1]):
                r = self.X[:,i]
                pairs = zip(r.nonzero()[0],r.data)
                pairs = sorted(pairs,key=itemgetter(1), reverse=True)
                if len(pairs) > 20:
                    pairs=pairs[:20]
                associated = {str(k):int(v) for k,v in pairs}
                c.writerow((inv_map[i],json.dumps(associated)))


    def get_conflict(self, phash):
        for idx,_,h in self.pdfs:
            if phash == h:
                return idx


if __name__ == "__main__":
    url = "https://hps.vi4io.org/research/publications?csvlist"
    words = ["et","al","example","kunkel","see","figure","limitless","per"]
    pdfFreq = PDFFreq(words,find_termfreq=False,find_collocations=True)
    pdfFreq.load_csv()
    files = pdfFreq.crawl_html(url)
    for idx,url,html in files:
        pdfFreq.add_pdf(url,html)
    pdfFreq.save_csv()



#print(pdf)
