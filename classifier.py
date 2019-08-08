from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, strip_accents_unicode
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from time import time
import os
import subprocess
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Classifier():
    def __init__(self):
        self.tf = None
        self.vocab = None

    def count(self, path):
        stopwords = frozenset(nltk_stopwords.words('english'))#very slow if not set
        self.filenames = [x for x in list(os.walk(path))[0][2]]
        files = [f'{path}/{x}' for x in filenames]
        vectorizer = CountVectorizer(preprocessor=Classifier.preprocess,
                analyzer="word",
                tokenizer=Classifier.tokenizer,
                stop_words=[],
#               max_df=0.5,#ignore top 50%
                min_df=2,
                max_features=10000)
        self.tf = vectorizer.fit_transform(files)
        self.tfidf = self._tfidf_transform(self.tf)
        self.vocab = vectorizer.vocabulary_
        return self.tfidf

    def _tfidf_transform(self,tf):
        tfidfTransformer = TfidfTransformer()
        self.tfidf = tfidfTransformer.fit_transform(tf)
        return self.tfidf

    def load(self, path="."):
        self.tf = sp.load_npz(f"{path}/tfs.npz")
        self.vocab = np.load(f"{path}/vocab.npz", allow_pickle=True)["arr_0"].item()
        self.tfidf = self._tfidf_transform(self.tf)
        return self.tfidf

    def save(self, path="."):
        if self.tfidf == None or self.vocab == None:
            print("No term frequencies loaded")
            return
        sp.save_npz(f"{path}/tfs.npz",self.tfidf)
        np.savez(f"{path}/vocab.npz",self.vocab)

    def classify(self, tfidf=None, clusters=5, verbose=False, plot=False):
        if tfidf == None:
            tfidf = self.tfidf
        self.clusters = clusters
        svd = TruncatedSVD(n_components=2)
        reduced = svd.fit_transform(tfidf)
        km = MiniBatchKMeans(n_clusters=clusters, init='k-means++', verbose=verbose)
        #km = KMeans(n_clusters=clusters, init='k-means++', max_iter=100, n_init=1, verbose=verbose)
        km.fit(reduced)
        centers = km.cluster_centers_
        y_kmeans = km.predict(reduced)
        if plot:
            ax = plt.figure().add_subplot(111)#, projection='3d')
            ax.scatter(reduced[:, 0], reduced[:, 1], c=y_kmeans,  cmap='viridis')
            ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='x')
            #for idx,name in enumerate(filenames):
            #    ax.annotate(name[3:],reduced[idx])
            plt.show()
        self.y_kmeans = y_kmeans
        return y_kmeans

    def wordcloud(self):
        classes = [[] for x in range(self.clusters)]
        for idx,c in enumerate(self.y_kmeans):
            classes[c].append(idx)

        for i in classes:
            print(i)


    def preprocess(path):
        text = Classifier.file_to_text(path)
        text = text.lower()
        text = strip_accents_unicode(text)
        return text

    def tokenizer(text):
        tokens = word_tokenize(text)
        tokens = filter(lambda x:x.isalpha(),tokens)
        return tokens

    def file_to_text(path):
        filetype = subprocess.run(["file","-b",path], stdout=subprocess.PIPE).stdout.decode('utf-8')
        if filetype.startswith("PDF"):
            cmd = ["pdftotext",path,"-"]
        elif filetype.startswith("HTML"):
            cmd = ["html2text",path]
        else:
            cmd = ['cat']
        result = subprocess.run(cmd, stdout=subprocess.PIPE).stdout
        return result.decode('utf-8')



if __name__ == "__main__":
    a = Classifier()
    a.load()
    #X = a.count("downloads")
    a.classify(clusters=3,plot=True)
    a.save()
    a.wordcloud()




