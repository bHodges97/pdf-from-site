from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from savenpz import save_npz,load_npz

class Classifier():
    def __init__(self):
        self.tf = None

    def _tfidf_transform(self,tf):
        tfidfTransformer = TfidfTransformer()
        self.tfidf = tfidfTransformer.fit_transform(tf)
        return self.tfidf

    def load(self, path="."):
        self.tf = load_npz(f"{path}/tfs.npz")[0]
        return self.tf

    def save(self, path="."):
        if self.tf == None:
            print("No term frequencies loaded")
            return
        save_npz(f'{path}/tfs.npz', self.tf, ["none"])

    def classify(self, clusters=None, verbose=False, plot=False):
        self.tfidf = self._tfidf_transform(self.tf)
        self.clusters = clusters
        svd = TruncatedSVD(n_components=2)
        reduced = svd.fit_transform(self.tfidf)

        if not clusters:
            km = self.guess_k(reduced)
        else:
            #km = KMeans(n_clusters=clusters, init='k-means++', max_iter=100, n_init=1, verbose=verbose)
            km = MiniBatchKMeans(n_clusters=clusters, verbose=verbose)
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

    def guess_k(self, x, kmin=2,kmax=6):
        score = 0
        km = None
        for k in range(kmin, kmax+1):
            print(k)
            kmeans = MiniBatchKMeans(n_clusters=k).fit(x)
            print(kmeans.labels_)
            newscore = silhouette_score(x, kmeans.labels_, metric = 'euclidean')
            print(newscore)
            if newscore > score:
                score = newscore
                km = kmeans
        return km

    def wordcloud(self):#TODO
        classes = [[] for x in range(self.clusters)]
        for idx,c in enumerate(self.y_kmeans):
            classes[c].append(idx)

        for i in classes:
            print(i)


if __name__ == "__main__":
    a = Classifier()
    a.load()
    #X = a.count("downloads")
    a.classify(plot=True)
    #a.wordcloud()




