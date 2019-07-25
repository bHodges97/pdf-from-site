from urllib.request import urlopen
from urllib.error import HTTPError
from urllib.parse import urlparse
from html.parser import HTMLParser
from collections import defaultdict
from os import path
from hashlib import sha256
import json
import subprocess
import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from pdffinder import PDFFinder


nltk.download('averaged_perceptron_tagger',quiet=True)
nltk.download('wordnet',quiet=True)
nltk.download('stopwords',quiet=True)
nltk.download('punkt',quiet=True)


class PDFFreq():
    def __init__(self, exclude = []):
        self.pdf_stopwords = set(stopwords.words('english'))
        self.pdf_stopwords.update(stopwords.words('german'))
        self.pdf_stopwords.update(exclude)#add additional stop words here

        self.term_frequency = defaultdict(int)
        self.pdf_association = defaultdict(dict)
        self.stemmer = WordNetLemmatizer()
        self.pdfs = []
        self.hashes = set()
        self._nextid = 1


    def crawl_html(self, url):
        with urlopen(url) as responce:
            html = responce.read().decode('utf-8')
        parsed_uri = urlparse(url)
        root = '{uri.scheme}://{uri.netloc}'.format(uri=parsed_uri)
        files = PDFFinder(root).feed(html)

        for idx,url,html in files:
            self.add_pdf(url,html)

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
                return

        #pdf to t_freq
        t_freq = defaultdict(int)
        result = subprocess.run(["pdftotext",file_path,"-"], stdout=subprocess.PIPE).stdout.decode('utf-8')
        words = word_tokenize(result)
        words = [word.lower() for word in words if word.isalpha()]#strip garbage
        tagged = nltk.pos_tag(words)
        for word,tag in tagged:
            if tag == 'NNS': #If plural , make singular
                word = self.stemmer.lemmatize(word)
            if tag[:2] == 'NN' and len(word) > 1 and word not in self.pdf_stopwords: # word is noun and not stop word
                t_freq[word] += 1

        #save
        idx = self._nextid
        self._nextid+=1
        self.hashes.add(phash)
        self.pdfs.append([idx,html,phash])
        for word,count in t_freq.items():
            self.term_frequency[word]+=count
            self.pdf_association[word][idx] = count
        del t_freq
        print("Success")

    def start(self):
        for idx,url,html in publications:
            print(idx, url)
            if len(url) < 1:
                print("Empty")
                continue

    def load_csv(self):
        try:
            with open("freq_data.csv","r",encoding='utf-8') as f:
                reader = csv.reader(f)
                for word,count in reader:
                    self.term_frequency[word] = int(count)

            with open("related_papers.csv","r",encoding='utf-8') as f:
                reader = csv.reader(f)
                for word,related in reader:
                    related = json.loads(related)
                    self.pdf_association[word] = {int(k):v for k,v in related.items()}

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


    def save_csv(self, max_count = 1000):
        ordered = list(sorted(self.term_frequency.items(), key=lambda kv: kv[1], reverse=True))
        if len(ordered) > max_count:
            ordered = ordered[:max_count]

        print("Writing freq_data.csv")
        with open("freq_data.csv","w",encoding='utf-8') as f:
            c = csv.writer(f, quoting=csv.QUOTE_NONE)
            for row in ordered:
                c.writerow(row)

        print("Writing related_papers.csv")
        with open("related_papers.csv","w",encoding='utf-8') as f:
            c = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            for word,count in ordered:
                associated = self.pdf_association[word]
                associated = sorted(associated.items(), key=lambda kv:kv[1], reverse=True)
                if len(associated) > 20:
                    associated = associated[:20]
                associated = {k:v for k,v in associated}#format as dictionary
                c.writerow((word,json.dumps(associated)))

        print("Writing papers.csv")
        with open("papers.csv","w",encoding='utf-8') as f:
            c = csv.writer(f,quoting=csv.QUOTE_NONNUMERIC)
            for idx,html,phash in self.pdfs:
                c.writerow([idx,html,phash])

    def get_conflict(self, phash):
        for idx,_,h in self.pdfs:
            if phash == h:
                return idx


if __name__ == "__main__":
    url = "https://hps.vi4io.org/research/publications?csvlist"
    pdfFreq =  PDFFreq([])
    pdfFreq.load_csv()
    pdfFreq.crawl_html(url)
    pdfFreq.save_csv()



#print(pdf)
