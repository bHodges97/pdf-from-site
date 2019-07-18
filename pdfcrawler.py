from urllib.request import urlopen
from urllib.error import HTTPError
from html.parser import HTMLParser
from collections import defaultdict
import tempfile
import subprocess
import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class PDFFinder(HTMLParser):
    def __init__(self):
        super().__init__()
        self.pdflist = []

    def handle_starttag(self, tag, attr):
        self.tag = tag
        self.attr = attr

    def handle_data(self,data):
        if data == "PDF":
            self.pdflist.append(self.attr[0][1])

url = "https://hps.vi4io.org/research/publications"
base_url = 'https://hps.vi4io.org'

sourcepage = urlopen(url).read()
finder = PDFFinder()
finder.feed(str(sourcepage))
pdflist = [base_url + x if x[0] == '/' else x for x in finder.pdflist]
term_frequency = defaultdict(int)
document_frequency = []


print(len(pdflist),"pdfs found.")
for url in pdflist:
    try:
        pdf = urlopen(url).read()
        doc_name = url.split('/')[-1]
        with tempfile.NamedTemporaryFile() as fp:
            fp.write(pdf)
            result = subprocess.run(["pdftotext",fp.name,"-"], stdout=subprocess.PIPE).stdout.decode('utf-8')
        words = word_tokenize(result)
        words = [word.lower() for word in words if word.isalpha()]#strip garbage
        t_freq = defaultdict(int)
        tagged = nltk.pos_tag(words)
        for word,tag in tagged:
            if tag[:2] == 'NN' and len(word) > 1 and word not in stopwords:
                t_freq[word] += 1
        for word,count in t_freq.items():
            term_frequency[word]+=count
        document_frequency.append((doc_name,t_freq))
        print(doc_name,"done")

    except HTTPError as err:
        print("error",err.code,url)

out = (document_frequency,term_frequency)

with open("freq_data.csv","w") as f:
    s = sorted(term_frequency.items(), key=lambda kv: kv[1])
    c = csv.writer(f, quoting=csv.QUOTE_ALL)
    for i in s:
        c.writerow(i)


#print(pdf)
