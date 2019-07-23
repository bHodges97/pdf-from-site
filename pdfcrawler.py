from urllib.request import urlopen
from urllib.error import HTTPError
from html.parser import HTMLParser
from collections import defaultdict
from os import path
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


pdf_stopwords = set(stopwords.words('english'))
pdf_stopwords.update(stopwords.words('german'))
pdf_stopwords.update(['slide'])#add additional stop words here

max_count = 500 #Only keep track of top 500 words

url = "https://hps.vi4io.org/research/publications?csvlist"
base_url = 'https://hps.vi4io.org'

term_frequency = defaultdict(int)
pdf_association = defaultdict(dict)
stemmer = WordNetLemmatizer()

def download(url, file_path):
    try:
        pdf = urlopen(url).read()
        with open(file_path, "bw") as fp:
            fp.write(pdf)
    except HTTPError as err:
        return err.code
    return 200

with urlopen(url) as responce:
    html = responce.read().decode('utf-8')
publications = PDFFinder(base_url).feed(html)

print("Writing papers.csv")
with open("papers.csv","w",encoding='utf-8') as f:
    c = csv.writer(f,quoting=csv.QUOTE_NONNUMERIC)
    lastidx = 0
    for idx,url,html in publications:
        while lastidx+1 != idx:
            lastidx+=1
            c.writerow([lastidx,None])
        c.writerow([idx,html])
        lastidx+=1


for idx,url,html in publications:
    print(idx, url)
    if len(url) < 1:
        print("Empty")
        continue

    file_path = url.replace("https://hps.vi4io.org/_", "../../data/")
    if path.isfile(file_path):
        print("Using local copy")
    else:
        file_path = url.split('/')[-1]
        if path.isfile(file_path):
            print("Using cached value")
        else:
            print("Downloading pdf")
            status = download(url, file_path)
            if status != 200:
                print("Error", status)
                continue

    filetype = subprocess.run(["file","-i",file_path], stdout=subprocess.PIPE).stdout.decode('utf-8').split(":",1)[1]
    if "pdf" not in filetype:
        print("Not pdf",filetype,end='')
        continue
    result = subprocess.run(["pdftotext",file_path,"-"], stdout=subprocess.PIPE).stdout.decode('utf-8')

    words = word_tokenize(result)
    words = [word.lower() for word in words if word.isalpha()]#strip garbage
    t_freq = defaultdict(int)
    tagged = nltk.pos_tag(words)
    for word,tag in tagged:
        if tag == 'NNS': #If plural , make singular
            word = stemmer.lemmatize(word)
        if tag[:2] == 'NN' and len(word) > 1 and word not in pdf_stopwords: # word is noun and not stop word
            t_freq[word] += 1
    for word,count in t_freq.items():
        term_frequency[word]+=count
        pdf_association[word][idx] = count
    del t_freq
    print("Done")


ordered = list(sorted(term_frequency.items(), key=lambda kv: kv[1], reverse=True))
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
        associated = pdf_association[word]
        if len(associated) > 20:
            associated = sorted(associated.items(), key=lambda kv:kv[1], reverse=True)
            associated = associated[:20]
            associated = {k:v for k,v in associated}#format as dictionary
        c.writerow((word,json.dumps(associated)))


#print(pdf)
