from urllib.request import urlopen
from urllib.error import HTTPError
from html.parser import HTMLParser
from collections import defaultdict
import tempfile
import subprocess
import csv
import os.path
import nltk
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


nltk.download('averaged_perceptron_tagger',quiet=True)
nltk.download('wordnet',quiet=True)
nltk.download('stopwords',quiet=True)
nltk.download('punkt',quiet=True)


pdf_stopwords = set(stopwords.words('english'))
pdf_stopwords.update(stopwords.words('german'))
url = "https://hps.vi4io.org/research/publications?csvlist"
base_url = 'https://hps.vi4io.org'

class LinkParser(HTMLParser):
    def handle_starttag(self,tag,attr):
        self.link = attr[0][1]
        if self.link[0] == '/' :
            self.link = base_url + self.link

class PDFFinder(HTMLParser):
    def __init__(self):
        super().__init__()
        self.recording = 0
        self.elements = []

    def handle_starttag(self,tag,attr):
        if tag == "div":
            if attr and attr[0][1] == 'li':
                self.recording = 1
                self.pos = self.getpos()[1]
            elif self.recording > 0:
                self.recording +=1

    def handle_endtag(self,tag):
        if tag == "div" and self.recording > 0:
            self.recording -=1
            if self.recording == 0:
                self.end_pos = self.getpos()[1]
                contents = self.data[self.pos:self.end_pos].replace("<div class=\"li\">","",1).lstrip()
                contents = contents.split(",",2)
                contents[0] = int(contents[0])
                if contents[1]:
                    linkParser = LinkParser()
                    linkParser.feed(contents[1])
                    contents[1] = linkParser.link

                self.elements.append(contents)

    def feed(self, data):
        data = "".join(data.splitlines())
        self.data = data
        super().feed(data)
        return self.elements


html = ""
with urlopen(url) as responce:
    for line in responce:
        html += (line.decode("utf-8"))
publications = PDFFinder().feed(html)

print("Writing papers.csv")
with open("papers.csv","w",encoding='utf-8') as f:
    c = csv.writer(f,quoting=csv.QUOTE_NONNUMERIC)
    for idx,url,html in publications:
        c.writerow([idx,html])

term_frequency = defaultdict(int)
pdf_association = defaultdict(dict)
stemmer = WordNetLemmatizer()

for idx,url,html in publications:
    print(idx, url)
    if len(url) < 1:
        print("Empty")
        continue
    try:
        local_copy = url.replace("https://hps.vi4io.org/_", "../../data/x")
        if os.path.isfile(local_copy):
          print ("using local copy")
          file_name = local_copy
        else:
          doc_name = url.split('/')[-1]
          if os.path.isfile(doc_name):
            print("using cached value")
          else:
            pdf = urlopen(url).read()
            fp = open(doc_name, "bw")
            fp.write(pdf)
            fp.close()
          file_name = doc_name
        print(file_name)
        filetype = subprocess.run(["file","-i",file_name], stdout=subprocess.PIPE).stdout.decode('utf-8')
        if "pdf" not in filetype:
            print("Not pdf",filetype.split(":",1)[1],end='')
            continue
        result = subprocess.run(["pdftotext",file_name,"-"], stdout=subprocess.PIPE).stdout.decode('utf-8')

        words = word_tokenize(result)
        words = [word.lower() for word in words if word.isalpha()]#strip garbage
        t_freq = defaultdict(int)
        tagged = nltk.pos_tag(words)
        for word,tag in tagged:
            if tag == 'NNS':
                word = stemmer.lemmatize(word)
            if tag[:2] == 'NN' and len(word) > 1 and word not in pdf_stopwords:
                t_freq[word] += 1
        for word,count in t_freq.items():
            term_frequency[word]+=count
            pdf_association[word][idx] = count
        del t_freq
        print("Done")

    except HTTPError as err:
        print("Error",err.code)


print("Writing freq_data.csv")
with open("freq_data.csv","w",encoding='utf-8') as f:
    s = sorted(term_frequency.items(), key=lambda kv: kv[1],reverse=True)#[:200]
    c = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
    for word,count in s:
        associated = pdf_association[word]
        if len(associated) > 20:
            associated = sorted(associated.items(), key=lambda kv:kv[1],reverse=True)
            associated = associated[:20]
            associated = {k:v for k,v in associated}
        c.writerow((word,count,associated))#json.dumps(associated)))#using json for parsing in javascript
        if count <= 250:
            break

#print(pdf)
