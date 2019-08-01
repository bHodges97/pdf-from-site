# PDF  Word Frequency

Grabs all pdf files from a url and collections word frequencies.

Outputs:

freq_data.csv: word, freq
papers.csv: id, HTML, sha256 hash
related_papers.csv: word, json id:key dict  

Usage:
```
from pdfcrawler import PDFFreq
pdfFreq =  PDFFreq(["words", "to", "be", "excluded"])
pdfFreq.load_csv() #if csv exists load them
url = "https://hps.vi4io.org/research/publications?csvlist"
files = pdfFreq.crawl_html(url) #add pdfs from a page
for _,url,html in files:
  pdfFreq.add(url,html) #add a specific pdf
pdfFreq.save_csv(max_count=10000)# save top max_count words to file
```
