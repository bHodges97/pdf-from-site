# PDF  Word Frequency
Produced as part of an internship under [Julian Kunkel](https://hps.vi4io.org/about/people/julian_kunkel)

### pdffreq.py
Converts a lists of files to a term frequency matrix.

**Outputs:**

 - **freq_data.csv**: word, freq 
 - **related_papers.csv**: word, json id:key 
 - **papers.csv**: id, HTML, sha256 hash
 - **tfs.npz**: vocab list and scipy sparse row matrix, shape (row,words)

### crawler.py
Crawls google scholar for papers and download them.
**Outputs:**
  - **out.csv** bib entries for each paper

### csvfinder.py
Crawls pdfs from vi4io

### classifier.py
Use k-means to cluster documents

### download.py
Downloads from url. If url points to html page, then search for pdf links in page.



## Usage:
```
from pdffreq import PDFFreq
url = "https://hps.vi4io.org/research/publications?csvlist"
words = ["et","al","example","kunkel","see","figure","limitless","per"]
pdfFreq = PDFFreq(words,find_termfreq=False,find_collocations=True)
pdfFreq.load_csv()
files = CSVFinder().crawl_html(url)
for url,html in files:
  pdfFreq.add_pdf(url,html)
pdfFreq.save_csv(max_count=1000)

```
