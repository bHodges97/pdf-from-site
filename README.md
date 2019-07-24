# PDF  Word Frequency

Grabs all pdf files from a url and collections word frequencies.

Outputs to freq_data.csv , papers.csv

Usage:
```
from pdfcrawler import PDFFreq
url = "https://hps.vi4io.org/research/publications?csvlist"
pdfurl,html = ""
pdfFreq =  PDFFreq(["words", "to", "be", "excluded"]) 
pdfFreq.load_csv() #if csv exists load them
pdfFreq.crawl_html(url) #add pdfs from a page
pdfFreq.add(pdfurl,html) #add a specific pdf 
pdfFreq.save_csv(max_count=1000)# save top max_count words to file
```


