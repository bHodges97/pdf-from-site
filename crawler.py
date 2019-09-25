import scholarly
import csv
from download import download
from pdffreq import PDFFreq


class Crawler():
    """
    Crawls google scholar for pdfs
    Use download_all to download the pdfs
    """
    def query(self,query,path="out.csv",limit = 50):
        search_query = scholarly.search_pubs_query(query)
        count = 0
        pubs = []
        self.files = []
        fieldnames = ['abstract','author','eprint','title','url']
        with open(self.outfile,"w") as f:
            f.write(",".join(fieldnames) + "\n")
        for x in search_query:
            if 'eprint' in x.bib:
                with open(path,"a") as f:
                    print("Found ",x.bib["title"])
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(x.bib)
                count+=1
                if count == limit:
                    break

    def download_all(self,path="out.csv",out="./downloads"):
        files = []
        with open(path) as csvfile:
            reader = csv.DictReader(csvfile)
            for idx,row in enumerate(reader):
                print(row['title'])
                url = row['eprint']
                out = download(url,path=out,save_copy=True)
                files.append((out,row['title']))

        return files


if __name__ == "__main__":
    c = Crawler()
    #c.query("hpc",limit=1000)
    files = c.download_all()
    pdfFreq = PDFFreq()
    #files = pdfFreq.list_directory("./downloads")
    pdfFreq.load()
    for url,html in files:
        pdfFreq.add_pdf(url,html)
    pdfFreq.save()
