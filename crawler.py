import scholarly
import csv
from download import download


class Crawler():
    """
    Crawls google scholar for pdfs
    Use download_all to download the pdfs
    """
    def __init__(self):
        #pretend to be browser to avoid 403 error
        self.outfile = "out.csv"

    def query(self,query,limit = 50):
        search_query = scholarly.search_pubs_query(query)
        count = 0
        pubs = []
        self.files = []
        fieldnames = ['abstract','author','eprint','title','url']
        with open(self.outfile,"w") as f:
            f.write(",".join(fieldnames) + "\n")
        for x in search_query:
            if 'eprint' in x.bib:
                with open(self.outfile,"a") as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(x.bib)
                count+=1
                if count == limit:
                    break

    def download_all(self):
        with open(self.outfile) as csvfile:
            reader = csv.DictReader(csvfile)
            for idx,row in enumerate(reader):
                print(row['title'])
                url = row['eprint']
                download(url,file_name=str(idx))


if __name__ == "__main__":
    c = Crawler()
    #c.query("hpc")
    c.download()
