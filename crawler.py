import scholarly
import csv
import http.cookiejar
from urllib.error import HTTPError
from pdffinder import *
from urllib.parse import urlparse
from requests.exceptions import TooManyRedirects
import requests




class Crawler():
    def __init__(self):
        #pretend to be browser to avoid 403 error
        self.headers = {'User-Agent':'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:68.0) Gecko/20100101 Firefox/68.0'}
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

    def download(self):
        self.jar = requests.cookies.RequestsCookieJar()
        with open(self.outfile) as csvfile:
            reader = csv.DictReader(csvfile)
            for idx,row in enumerate(reader):
                print(row['title'])
                url = row['eprint']
                if url[0] == '/':
                    #res = requests.get("https://scholar.google.co.uk"+url,headers=self.headers,cookies=self.jar)
                    #rg = LibraryLink()
                    #rg.feed(ascii(res.content))
                    #url = "http://zp2yn2et6f.scholar.serialssolutions.com/" + rg.link[1:]
                    #print("Library link")
                    continue
                parsed_uri = urlparse(url)
                root = '{uri.scheme}://{uri.netloc}'.format(uri=parsed_uri)
                file_path = f"downloads/{idx:05d}"
                try:
                    res = requests.get(url,headers=self.headers,cookies=self.jar) # set stream=True for chunk by chunk
                    content_type = Crawler.filetype(res.headers['content-Type'])
                    if "html" == content_type:
                        res = self.html_to_pdf(res, root)
                    if res == None:
                        print("Warning: failed to find PDF, skipping download",content_type)
                        continue
                    extension = Crawler.filetype(res.headers['content-Type'])
                    file_path = f"{file_path}.{extension}"
                    print("Downloading:",url)
                    with open(file_path, "wb") as fp:
                        fp.write(res.content)
                except HTTPError as err:
                    print("Error", err.code)
                except (ConnectionError, TooManyRedirects) as err:
                    print(err)

    def html_to_pdf(self,res,root,depth=0):
        #find pdf link in page
        html = ascii(res.content) #.decode("utf-8") not all pages are unicode
        finder = PDFFinder()
        finder.feed(html)
        url = finder.pdflink()
        if url == "":
            return None
        elif url[0] == "/":
            url = root+url

        #try to download pdf
        res = requests.get(url,headers=self.headers,cookies=self.jar)
        content_type = Crawler.filetype(res.headers['content-Type'])
        if "html" == content_type:#link might be a redirect, search the page for link
            if depth < 1:
                print("Warning: PDF Link is redirect! retrying")
                with open("redirect.html",'wb') as f:
                    f.write(res.content)
                return self.html_to_pdf(res,root,depth=1) #try again, some pages go to a redirect link
            else:
                return None
        else:
            return res

    def filetype(content_type):
        return content_type.split(";")[0].split("/")[1].lower()




if __name__ == "__main__":
    c = Crawler()
    #c.query("hpc")
    c.download()
