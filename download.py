from urllib.error import HTTPError
from urllib.parse import urlparse
from html.parser import HTMLParser
import requests
import cgi
import os
import sys

def download(url, path="./downloads", save_copy=False, filename=None, headers=None, jar=None, redirects=0, redirect_limit=1):
    """
    Downloads pdf from url.
    If url is html page, try to find a download link within.
    """
    if os.path.exists(url):
        return url
    if url == "":
        return ""
    if url[0] == '/':#likely a library link
        return ""
    if headers == None:
        headers = {'User-Agent':'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:68.0) Gecko/20100101 Firefox/68.0'}
    if jar == None:
        jar = requests.cookies.RequestsCookieJar()
    if not os.path.exists(path):
        os.makedirs(path)

    try:
        res = requests.head(url, headers=headers, cookies=jar, allow_redirects=True)
        if 'Content-Type' in res.headers:
            content_type, encoding = filetype(res.headers['Content-Type'])
        else:
            print("No content-type specified",res.headers.keys())
            return ""
        #if link is html crawl page for pdf link
        if content_type == "html" and redirects < redirect_limit:
            res = requests.get(url, headers=headers, cookies=jar)
            if not encoding:#assume ascii and strip non ascii bits
                html =res.content.decode('ascii', 'ignore')
            else:
                html = res.content.decode(encoding, 'ignore')
            pdfurl = findpdflink(html, url)
            return download(pdfurl, headers=headers, jar=jar, path=path, filename=filename, save_copy=save_copy, redirects=redirects+1, redirect_limit=redirect_limit)
        #guess a file type
        if filename == None:
            #get filename from request header
            if 'Content-Disposition' in res.headers:
                header = res.headers['Content-Disposition']
                filename = cgi.parse_header(header)[1]['filename']
            #if none avaliable take the url end
            else:
                filename = url.rsplit('/',1)[-1]
        file_path = f"{path}/{filename}"

        exists = os.path.isfile(file_path)
        if exists and not save_copy:
            print("Exists:", file_path)
            return file_path
        res = requests.get(url, headers=headers, cookies=jar)
        if exists and save_copy:#give new name
            i=0
            while os.path.exists(f"{path}/{filename}.{i}"):
                i += 1
            file_path = f"{path}/{filename}.{i}"
        print("Downloaded to:",file_path)
        with open(file_path, "wb") as fp:
            fp.write(res.content)
        return file_path
    except HTTPError as err:
        print("Error", err.code)
    except (ConnectionError, requests.exceptions.TooManyRedirects) as err:
        print(err)
    except:
        print("Unexpected error:", sys.exc_info()[0])

    return ""

def findpdflink(html, url):
    finder = PDFFinder()
    finder.feed(html)
    pdfurl = finder.pdflink()
    if pdfurl == "":
        pass
    elif pdfurl[0] == "/":
        parsed_uri = urlparse(url)
        pdfurl = '{uri.scheme}://{uri.netloc}/{pdfurl}'.format(uri=parsed_uri,pdfurl=pdfurl)
    return pdfurl

def filetype(header):
    ctype,encoding = cgi.parse_header(header)
    ctype = ctype.split("/")[1]
    if encoding:
        encoding = encoding['charset']
    return ctype,encoding

class LibraryLink(HTMLParser):
    def handle_starttag(self,tag,attr):
        if tag == "a" and attr[0] == ("id","onClickExclude"):
            self.link = attr[1][1]

class PDFFinder(HTMLParser):
    def __init__(self):
        super().__init__()
        self.pdflist = set()

    def handle_starttag(self, tag, attr):
        self.tag = tag
        self.attr = attr
        url = False
        pdf = False
        if tag == "a":
            for name,val in attr:
                if name == "href" and val:
                    url = val
                    if ".pdf" == val.lower()[-4:]:
                        self.pdflist.add(url)
                        return
                if name == "title" and "pdf" in val.lower():
                    pdf = True
            if url and pdf:
                self.pdflist.add(url)
        elif tag == 'iframe':
            for name,val in attr:
                if name == "src":# and ".pdf" in val.lower():
                    self.pdflist.add(val)

    def handle_data(self,data):
        if data == "here" and self.tag == "a": #click here to redirect
            for name,val in self.attr:
                if name == "href":
                    self.pdflist.add(val)


    def pdflink(self):
        words = ["epdf","supplement","google","search"]
        pdfs = [x for x in self.pdflist if all([y not in x.lower() for y in words])]
        pdfs = [x for x in pdfs if x[:2] != "//"]

        if len(pdfs) == 0:
            return ""
        elif len(pdfs) == 1:
            return pdfs[0]
        else:
            print("Can't guess correct link", pdfs)
            return ""

if __name__ == "__main__":
    #test download
    url = "http://archiv.ub.uni-heidelberg.de/volltextserver/volltexte/2006/6330/pdf/PerformanceAnalysis.pdf"
    download(url)

