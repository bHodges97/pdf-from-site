from urllib.error import HTTPError
from urllib.parse import urlparse
import requests
import cgi
import os
from pdffinder import PDFFinder

def download(url, headers=None, jar=None, path="./downloads", redirects=0, redirect_limit=1):
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
        res = requests.get(url, headers=headers, cookies=jar)
        content_type, encoding = filetype(res.headers['Content-Type'])
        if content_type == "html" and redirects < redirect_limit:
            html = res.content.decode(encoding)
            pdfurl = findpdflink(html, url)
            return download(pdfurl, headers=headers, jar=jar, path=path, redirects=redirects+1, redirect_limit=redirect_limit)

        #save to file
        if 'Content-Disposition' in res.headers:
            header = res.headers['Content-Disposition']
            filename = cgi.parse_header(header)[1]['filename']
        else:#html pages may not have a file name, so guess from url
            filename = url.rsplit('/',1)[-1]
        file_path = f"{path}/{filename}"

        if os.path.isfile(file_path):
            print("Exists:", file_path)
            return file_path
        print("Downloading:",url)
        with open(file_path, "wb") as fp:
            fp.write(res.content)
        return file_path
    except HTTPError as err:
        print("Error", err.code)
    except (ConnectionError, requests.exceptions.TooManyRedirects) as err:
        print(err)
    return ""

def findpdflink(html, url):
    finder = PDFFinder()
    finder.feed(html)
    pdfurl = finder.pdflink()
    if pdfurl == "":
        pass
    elif pdfurl[0] == "/":
        parsed_uri = urlparse(url)
        pdfurl = '{uri.scheme}://{uri.netloc}/{pdfurl}'.format(uri=parsed_uri)
    return pdfurl

def filetype(header):
    ctype,encoding = cgi.parse_header(header)
    ctype = ctype.split("/")[1]
    if encoding:
        encoding = encoding['charset']
    return ctype,encoding

if __name__ == "__main__":
    #test download
    url = "http://archiv.ub.uni-heidelberg.de/volltextserver/volltexte/2006/6330/pdf/PerformanceAnalysis.pdf"
    download(url)

