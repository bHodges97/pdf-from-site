from html.parser import HTMLParser
from urllib.error import HTTPError
from urllib.parse import urlparse
from download import download
import requests
import os
import cgi

class LinkParser(HTMLParser):
    def handle_starttag(self,tag,attr):
        self.link = attr[0][1]

class CSVFinder(HTMLParser):
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
                    link = linkParser.link
                    if link[0] == '/' :
                        link = self.base_url + link
                    contents[1] = link
                self.elements.append(contents)

    def feed(self, data):
        data = "".join(data.splitlines())
        self.data = data
        super().feed(data)
        return self.elements

    def crawl_html(self, url):
        res = requests.get(url)
        parsed_uri = urlparse(url)
        self.base_url = '{uri.scheme}://{uri.netloc}'.format(uri=parsed_uri)
        files = self.feed(res.content.decode("utf-8"))
        for f in files:
            path = f[1].replace("https://hps.vi4io.org/_", "../data/")
            if not os.path.exists(path):
                f[1] = download(f[1])
            else:
                f[1] = path
        files = [(f[1],f[2]) for f in files]
        return files

