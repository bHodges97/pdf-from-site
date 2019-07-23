from html.parser import HTMLParser

class LinkParser(HTMLParser):
    def handle_starttag(self,tag,attr):
        self.link = attr[0][1]

class PDFFinder(HTMLParser):
    def __init__(self,base_url):
        super().__init__()
        self.recording = 0
        self.elements = []
        self.base_url = base_url

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

