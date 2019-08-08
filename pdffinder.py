from html.parser import HTMLParser

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
                if name == "href":
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
        words = ["epdf","supplement","google"]
        pdfs = [x for x in self.pdflist if all([y not in x.lower() for y in words])]
        pdfs = [x for x in pdfs if x[:2] != "//"]

        if len(pdfs) > 1:
            pdfs = [x for x in pdfs if "search" not in x]
        if len(pdfs) == 0:
            return ""
        elif len(pdfs) == 1:
            return pdfs[0]
        else:
            print("Can't guess correct link", pdfs)
            return ""
