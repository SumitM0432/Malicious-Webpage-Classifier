from bs4 import BeautifulSoup as bs
import requests
from lxml import html
import re
import warnings
warnings.filterwarnings('ignore')


def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

def get_content(url):
    page = requests.get(url, stream = True)
    soup = bs(page.content, features="html")
    text = soup.get_text()
    # print (soup)

    with open('html_file.html', 'w') as outFile:
        outFile.write(str(page.content))

    # Finding the JS code in HTML
    js = re.findall(r'<script(.*?)>(.*?)</script>', str(soup))
    complete_js = ""
    for i in range(len(js)):
        complete_js = complete_js + js[i][0]

    # Cleaning HTML
    text = cleanhtml(text)
    
    # Removing extra whitespaces
    content = re.sub(' +', ' ', text)

    content = content.encode('utf-8')
    complete_js_len = len(complete_js.encode('utf-8'))/1000

    return complete_js_len, content









