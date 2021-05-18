import whois
import socket
import requests
from tld import get_tld
from urllib.parse import urlparse
import os
import codecs
import re
from bs4 import BeautifulSoup as bs
import requests
import http.client

def count_special(string):
    count = 0
    for char in string:
        char = str(char)
        if not(char.islower()) and not(char.isupper()) and not(char.isdigit()):
            if char != ' ':
                count += 1
    return count

def network_type(ip):

    ip_str = ip.split(".")
    ip = [int(x) for x in ip_str]
    
    if ip[0]>=0 and ip[0]<=127:
        return (ip_str[0], "A")
    elif ip[0]>=128 and ip[0]<=191:
        return (".".join(ip_str[0:2]), "B")
    else:
        return (".".join(ip_str[0:3]), "C")

def ip_address(url):

    # getting the main domain as gethostbyname gives error for full url
    domain = urlparse(url).netloc
    return socket.gethostbyname(domain)

def topld(url):
    try:
        res = get_tld(url, fix_protocol=True)
        return (res)
    except:
        return ("")

def http_https(url):
    try:
        if 'https://' in url:
            url = url.split('https://')[1]
        else:
            url = url.split('http://')[1]

        conn = http.client.HTTPSConnection(url)
        conn.request("HEAD", "/")
        conn = conn.getresponse()
        
        if conn.status == 200 or conn.status ==  301 or conn.status == 302:
            return ('yes')
    except:
        return ('no')

def whois_domain(url):
    domain = whois.whois(url)

    print ("Registrar -- {}".format(domain.registrar))
    print ("Creation Date -- {}".format(domain.creation_date[0]))
    print ("Domain Organisation -- {}".format(domain.org))
    print ("DNS Security -- {}".format(domain.dnssec))
    print ("Country -- {}".format(domain.country))
    print ("City -- {}".format(domain.city))
    print ("State -- {}".format(domain.state))

def whois_status(ip_add, url):
    try:
        res_ip = whois.whois(ip_add)
        regis_ip = res_ip.registrar

        res_url = whois.whois(url)
        _ = res_url.registrar

        return ("complete")
    except:

        if regis_ip != None:
            return ("complete")
        else:
            return ("incomplete")

def url_len(url):
    return len(str(url))

def geo_location(ip_add):
    response = requests.get("http://ip-api.com/json/" + str(ip_add)).json()
    return (response['country'])

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

#deobf

def deobf():
    os.system('python3 jsado.py html_file.html function_to_hack [nExec:1] [useJB] [useS] [injStart]')

    f1 = codecs.open("html_file.html", 'r')
    html_file = f1.read()

    f2 = codecs.open("t.html", 'r')
    t = f2.read()

    # Finding the JS code in HTML
    js = re.findall(r'>(.*?)</script>', html_file)
    complete_js = "".join(js)
    complete_js.encode('utf-8')

    js2 = re.findall(r'>(.*?)</script>', t)
    complete_js2 = "".join(js2)
    complete_js2.encode('utf-8')

    length = len(complete_js2) - len(complete_js)

    if length <= 0:
        return (0)
    else:
        return (length)
