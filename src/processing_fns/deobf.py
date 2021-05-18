import os
import codecs
import re
os.system('python3 jsado.py html_file.html function_to_hack [nExec:1] [useJB] [useS] [injStart]')

def deobf():
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