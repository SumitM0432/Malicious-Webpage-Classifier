import tld
from tld import get_tld

def topld(url):
    res = get_tld(url, fail_silently = True)
    print (res)

# topld("https://www.google.co.uk")