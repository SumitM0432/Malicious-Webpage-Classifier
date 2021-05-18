import http.client

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
