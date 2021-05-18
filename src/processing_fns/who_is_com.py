import whois

def whois_status(ip_add, url):
    try:
        res_ip = whois.whois(ip_add)
        regis_ip = res_ip.registrar

        res_url = whois.whois(url)
        _ = res_url.registrar

        print ("Complete")
    except:

        if regis_ip != None:
            print ("Complete")
        else:
            print ("incomplete")


# whois_status('147.22.38.45', 'http://www.ff-b2b.de/')
