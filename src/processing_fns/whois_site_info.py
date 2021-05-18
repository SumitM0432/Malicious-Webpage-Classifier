import whois

def whois_domain(url):

    domain = whois.whois(url)

    print ("Registrar -- {}".format(domain.registrar))
    print ("Creation Date -- {}".format(domain.creation_date[0]))
    print ("Domain Organisation -- {}".format(domain.org))
    print ("DNS Security -- {}".format(domain.dnssec))
    print ("Country -- {}".format(domain.country))
    print ("City -- {}".format(domain.city))
    print ("State -- {}".format(domain.state))

# whois_domain("http://www.ddj.com")
# whois_domain("http://www.gamestats.com")
