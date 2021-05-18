def network_type(ip):

    ip_str = ip.split(".")
    ip = [int(x) for x in ip_str]
    
    if ip[0]>=0 and ip[0]<=127:
        return (ip_str[0], "A")
    elif ip[0]>=128 and ip[0]<=191:
        return (".".join(ip_str[0:2]), "B")
    else:
        return (".".join(ip_str[0:3]), "C")

# a = '75.4.10.4'
# d, e = network_type(a)
# print (d, e)
