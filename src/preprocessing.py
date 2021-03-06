import config
import pandas as pd
import preprocfunctions
import pickle

pkl_file = open(config.who_is_en, 'rb')
who_is_en = pickle.load(pkl_file) 

pkl_file = open(config.net_type_en, 'rb')
net_type_en = pickle.load(pkl_file) 

pkl_file = open(config.tld_en, 'rb')
tld_en = pickle.load(pkl_file) 

pkl_file = open(config.geo_loc_en, 'rb')
geo_loc_en = pickle.load(pkl_file)

pkl_file = open(config.https_en, 'rb')
https_en = pickle.load(pkl_file)

pkl_file = open(config.special_char_ss, 'rb')
special_char_ss = pickle.load(pkl_file)

pkl_file = open(config.content_len_ss, 'rb')
content_len_ss = pickle.load(pkl_file)

def preprocessing(df):
    
    # Network type
    df['Network']= df['ip_add'].apply(lambda x : preprocfunctions.network_type(x))
    df['net_part'], df['net_type'] = zip(*df.Network)
    df.drop(columns = ['Network'], inplace = True)

    # Counting Special Chars
    df['special_char'] = df['content'].apply(lambda x: preprocfunctions.count_special(x))

    # Content len
    df["content_len"] = df['content'].apply(lambda x: len(x))

    # Labels
    df.label.replace({'good' : 0, 'bad' : 1}, inplace = True)

    # Label Encoding
    df['tld'] = tld_en.transform(df['tld'])
    df['https'] = https_en.transform(df['https'])
    df['geo_loc'] = geo_loc_en.transform(df['geo_loc'])
    df['who_is'] = who_is_en.transform(df['who_is'])
    df['net_type'] = net_type_en.transform(df['net_type'])

    # Standard Scaler
    con = content_len_ss.transform(df['content_len'].values.reshape(-1, 1))
    spec = special_char_ss.transform(df['special_char'].values.reshape(-1, 1))

    df['content_len'] = pd.DataFrame(con, index = df.index, columns = ['content_len'])
    df['special_char'] = pd.DataFrame(spec, index = df.index, columns = ['special_char'])

    df = df[['url_len', 'geo_loc', 'tld', 'who_is', 'https', 'js_len', 'js_obf_len', 'label', 'net_type', 'special_char', 'content_len']]

    return df
