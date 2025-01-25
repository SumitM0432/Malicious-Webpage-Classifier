from config_loader import config
import pickle
import joblib
import pandas as pd

from src.domain_functions import *

# Loading the encoders
who_is_en = joblib.load(config['OUTPUTS']['ENCODERS'] + "/"+ config['ENCODERS_SCALERS']['ECODER_OBJECTS']['who_is_encoder'])
net_type_en = joblib.load(config['OUTPUTS']['ENCODERS'] + "/"+ config['ENCODERS_SCALERS']['ECODER_OBJECTS']['net_type_encoder'])
tld_en = joblib.load(config['OUTPUTS']['ENCODERS'] + "/"+ config['ENCODERS_SCALERS']['ECODER_OBJECTS']['tld_encoder'])
geo_loc_en = joblib.load(config['OUTPUTS']['ENCODERS'] + "/"+ config['ENCODERS_SCALERS']['ECODER_OBJECTS']['geo_loc_encoder'])
https_en = joblib.load(config['OUTPUTS']['ENCODERS'] + "/"+ config['ENCODERS_SCALERS']['ECODER_OBJECTS']['https_encoder'])

# Loading the scalers
special_char_ss = joblib.load(config['OUTPUTS']['SCALERS'] + "/"+ config['ENCODERS_SCALERS']['SCALER_OBJECTS']['special_char_ss'])
content_len_ss = joblib.load(config['OUTPUTS']['SCALERS'] + "/"+ config['ENCODERS_SCALERS']['SCALER_OBJECTS']['content_len_ss'])

def data_preprocessing(df):
    
    # Network type
    df['Network']= df['ip_add'].apply(lambda x : network_type(x))
    df['net_part'], df['net_type'] = zip(*df.Network)
    df.drop(columns = ['Network'], inplace = True)

    # Counting Special Chars
    df['special_char'] = df['content'].apply(lambda x: count_special(x))

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

    return (df)