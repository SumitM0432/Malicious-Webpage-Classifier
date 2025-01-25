from pywebio.input import *
from pywebio.output import *
from pywebio.platform.flask import webio_view
import pandas as pd
from flask import Flask, redirect, send_from_directory
import webbrowser
import joblib
import whois
from PIL.Image import open
import torch
from torch.utils.data import DataLoader
from config_loader import config
from src import domain_functions
import src.dataset as dataset
from scripts import model_dispatcher
from scripts.preprocessing import data_preprocessing

app = Flask(__name__)

@app.route('/')
def root():
    # Redirect to /maliciousWPC
    return redirect('/maliciousWPC')

def deploy():
    data_dict = input_group("Malicious Webpage Classifier",[

    #url
    input('URL', name = 'url', type = TEXT),

    #geo_loc
    select('Geopgraphical Location', options = [
        'Afghanistan', 'Albania', 'Algeria', 'American Samoa', 'Andorra',
       'Angola', 'Anguilla', 'Antarctica', 'Antigua and Barbuda',
       'Argentina', 'Armenia', 'Aruba', 'Australia', 'Austria',
       'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados',
       'Belarus', 'Belgium', 'Belize', 'Benin', 'Bermuda', 'Bhutan',
       'Bolivia', 'Bonaire, Sint Eustatius, and Saba',
       'Bosnia and Herzegovina', 'Botswana', 'Brazil',
       'British Indian Ocean Territory', 'British Virgin Islands',
       'Brunei', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cabo Verde',
       'Cambodia', 'Cameroon', 'Canada', 'Cayman Islands',
       'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia',
       'Congo Republic', 'Cook Islands', 'Costa Rica', 'Croatia', 'Cuba',
       'Curaçao', 'Cyprus', 'Czechia', 'DR Congo', 'Denmark', 'Djibouti',
       'Dominica', 'Dominican Republic', 'East Timor', 'Ecuador', 'Egypt',
       'El Salvador', 'Equatorial Guinea', 'Estonia', 'Eswatini',
       'Ethiopia', 'Faroe Islands', 'Federated States of Micronesia',
       'Fiji', 'Finland', 'France', 'French Guiana', 'French Polynesia',
       'French Southern Territories', 'Gabon', 'Gambia', 'Georgia',
       'Germany', 'Ghana', 'Gibraltar', 'Greece', 'Greenland', 'Grenada',
       'Guadeloupe', 'Guam', 'Guatemala', 'Guernsey', 'Guinea',
       'Guinea-Bissau', 'Guyana', 'Haiti', 'Hashemite Kingdom of Jordan',
       'Honduras', 'Hong Kong', 'Hungary', 'Iceland', 'India',
       'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Isle of Man', 'Israel',
       'Italy', 'Ivory Coast', 'Jamaica', 'Japan', 'Jersey', 'Kazakhstan',
       'Kenya', 'Kiribati', 'Kosovo', 'Kuwait', 'Kyrgyzstan', 'Laos',
       'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya',
       'Liechtenstein', 'Luxembourg', 'Macao', 'Madagascar', 'Malawi',
       'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands',
       'Martinique', 'Mauritania', 'Mauritius', 'Mayotte', 'Mexico',
       'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique',
       'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands',
       'New Caledonia', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria',
       'Niue', 'North Macedonia', 'Northern Mariana Islands', 'Norway',
       'Oman', 'Pakistan', 'Palau', 'Palestine', 'Panama',
       'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland',
       'Portugal', 'Puerto Rico', 'Qatar', 'Republic of Lithuania',
       'Republic of Moldova', 'Romania', 'Russia', 'Rwanda', 'Réunion',
       'Saint Barthélemy', 'Saint Helena', 'Saint Lucia', 'Saint Martin',
       'Saint Pierre and Miquelon', 'Saint Vincent and the Grenadines',
       'Samoa', 'San Marino', 'Saudi Arabia', 'Senegal', 'Serbia',
       'Seychelles', 'Sierra Leone', 'Singapore', 'Sint Maarten',
       'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia',
       'South Africa', 'South Korea', 'South Sudan', 'Spain', 'Sri Lanka',
       'St Kitts and Nevis', 'Sudan', 'Suriname', 'Sweden', 'Switzerland',
       'Syria', 'São Tomé and Príncipe', 'Taiwan', 'Tajikistan',
       'Tanzania', 'Thailand', 'Togo', 'Tonga', 'Trinidad and Tobago',
       'Tunisia', 'Turkey', 'Turkmenistan', 'Turks and Caicos Islands',
       'Tuvalu', 'U.S. Virgin Islands', 'Uganda', 'Ukraine',
       'United Arab Emirates', 'United Kingdom', 'United States',
       'Uruguay', 'Uzbekistan', 'Vanuatu', 'Vatican City', 'Venezuela',
       'Vietnam', 'Wallis and Futuna', 'Yemen', 'Zambia', 'Zimbabwe',
       'Åland', 'other'], name = 'geo_loc', type = TEXT ),

       # ip_add
       input("IP Address", name = 'ip_add', type = TEXT),

       # tld
       select('Top Level Domain', options = [
        'com', 'org', 'edu', 'net', 'co.uk', 'ca', 'de', 'com.au', 'org.uk',
       'gov', 'ac.uk', 'nl', 'it', 'info', 'fr', 'blogspot.com', 'co.nz', 'ch',
       'se', 'dk', 'other'], name = 'tld', type = TEXT),

        # model
        select('Prediction Model', options = ['XGBoost', 'Decision Tree', 'Logistic Regression', 'Deep Neural Network'], name = 'model', type = TEXT)
    ])

    if data_dict['tld'] == 'other':
        data_dict['tld'] = domain_functions.topld(data_dict['url'])
    
    if data_dict['ip_add'] == '':
        data_dict['ip_add'] = domain_functions.ip_address(data_dict['url'])
    
    if data_dict['geo_loc'] == 'other':
        data_dict['geo_loc'] = domain_functions.geo_location(data_dict['ip_add'])
    
    data_dict['url_len'] = domain_functions.url_len(data_dict['url'])
    data_dict['who_is'] = domain_functions.whois_status(data_dict['ip_add'], data_dict['url'])

    js_len, content = domain_functions.get_content(data_dict['url'])

    data_dict['content'] = content
    
    data_dict['js_len'] = js_len
    
    data_dict['special_char'] = domain_functions.count_special(data_dict['content'])

    data_dict['https'] = domain_functions.http_https(data_dict['url'])

    data_dict['content_len'] = len(data_dict['content'])

    _, net_type = domain_functions.network_type(data_dict['ip_add'])
    data_dict['net_type'] = net_type
    
    data_dict['js_obf_len'] = domain_functions.deobf()

    df = pd.DataFrame([data_dict])
    df['label'] = [0]
    model = df.model.values[0]
    df.drop(columns = ['model'], inplace = True)

    df = data_preprocessing(df)
    ez = {
            'XGBoost':'xg',
            'Decision Tree':'dt',
            'Logistic Regression':'lr',
            'Deep Neural Network':'dt'
         }

    model = ez[model]

    if model in ['xg', 'lr', 'dt']:

        X_test = df.drop(columns = ['label'])
        
        mod = joblib.load(config['OUTPUTS']['MODEL_OUTPUT'] + "/" + str(model) + ".bin")

        predictions = mod.predict(X_test)
    
    elif model == 'dnn':

        model = model_dispatcher.dnn()
        model.to(config['DEVICE'])

        model.load_state_dict(torch.load(config['OUTPUTS']['MODEL_OUTPUT'] + '/dnn.pth'))

        cls = dataset.MaliciousBenignData(df)
        
        df_test = DataLoader(
                            cls,
                            batch_size = 1,
                            num_workers = 0
                            )

        # Putting the model in evaluation mode
        model.eval()

        y_pred = []

        with torch.no_grad():
            for X_test, _ in df_test:
                X_test = X_test.to(config['DEVICE'])

                predictions = model(X_test.float())
                pred = torch.round(torch.sigmoid(predictions))

                y_pred.append(pred.tolist())

            # Changing the Predictions into list 
            y_pred = [int(ele[0][0]) for ele in y_pred]
            predictions = y_pred

    domain = whois.whois(data_dict['url'])

    if predictions[0] == 0:
        status_ = 'Benign'
    else:
        status_ = 'Malicious'
       
    put_row([
    put_column([
        put_code('Webpage Status : '),
        put_code(status_),
    ]), None,
    put_table([
            ['Registrar', domain.registrar],
            ['Creation Date', domain.creation_date],
            ['Domain Organisation', domain.org],
            ['DNS Security', domain.dnssec],
            ['Country', domain.country],
            ['City', domain.city],
            ['State', domain.state],
            ])
    ])
    if status_ == 'Malicious':
        put_row([
            put_image('https://cdn1.iconfinder.com/data/icons/web-development-line-1/64/17-Malicious-Web-512.png')
        ])
    else:
        put_row([
            put_image('https://cdn4.iconfinder.com/data/icons/security-multi-color/128/Security-47-512.png')
        ])

# a = deploy()
# print (a)

app.add_url_rule('/maliciousWPC', 'webio_view', webio_view(deploy),
                methods = ['GET', 'POST', 'OPTIONS'])

if __name__ == '__main__':
    webbrowser.open("http://127.0.0.1:5000/maliciousWPC")
    app.run(host='127.0.0.1', port=5000, threaded=True)


