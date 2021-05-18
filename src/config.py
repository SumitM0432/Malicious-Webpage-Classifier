import torch

BATCH_SIZE = 32
EPOCHS = 2
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-5

path = '' # add your current folder path

TRAINING_FILE = path + 'input/Webpages_Classification_train_data.csv'
TESTING_FILE = path + 'input/Webpages_Classification_test_data.csv'
MODEL_OUTPUT = path + 'models/'

who_is_en = path + 'le_ss/who_is_encoder.pkl'
geo_loc_en = path + 'le_ss/geo_loc_encoder.pkl'
https_en = path + 'le_ss/https_encoder.pkl'
net_type_en = path + 'le_ss/net_type_encoder.pkl'
tld_en = path + 'le_ss/tld_encoder.pkl'

special_char_ss = path + 'le_ss/special_char_ss.pkl'
content_len_ss = path + 'le_ss/content_len_ss.pkl'
