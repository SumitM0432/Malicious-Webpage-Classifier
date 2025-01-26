# Malicious-Webpage-Classifier

## Objective
Malicious web pages are designed to install malware on your system, disrupt computer operations, and, in many cases, steal personal information. Classifying these web pages is crucial for enhancing user safety and providing a secure browsing experience.

This project aims to classify web pages into *Malicious[Bad]* and *Benign[Good]*. Through extensive Exploratory Data Analysis (EDA) and Geospatial Data Analysis, valuable insights were derived to understand the data better. The dataset underwent feature engineering and preprocessing to ensure the optimal performance of the models. Three Machine learning and one Deep Learning model are trained. The models are **XGBoost**, **Logistic Regression**, **Decision Tree** and **Deep Neural Network**. The Deep Neural Network is implemented in PyTorch and the others are implemented using scikit-learn.

## Dataset
The data set is taken from [Mendeley Data](https://data.mendeley.com/datasets/gdx3pkwp47/2). The dataset contains features like the raw webpage content, geographical location, javascript length, obfuscated JavaScript code of the webpage etc. The Dataset contains around 1.5 million web pages. A description of the whole dataset is provided at the link provided.

## File Structure
```
├── config
│   └── config.yaml
├── data
│   ├── dataset.txt
│   └── tableconvert_csv_pkcsig.csv
├── deployment
│   ├── config_loader.py
│   └── deployment.py
├── notebooks
│   ├── Exploratory Data Analysis.ipynb
│   └── Modelling.ipynb
├── output
│   ├── encoders
│   ├── models
│   └── scalers
├── scripts
│   ├── __init__.py
│   ├── config_loader.py
│   ├── model_dispatcher.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
├── src
│   ├── __init__.py
│   ├── config_loader.py
│   ├── cross_val.py
│   ├── dataset.py
│   ├── default_accuracy.py
│   ├── domain_functions.py
│   ├── dumper.js
│   ├── eval_metrics.py
└── └── jsado.py
```

## Files
- **config**: Configuration file to config the whole project </br>
	* config.yaml - Configuration file

- **data**: Contains the input files for the project
  * tableconvert_csv_pkcsig.csv - Contains the iso alpha3 code for the countries
  * dataset.txt - Link to download the dataset and paste it into the input folder
 
- **deployment**: Contains the deployment code for the project
  * config_loader.py - code to ingest the configs in the files
  * deployment.py - deployment code on localhost using PyWebIO and Flask

- **output/encoders and scalers**: Contains all the saved Label Encoder and Standard Scaler files for preprocessing </br>
	* content_len_ss.pkl - Standard Scaler for content len
	* geo_loc_encoder.pkl - Label Encoder for geolocation
	* https_encoder.pkl - Label Encoder for HTTPS features
	* net_type_encoder.pkl - Label Encoder for the network type
	* special_char_ss.pkl - Standard Scaler for Special Char length
	* tld_encoder.pkl - Label Encoder for Top-Level Domain
	* who_is_encoder.pkl - Label Encoder for who_is status

- **output/models**: Contains all the trained models [DNN, LR, DT, XG] 

- **notebooks**: Contain all the notebooks -- [Kaggle Notebook](https://www.kaggle.com/sumitm004/malicious-webpage-classifier-using-dnn-pytorch)



- **scripts**: Contains all the code used for the main scripts </br>
	* config_loader.py - code to ingest the configs in the files
	* model_dispatcher.py - Contains the ML and DL models
	* predict.py - Python file to make a prediction
	* preprocessing.py - Python file for preprocessing the dataset for training and testing
	* train.py - Main run file

- **src**: Contains all the code used for the main scripts </br>
	* config_loader.py - code to ingest the configs in the files
	* eval_metrics.py - Evaluation Metrics for the training and testing
	* cross_val.py - Crossvalidation code [StratifiedKFold]
	* dataset.py - Contains the code for a custom dataset for the PyTorch DNN mode
	* domain_function.py - Contains several functions to extract the features of the dataset if a particular feature is not given
	* jsado.py, dumper.js - Code to find the Obfuscated JS code if the feature is not given [Github](https://github.com/lucianogiuseppe/JS-Auto-DeObfuscator)

- **requirement.txt**: Packages required for the project to run.

- **LICENSE**: License File

## Exploratory Data Analysis
**EDA** and **GDA** are done on the dataset to get the maximum insights about the data and engineering features accordingly like the Distribution of the Malicious Webpages around the world on a choropleth map, Kernel Density Estimation of the Javascript Code and more. </br>
 > The Exploration Notebook is given in the `notebooks` folder.

## Preprocessing
The preprocessing is done on the data to make it ready for the modelling part and features engineering is done as well. First, several features are added to the dataset like the length of the content, count of the special characters in the raw content, and Type of the network (A, B, C) according to the IP address. </br>

The Categorical features are converted into numeric values and the normalization is done on the content length and the count of the special characters using the Standard Scaler from scikit learn. Some features are removed and are not used in the training. </br>
> The preprocessing functions and the code is given in the `src` folder.

## Models
A total of four models are used in the project named, **XGBoost**, **Logistic Regression**, **Deep Neural Network** and **Decision Tree**. The models are trained, validated and tested using 5 5-fold Cross-validation set [Stratified k folds]. The structure of the models is given in the `src/model_dispatcher.py` file. The best performing model was the XGBoost Classifier followed by Deep Neural Network. The trained models are given in the models folder which can be used for predictions or can be trained again with new features. </br>
> The Notebook containing the modelling and the results are given in the `notebooks` folder.

## How to Run
#### Installing the required libraries
> Install all the required libraries given in the requirement.txt file

#### Downloading the dataset
> Download the dataset given in the dataset.txt file and place the data in the `data/` folder

#### Using Command Prompt
> Run these commands in the main directory of the project to train the models and do prediction with deployment.

###### Training
New models can be trained using the `train.py` with different folds. </br>

`python3 scripts/train.py --folds [fold] --model [model name]`
- folds - [0, 4] </br>
- model - [xg, dt, lr, dnn]
	> dnn - Deep Neural Network </br>
	> xg - XGBoost </br>
	> dt - Decision Tree </br>
	> lr - Logistic Regression </br>

###### Predictions
Predictions can be made using the trained models. This can be done using the `predict.py`.

`python3 scripts/predict.py --path [path] --model [model]`
- path - path of the testing data
- model - The trained model in the `output/models` folder [xg, dt, dnn, lr]

#### Deployment
Run the `deployment/deployment.py`. The default browser will open the locally deployed app 'http://localhost:5000/maliciousWPC'

<img width="914" alt="main_1" src="https://github.com/user-attachments/assets/b43d8aeb-d149-402a-b361-96b656801131" />

> URL &emsp; : &emsp;URL of the web page. </br>
> Geographical Location &emsp; : &emsp; geo Loc of the web page [Choose 'other' if don't know -- The code will extract it] </br>
> IP Address &emsp; : &emsp; IP Address of the web page [Leave as it is if don't know -- The code will extract it] </br>
> Top Level Domain &emsp; : &emsp; TLD of the web page [Choose 'other' if don't know -- The code will extract it] </br>
> Prediction models &emsp; : &emsp; The model to be used for prediction. </br>

<img width="894" alt="main_2" src="https://github.com/user-attachments/assets/1092c70a-a3b5-40ac-987a-3994f6a72437" />

> The output page contains the 'WHO IS' Information of the webpage and the prediction *Malicious* or *Benign*.
