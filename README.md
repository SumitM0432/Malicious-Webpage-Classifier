# Malicious-Webpage-Classifier

## Objective
Malicious Webpages are the pages that install malware on your system that will disrupt the computer operation and gather your personal information and many worst cases. Classifying these web pages on the internet is a very important aspect to provide the user with a safe browsing experience. </br>

The objective of this project is to classify the web pages into two categories *Malicious[Bad]* and *Benign[Good]* webpages.  Exploratory Data Analysis and Geospatial Data Analysis are done to get more insights and knowledge about the data. Features are engineered and the data is preprocessed accordingly. A total of four ML and DL models are trained. The models are **XGBoost**, **Logistic Regression**, **Decision Tree** and **Deep Neural Network**. The DNN is implemented in PyTorch and the others are implemented using scikit learn.

## Dataset
The data set is taken from [Mendeley Data](https://data.mendeley.com/datasets/gdx3pkwp47/2). The dataset contains features like the raw webpage content, geographical location, javascript length, obfuscated JavaScript code of the webpage etc. The Dataset contains around 1.5 million web pages. The description of the whole dataset is given on the link provided.

## Files
- **input**: Contains the input files for the project
  * tableconvert_csv_pkcsig.csv - Contains the iso alpha3 code for the countries
  * dataset.txt - Link to download the dataset and paste it in the input folder

- **le_ss**: Contains all the saved Label Encoder and Standard Scaler file for preprocessing </br>
	* content_len_ss.pkl - Standard Scaler for content len
	* geo_loc_encoder.pkl - Label Encoder for geo location
	* https_encoder.pkl - Label Encoder for HTTPS features
	* net_type_encoder.pkl - Label Encoder for the network type
	* special_char_ss.pkl - Standard Scaler for Special Char length
	* tld_encoder.pkl - Label Encoder for Top Level Domain
	* who_is_encoder.pkl - Label Encoder for who_is status

- **models**: Contains all the trained model [DNN, LR, DT, XG] 

- **notebooks**: Contain all the notebooks -- [Kaggle Notebook](https://www.kaggle.com/sumitm004/malicious-webpage-classifier-using-dnn-pytorch)

- **src**: Contains all the code </br>
	* config.py - Configuration file to config the whole project
	* Metrics.py - Evaluation Metrics for the training and testing
	* cross_val.py - Crossvalidation code [StratifiedKFold]
	* dataset.py - Contains the code for a custom dataset for the PyTorch DNN model
	* model_dispatcher.py - Contains the ML and DL models
	* predict.py - Python file to make a prediction
	* preprocessing.py - Python file for preprocessing the dataset for training and testing
	* preprocfunctions.py - Contains several functions to extract the features of the dataset if a particular feature is not given
	* train.py - Main run file
	* deployment.py - Python file for the deployment of the project on localhost using PyWebIO and Flask
	* jsado.py, dumper.js - Code to find the Obfuscated JS code if the feature is not given [Github](https://github.com/lucianogiuseppe/JS-Auto-DeObfuscator)
	* processing_fns - Contains all the individual python files to extract the features

- **requirement.txt**: Packages required for the project to run.

- **LICENSE**: License File

## Exploratory Data Analysis
**EDA** and **GDA** are done on the dataset to get the maximum insights about the data and engineering features accordingly like the Distribution of the Malicious Webpages around the world on a choropleth map, Kernel Density Estimation of the Javascript Code and more. </br>
 > The Exploration Notebook is given in the `notebooks` folder.

## Preprocessing
The preprocessing is done on the data to make it ready for the modelling part and features engineering is done as well. First, several features are added in the dataset like the length of the content, count of the special characters in the raw content, Type of the network (A, B, C) according to the IP address. </br>

The Categorical features are converted into numeric values and the normalization is done on the content length and the count of the special characters using the Standard Scaler from scikit learn. Some features are removed and are not used in the training. </br>
> The preprocessing functions and the code is given in the `src` folder.

## Models
A total of four models are used in the project named, **XGBoost**, **Logistic Regression**, **Deep Neural Network** and **Decision Tree**. The models are trained, validated and tested using 5 fold Cross-validation set [Stratified k folds]. The structure of the models are given in the `src/model_dispatcher.py` file. The best performing model was the XGBoost Classifier followed by Deep Neural Network. The trained models are given in the models folder which can be used for predictions or can be trained again with new features. </br>
> The Notebook containing the modelling and the results are given in the `notebooks` folder.

## How to Run
> Add the folder path on your system in the `config.py` file.

#### Using Command Prompt
> Run these commands in the `src` Directory

###### Training
New models can be trained using the `train.py` with different folds. </br>

`python3 train.py --folds [fold] --model [model name]`
- folds - [0, 4]
- model - [xg, dt, lr, dnn]
> dnn - Deep Neural Network, 
> xg - XGBoost, 
> dt - Decision Tree, 
> lr - Logistic Regression

###### Predictions
Predictions can be made using the trained models. This can be done using the `predict.py`.

`python3 predict.py --path [path] --model [model]`
- path - path of the testing file
- model - The trained model in the `models` folder [xg, dt, dnn, lr]

#### Deployment
Run the `src/deployment.py` and go to the url 'http://localhost:5000/maliciousWPC'

<img src=https://user-images.githubusercontent.com/46062583/119026748-26769900-b9c3-11eb-97ba-092bf47ad464.png width="900" height="500"/>

> URL &emsp; : &emsp;URL of the web page. </br>
> Geographical Location &emsp; : &emsp; geo Loc of the web page [Choose 'other' if don't know -- The code will extract it] </br>
> IP Address &emsp; : &emsp; IP Address of the web page [Leave as it is if don't know -- The code will extract it] </br>
> Top Level Domain &emsp; : &emsp; TLD of the web page [Choose 'other' if don't know -- The code will extract it] </br>
> Prediction models &emsp; : &emsp; The model to be used for prediction. </br>

<img src=https://user-images.githubusercontent.com/46062583/119026744-24143f00-b9c3-11eb-838a-35df5f362121.png width="900" height="500"/>


> The output page contains the 'WHO IS' Information of the webpage and the prediction *Malicious* or *Benign*.
