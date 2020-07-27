from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import os
import json
#Instructions:
#Upload new dataset into csv format with a header tha contains the name class
#Load the dataset using this dataset object definition without any preprocessing parameters.
#ex: test_dataset = dataset(path='path to the uploaded dataset file',name='name of the dataset')
#Then a dataset folder containing a train and test set based on the original dataset will be created.
#if any preprocessing needs to be done a json file, named preprocessing.json must be placed in the dataset folder and preprocessing #parameters must be set to true in the dataset object.
#preprocess method must be called afterwards.
#Example preprocessing.json is on the same folder with dataset.py


#Define a dataset class   
class dataset(object):
    #Initialize dataset variables
    def __init__(self,path,name=None):
        self.path = path
        self.name = name
        self.train = None
        self.test = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        #Load the dataset when the dataset object is initialized
        self.load_dataset()
    #String represenation of a dataset object
    def __str__(self):
        return '< %s dataset >'%self.name
    
    #Load the dataset
    def load_dataset(self):
        #if there is a dataset file with the given name ...
        if os.path.isfile(self.path):
            #Read the file and load it to a dataframe
            data = pd.read_csv(self.path)
            #Split the dataset file in train and test set files(30% test)
            train,test = train_test_split(data,test_size=0.30,shuffle=True,stratify=data['class'],random_state=42)
            #New Dataset folder path
            dir_name = './datasets/'+self.name.lower().replace(' ','_')
            #if the dataset folder does not exist create it and change the dataset path to the newlly created dataset folder.
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
                self.path = dir_name
            #Save the train and test set into 2 sepearate files in the dataset folder
            train.to_csv(dir_name+'/train.csv',index=False)
            test.to_csv(dir_name+'/test.csv',index=False)
            #Remove the original dataset file
            os.remove(self.path+'.csv')
        #if there is a train file in the dataset folder
        if os.path.isfile(self.path+'/train.csv'):
            #Load the train file
            self.train = pd.read_csv(self.path+'/train.csv')
        #Else raise error
        else:
            raise Exception('No train set file found in the specified directory.')
        #if there is a test file in the dataset folder
        if os.path.isfile(self.path+'/test.csv'):
            #load the test file
            self.test = pd.read_csv(self.path+'/test.csv')
        #Else raise error
        else:
            raise Exception('No test set file found in the specified directory.')
    #Preprocess the dataset based on the json file in the dataset folder
    def preprocess(self,ord_cat=False,cat=False,cl=False,ord_cl=False,to_numeric=None,drop_cols=None,replace_nan=False,cat_lb=False):
        #Path to the preprocessing file
        json_path = self.path+'/preprocessing.json'
        #Make a copy of the train anf test set
        train_cp = self.train.copy(deep=True)
        test_cp = self.test.copy(deep=True)
        #Check if there is a preprocessing file in the dataset folder
        if os.path.isfile(json_path):
            #Open the json file 
            with open(json_path,'r') as pre_json:
                #Load the contents of the json file
                pre =  json.load(pre_json)
                #Check if the dataset contains ordinal categorical features
                if ord_cat == True and pre['ord_cat']:
                    #For each feature in the ord_cat key in the preprocessing json file
                    for feature in pre['ord_cat']:
                            #Perform ordinal label(integer) encoding 
                            train_cp[feature] = train_cp[feature].astype('category',ordered=True,categories=pre['ord_cat'][feature]).cat.codes
                            test_cp[feature] = test_cp[feature].astype('category',ordered=True,categories=pre['ord_cat'][feature]).cat.codes
                #Check if the dataset contains categorical features and perform simple label encoding
                if cat_lb == True and pre['cat_lb']:
                    le=LabelEncoder()
                    for feature in pre['cat_lb']:
                        data=train_cp[feature].append(test_cp[feature])
                        le.fit(data.values)
                        le.fit(data.values)
                        train_cp[feature] = le.transform(train_cp[feature])
                        test_cp[feature] = le.transform(test_cp[feature]) 
                #Check if the dataset contains categorical features and perform ohc
                if cat == True and pre['cat']:
                    #Perform one hot encoding on the features in the cat key in the preprocessing file
                    train_cp = pd.get_dummies(train_cp,columns=pre['cat'])
                    test_cp = pd.get_dummies(test_cp,columns=pre['cat'])   
                #Check if the class labels are ordinal
                if ord_cl == True and pre['ord_classes']:
                    #Perform ordinal label(integer) encoding on the class column
                    train_cp['class'] = train_cp['class'].astype('category',ordered=True,categories=pre['ord_classes']).cat.codes
                    test_cp['class'] = test_cp['class'].astype('category',ordered=True,categories=pre['ord_classes']).cat.codes  
                #Check if the class labels need encoding
                if cl == True:
                    #Perform label(integer) encoding on the class column
                    categories = train_cp['class'].append(test_cp['class']).unique().tolist()
                    train_cp['class'] = train_cp['class'].astype('category',categories=categories).cat.codes
                    test_cp['class'] = test_cp['class'].astype('category',categories=categories).cat.codes  
                if to_numeric == True and pre['to_numeric']:
                    for col in pre['to_numeric']:
                        if col != 'class': 
                            train_cp[col] = pd.to_numeric(train_cp[col].copy(),errors='coerce')
                    for col in pre['to_numeric']:
                        if col != 'class':
                            test_cp[col] = pd.to_numeric(test_cp[col].copy(),errors='coerce')
                if replace_nan == True:
                    for col in train_cp.columns.tolist():
                        if 'float' in str(train_cp[col].dtype):
                                train_cp[col] = train_cp[col].replace('NAN',train_cp[col].copy().mean())
                        else:
                            train_cp[col] = train_cp[col].replace('NAN',train_cp[col].copy().value_counts().idxmax())
                    for col in test_cp.columns.tolist():
                        if 'float' in str(test_cp[col].dtype):
                                test_cp[col] = train_cp[col].replace('NAN',train_cp[col].copy().mean())
                        else:
                            test_cp[col] = test_cp[col].replace('NAN',test_cp[col].copy().value_counts().idxmax())
                if drop_cols == True and pre['drop_cols']:
                    try:
                        train_cp = train_cp.drop(pre['drop_cols'],axis=1)
                        test_cp = test_cp.drop(pre['drop_cols'],axis=1)
                    except:
                        raise Exception('One or more columns specified is not contained in the dataset.')
        #else raise an error
        else:
            raise Exception('No preprocessing file found.')
        #Copy the preprocessed datasets into the dataset object 
        self.train = train_cp.copy()
        self.test = test_cp.copy()
    #Split the dataset(train,test) in features(X) and class(y)    
    def feat_class_split(self):
        #List with the features of the dataset
        columns = self.train.columns.tolist()
        #Remove the class feature
        columns.remove('class')
        #Split the dataset to X and y
        self.X_train = self.train[columns]
        self.y_train = self.train['class']
        self.X_test = self.test[columns]
        self.y_test = self.test['class']
'''       
def load_datasets(path):
    datasets=[]
    past_loaded = next(os.walk('./datasets'))[2]
    for dataset in 
'''