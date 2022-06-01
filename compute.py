#!/usr/bin/env python3
import os
import sys
import yaml

import pandas as pd
import numpy as np
import sklearn

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# define function to read file and return data size

def read_file(name: str) -> str:
    try:
        df = pd.read_csv(name)
        shape = "Shape is:" + str(df.shape)
        return shape
    except IOError as e:
        return str(e.errno)

def get_df(name: str):
    df = pd.read_csv(name)
    return df

def name_proc(df):
    df['Title'] = df['Name'].apply(lambda x: x.split(','))
    df['Title'] = df['Title'].apply(lambda x: x[-1].split('.')[0].strip())
    df['Title'] = df['Title'].replace(['the Countess','Dr','Jonkheer','Master','Mlle','Mile','Mme','Ms','Rev'],'Other')
    df['Title'] = df['Title'].replace(['Don','Sir','Capt','Col','Lady','Major','Dona'],'Old')
    return df
# Remove missing values
def remove_missing(df):
    df['Embarked'].fillna('S', inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Sex'].fillna('other', inplace=True)
    df['Pclass'].fillna(value=3, inplace=True)
    df['SibSp'].fillna(value=0, inplace=True)
    df['Parch'].fillna(value=0, inplace=True)
    return df
# Change categorical data to numerical data
def cat_to_num(df):
    df['Sex'].replace('female',0 ,inplace=True)
    df['Sex'].replace('male',1 ,inplace=True)
    df['Sex'].replace('other', 2, inplace=True)
    df['Embarked'].replace('S', 0, inplace=True)
    df['Embarked'].replace('C', 1, inplace=True)
    df['Embarked'].replace('Q', 2, inplace=True)
    df['Title']=df['Title'].map({'Miss':0,'Mr': 1,'Mrs': 2,'Old':3,'Other':4})
    return df
# Handling Age feature
def missingAge(df):
    guess_ages = np.zeros((2,3))
    guess_ages
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = df[(df['Sex'] == i) & (df['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[ (df.Age.isnull()) & (df.Sex == i) & (df.Pclass == j+1),'Age'] = guess_ages[i,j]
    df['Age'] = df['Age'].astype(int)
    df['AgeBand'] = pd.cut(df['Age'], 5)
    df.loc[df['Age'] <= 16, 'Age'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
    df.loc[df['Age'] > 64, 'Age']
    return df
# Combine Parch and SibSp - create feature 'IsAlone'
def family(df):
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    for data in [df]:
        data['IsAlone'] = 0
        data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1
# Preprocessing
def preprocessing(name: str, isTrain: int) -> int:
    df = get_df(name)
    df = name_proc(df)
    df = remove_missing(df)
    df = cat_to_num(df)
    df = missingAge(df)
#     df = family(df)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    for data in [df]:
        data['IsAlone'] = 0
        data.loc[data['FamilySize'] > 1, 'IsAlone'] = 1
    if(isTrain):
        df = df.drop(['Cabin', 'Ticket', 'Name', 'AgeBand', 'SibSp', 'Parch', 'FamilySize'], axis='columns')
    else:
        df = df.drop(['Name', 'Ticket', 'Cabin', 'AgeBand', 'SibSp', 'Parch', 'FamilySize'], axis='columns')
    
    try:
        df.to_csv("/data/prep_data"+str(isTrain)+ ".csv")
        return 0
    except IOError as e:
        return e.errno
    # return temp, df

'''TRAINING THE MODEL'''
'''TESTING AND PREDICTIONS'''
def modelling(name_train: str, name_test: str, mode: str) -> int:
    df_train = get_df(name_train)
    y_train = df_train['Survived']
    x_train = df_train.drop('Survived', axis='columns')
    
    model = get_model(mode)
    model.fit(x_train, y_train)

    x_test = get_df(name_test)
    y_pred = model.predict(x_test)

    sample_submission = x_test.copy(deep=True)
    sample_submission['Survived'] = y_pred
    # sample_submission.head()
    sample_submission.drop(sample_submission.columns.difference(['PassengerId','Survived']), 1, inplace=True)
    
    try:
        sample_submission.to_csv("/data/prediction_" + str(mode) + ".csv", index= False)
        return 0
    except IOError as e:
        return e.errno

def get_model(name):
    if(name=='dtc'):
        model = DecisionTreeClassifier()
    elif(name=='rfc'):
        model = RandomForestClassifier(n_estimators=200, bootstrap=True, criterion= 'entropy', min_samples_leaf=5, min_samples_split=2, random_state=1)
    elif(name=='bnb'):
        model = BernoulliNB()
    return model

def get_model_accuracy(name_train: str, mode: str) -> str:
    model = get_model(mode)
    df_train = get_df(name_train)
    y_train = df_train['Survived']
    X_train = df_train.drop('Survived', axis='columns')
    all_accuracies = cross_val_score(estimator=model, X=X_train, y=y_train, cv=5)
    result = str(all_accuracies.mean())
    return result



if __name__ == "__main__":   
    if len(sys.argv) != 2 or (sys.argv[1] != "read" and sys.argv[1] != "preprocess" and sys.argv[1] != "model" and sys.argv[1] != "accuracy"):
        print(f"Usage: {sys.argv[0]} write|read")
        exit(1)

    # If it checks out, call the appropriate function
    command = sys.argv[1]
    if command == "read":
        # Write the file and print the error code
        print(yaml.dump({ "shape": read_file(os.environ["NAME"]) }))
    elif command == "preprocess":
        # Read the file and print the contents
        print(yaml.dump({ "code": preprocessing(os.environ["NAME"], os.environ["ISTRAIN"])}))

    elif command == "model":
        # Read the file and print the contents
        print(yaml.dump({ "code": modelling(os.environ["NTRAIN"], os.environ["NTEST"], os.environ["MODE"])}))    
    
    elif command == "accuracy":
        print(yaml.dump({ "code": get_model_accuracy(os.environ["NTRAIN"], os.environ["MODE"])}))
