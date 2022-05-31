#!/usr/bin/env python3
import os
import sys
import yaml

import pandas as pd
import numpy as np


# define function to read file and return data size

def read_file(name: str) -> int:
    try:
        df = pd.read_csv(f"/data/{name}.csv")
        return 0
    except IOError as e:
        return e.errno

def get_df(name):
    df = pd.read_csv(f"/data/{name}.csv")
    return df

def find_missing_values(name: str) -> int:
    df = get_df(name)
    count = df.isna().sum()
    try:
        count.to_csv("/data/dataNull.csv")
        return 0
    except IOError as e:
        return e.errno


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
    # df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
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
        # temp = df['Survived']
        df = df.drop(['Cabin', 'Ticket', 'Name', 'AgeBand', 'SibSp', 'Parch', 'FamilySize'], axis='columns')
    else:
        # temp = df['PassengerId']
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
    x_train = df.drop('Survived', axis='columns')
    
    if(mode=='dtc'):
        dtc_model = DecisionTreeClassifier()
    #Random Forest
    elif(mode=='rfc'):
        model = RandomForestClassifier()
    #XGBoost
    elif(mode=='xgb'):
        model = XGBClassifier()
    #BernoulliNB
    elif(mode=='bnb'):
        model = BernoulliNB()

    model.fit(x_train, y_train)

    x_test = get_df(name_test)
    y_pred = model.predict(x_test)

    try:
        y_pred.to_csv("/data/prediction_" + str(mode) + ".csv")
        return 0
    except IOError as e:
        return e.errno



def test_pred(name: str):
    #Preprocess testing dataset

    Pid, test_data = preprocessing(test_data, 0)
    test_data['PassengerId'] = Pid
    x_test = test_data[['PassengerId', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'IsAlone']]
    #Predictions
    y_pred = model.predict(x_test)

if __name__ == "__main__":
   
    if len(sys.argv) != 2 or (sys.argv[1] != "nullwrite" and sys.argv[1] != "read" and sys.argv[1] != "preprocess" and sys.argv[1] != "modelling"):
        print(f"Usage: {sys.argv[0]} write|read")
        exit(1)

    # If it checks out, call the appropriate function
    command = sys.argv[1]
    if command == "read":
        # Write the file and print the error code
        print(yaml.dump({ "code": read_file(os.environ["NAME"]) }))
    elif command == "nullwrite":
        # Read the file and print the contents
        print(yaml.dump({ "code": find_missing_values(os.environ["NAME"])}))

    elif command == "preprocess":
        # Read the file and print the contents
        print(yaml.dump({ "code": preprocessing(os.environ["NAME"], os.environ["ISTRAIN"])}))

    elif command == "model":
        # Read the file and print the contents
        print(yaml.dump({ "code": modelling(os.environ["NAME_TRAIN"], os.environ["NAME_TEST"], os.environ["MODE"])}))    
    # Done!