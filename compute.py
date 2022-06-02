#!/usr/bin/env python3

# importing python modules
import os
import sys
import yaml

# importing data analysis and ml packages
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score

# The functions are listed now.
# There are two type of functions - 1) brane/external functions that will be run in the containers 2) helper functions which are
# just python functions that will be called in the bran/external functions to do some tasks
# If a function has type annotation, then it is a brane/external function or else it's just a helper function


def data_shape(path: str) -> str:
    """
        Function that returns the shape of the dataframe after reading the data from a file.
        Function type: Brane/external function that will be run in a container runtime
        Input: path of the file (the file should be placed under /data)
        Output: Shape of the dataframe
    """
    try:
        df = pd.read_csv(path)
        shape = "Shape is:" + str(df.shape)
        return shape
    except IOError as e:
        return str(e.errno)


def get_df(path: str):
    """
        Function that reads data from csv files and returns a dataframe
        Function type: Helper function, that means will be called inside the brane functions
        Input: path of the dataset
        Output: dataframe
    """
    df = pd.read_csv(path)
    return df


def name_proc(df):
    """
        Function that preprocesses the name feature to extract the title and create a different feature out of it. 
        Function type: Helper function, that means will be called inside the brane functions
        Input: dataframe
        Output: dataframe
    """
    df['Title'] = df['Name'].apply(lambda x: x.split(','))
    df['Title'] = df['Title'].apply(lambda x: x[-1].split('.')[0].strip())
    # Now on the basis of the different titles, we divide passenger in two different categories
    df['Title'] = df['Title'].replace(
        ['the Countess', 'Dr', 'Jonkheer', 'Master', 'Mlle', 'Mile', 'Mme', 'Ms', 'Rev'], 'Other')
    df['Title'] = df['Title'].replace(
        ['Don', 'Sir', 'Capt', 'Col', 'Lady', 'Major', 'Dona'], 'Old')
    return df


def imputting_na_values(df):
    """
        Function that imputs some logical value for the null/na value in a feature
        Function type: Helper function, that means will be called inside the brane functions
        Input: dataframe
        Output: dataframe
    """
    df['Embarked'].fillna('S', inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Sex'].fillna('other', inplace=True)
    df['Pclass'].fillna(value=3, inplace=True)
    df['SibSp'].fillna(value=0, inplace=True)
    df['Parch'].fillna(value=0, inplace=True)
    return df


def cat_to_num(df):
    """
        Function that changes categorical data to numerical data
        Function type: Helper function, that means will be called inside the brane functions
        Input: dataframe
        Output: dataframe
    """
    df['Sex'].replace('female', 0, inplace=True)
    df['Sex'].replace('male', 1, inplace=True)
    df['Sex'].replace('other', 2, inplace=True)
    df['Embarked'].replace('S', 0, inplace=True)
    df['Embarked'].replace('C', 1, inplace=True)
    df['Embarked'].replace('Q', 2, inplace=True)
    df['Title'] = df['Title'].map(
        {'Miss': 0, 'Mr': 1, 'Mrs': 2, 'Old': 3, 'Other': 4})
    return df


def missingAge(df):
    """
        Function that handles the missing age for instances
        Function type: Helper function, that means will be called inside the brane functions
        Input: dataframe
        Output: dataframe
    """
    guess_ages = np.zeros((2, 3))
    guess_ages
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = df[(df['Sex'] == i) & (
                df['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            # Convert random age float to nearest .5 age
            guess_ages[i, j] = int(age_guess/0.5 + 0.5) * 0.5
    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[(df.Age.isnull()) & (df.Sex == i) & (
                df.Pclass == j+1), 'Age'] = guess_ages[i, j]
    df['Age'] = df['Age'].astype(int)
    df['AgeBand'] = pd.cut(df['Age'], 5)
    df.loc[df['Age'] <= 16, 'Age'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
    df.loc[df['Age'] > 64, 'Age']
    return df


def family(df):
    """
        Function to create a new feature that tells if a person is alone or not
        Function type: Helper function, that means will be called inside the brane functions
        Input: dataframe
        Output: None
    """
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    for data in [df]:
        data['IsAlone'] = 0
        data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1
# Preprocessing


def preprocessing(path: str, isTrain: int) -> int:
    """
        Function that preprocesses the dataset (train, test)
        Function type: Brane/external function that will be run in a container runtime
        Input: path of the file (the file should be placed under /data), isTrain bool (whether training dataset or not)
        Output: 0 (successfully wrote the processed the data), non-zero (unsuccessful)
    """
    df = get_df(path)
    df = name_proc(df)
    df = imputting_na_values(df)
    df = cat_to_num(df)
    df = missingAge(df)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    for data in [df]:
        data['IsAlone'] = 0
        data.loc[data['FamilySize'] > 1, 'IsAlone'] = 1
        df = df.drop(['Cabin', 'Ticket', 'Name', 'AgeBand',
                     'SibSp', 'Parch', 'FamilySize', 'PassengerId'], axis='columns')
    try:
        df.to_csv("/data/prep_data"+str(isTrain) + ".csv")
        return 0
    except IOError as e:
        return e.errno


'''TRAINING THE MODEL'''
'''TESTING AND PREDICTIONS'''


def modelling(path_train: str, path_test: str, mode: str) -> int:
    """
        Function to train the model on the basis mode provided. Mode is the identifier for the machine learning model provided. 
        Function type: Brane/external function that will be run in a container runtime
        Input: path of the train file (the file should be placed under /data), path of the test file, mode (model name identifier)
        Output: 0 (successfully wrote the submission file), non-zero (unsuccessful)
    """
    df_train = get_df(path_train)
    y_train = df_train['Survived']
    # dropping the survived column because it is the value we want to pridict in the test set
    x_train = df_train.drop('Survived', axis='columns')

    model = get_model(mode)  # getting the model on the basis of mode
    model.fit(x_train, y_train)  # fitting the model

    x_test = get_df(path_test)
    y_pred = model.predict(x_test)  # prediction

    sample_submission = x_test.copy(deep=True)
    sample_submission['Survived'] = y_pred
    sample_submission.drop(sample_submission.columns.difference(
        ['PassengerId', 'Survived']), 1, inplace=True)

    try:
        sample_submission.to_csv(
            "/data/prediction_" + str(mode) + ".csv", index=False)  # writing to output file
        return 0
    except IOError as e:
        return e.errno


def get_model(name):
    """
        Function that returns the model
        Function type: Helper function, that means will be called inside the brane functions
        Input: Name of the model
        Output: Model class
    """
    if(name == 'dtc'):
        model = DecisionTreeClassifier()
    elif(name == 'rfc'):
        model = RandomForestClassifier(n_estimators=200, bootstrap=True, criterion='entropy',
                                       min_samples_leaf=5, min_samples_split=2, random_state=1)
    elif(name == 'bnb'):
        model = BernoulliNB()
    return model


def get_model_accuracy(path_train: str, mode: str) -> str:
    """
        Function to check the model accuracy
        Function type: Brane/external function that will be run in a container runtime
        Input: path of the train file (the file should be placed under /data) mode (model name identifier)
        Output: Validation score
    """
    model = get_model(mode)
    df_train = get_df(path_train)
    y_train = df_train['Survived']
    X_train = df_train.drop('Survived', axis='columns')
    all_accuracies = cross_val_score(
        estimator=model, X=X_train, y=y_train, cv=5)
    result = str(all_accuracies.mean())
    return result


# The entrypoint of the script
if __name__ == "__main__":
    # Make sure that at least one argument is given, that is either - 'shape' or 'preprocess' or 'model' or 'accuracy'
    if len(sys.argv) != 2 or (sys.argv[1] != "shape" and sys.argv[1] != "preprocess" and sys.argv[1] != "model" and sys.argv[1] != "accuracy"):
        print(f"Usage: {sys.argv[0]} write|read")
        exit(1)

    # If it checks out, call the appropriate function
    command = sys.argv[1]
    if command == "shape":
        # Print the result with the YAML package
        print(yaml.dump({"shape": data_shape(os.environ["PATH"])}))
    elif command == "preprocess":
        # Print the result with the YAML package
        print(yaml.dump({"code": preprocessing(
            os.environ["NAME"], os.environ["ISTRAIN"])}))

    elif command == "model":
        # Print the result with the YAML package
        print(yaml.dump({"code": modelling(
            os.environ["NTRAIN"], os.environ["NTEST"], os.environ["MODE"])}))

    elif command == "accuracy":
        # Print the result with the YAML package
        print(yaml.dump({"code": get_model_accuracy(
            os.environ["NTRAIN"], os.environ["MODE"])}))
