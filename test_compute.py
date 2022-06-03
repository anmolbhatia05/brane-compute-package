from turtle import done

import pytest
import pandas as pd

import compute


def test_get_df():
    # checks if the internal function get_df can read and return a dataframe or not
    test_input_path = './sample.csv'
    assert type(compute.get_df(test_input_path)) == pd.DataFrame


def test_mount():
    done

def test_get_shape():
    # checks if the function data_shape can read and returns correct shape of dataframe or not
    test_input_path = './sample.csv'
    assert compute.data_shape(test_input_path) == 'Shape is:(10, 12)'

def test_get_model_accuracy():
    # checks if the function get_model_accuracy returns accuracy as float number for given model
    test_input_path = './sample.csv'
    assert type(compute.get_model_accuracy(test_input_path, 'dtc')) == float

def test_modelling():
    # checks if modelling functions works for given data dataset and model
    # ie. after training model and prediction returns 0
    test_input_path = './test.csv'
    train_data_input_path = './train.csv'
    # third parameter takes model alias here rfc stands for RandomForestClassifier
    result = modelling(train_data_input_path, test_input_path, 'rfc')
    expected_output = 0
    assert result == expected_output

