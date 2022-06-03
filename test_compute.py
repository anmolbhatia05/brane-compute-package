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

def get_model_accuracy():
    # checks if the function get_model_accuracy returns accuracy as float number for given model
    test_input_path = './sample.csv'
    assert type(compute.get_model_accuracy(test_input_path, 'dtc')) == float