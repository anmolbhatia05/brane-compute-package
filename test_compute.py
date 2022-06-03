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
    test_input_path = './sample.csv'
    assert compute.data_shape(test_input_path) == 'Shape is:(10, 12)'
