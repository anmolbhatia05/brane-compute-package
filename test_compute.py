import pytest
import pandas as pd

import compute


def test_get_df():
    test_input_path = './data/sample.csv'
    assert type(compute.get_df(test_input_path)) == pd.DataFrame
