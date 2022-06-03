import pytest
import pandas as pd

import compute


def test_get_df():
    test_input_path = './sample.csv'
    assert type(compute.get_df(test_input_path)) == pd.DataFrame
