import pandas as pd
import numpy as np

def test_drop_duplicates():
    df = pd.DataFrame({'ISBN': ['111', '111', '222'], 'Title': ['A', 'A', 'B']})
    df = df.drop_duplicates(subset='ISBN')
    assert df.shape[0] == 2

def test_fill_missing_values():
    df = pd.DataFrame({'Publisher': [None, 'ABC', None]})
    df['Publisher'].fillna("Unknown", inplace=True)
    assert df['Publisher'].isnull().sum() == 0
