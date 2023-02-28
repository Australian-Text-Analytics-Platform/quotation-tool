# issue: pd.concat on category will result in object.
# https://github.com/pandas-dev/pandas/issues/25412
# soln: https://stackoverflow.com/questions/45639350/retaining-categorical-dtype-upon-dataframe-concatenation

import pandas as pd
from pandas.api.types import union_categoricals


def row_concat(dfs, **concat_args):
    dfs = list(dfs)
    df = pd.concat(dfs, axis=0, **concat_args)
    if type(df) == pd.DataFrame:
        # categorical dtype is not kept after concat (if there are different categories)
        cat_cols = {col for col in dfs[0].columns if dfs[0][col].dtype == 'category'}
        for col in df.columns:
            if col in cat_cols:
                df[col] = df[col].astype('category')
    else:  # series
        if dfs[0].dtype == 'category':
            df = df.astype('category')
    return df


if __name__ == '__main__':
    category = ['a', 'b']
    text = ['sometext', 'anothertext']
    df = pd.DataFrame(zip(text, category), columns=['text', 'category'])
    df['category'] = df['category'].astype('category')

    category = ['c', 'd']
    text = ['sometext', 'anothertext']
    dff = pd.DataFrame(zip(text, category), columns=['text', 'category'])
    dff['category'] = dff['category'].astype('category')
    row_concat(df, dff, ignore_index=True)
