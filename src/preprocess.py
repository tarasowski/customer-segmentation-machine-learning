import numpy as np
import pandas as pd
import json
from functools import reduce
import joblib

pipe = lambda fns: lambda x: reduce(lambda v, f: f(v), fns, x)


def missing_to_list(dfs):
    main_df, feat_df = dfs
    feat_df_xs = feat_df['missing_or_unknown'].apply(
            lambda x: x[1:-1].split(','))
    feat_df_enc = feat_df_xs.apply(
            lambda x: list(map(
                    lambda y: y if y == 'XX' or y == 'X' or y == '' else int(y), x)))
    feat_df['missing_or_unknown'] = feat_df_enc
    return (main_df, feat_df)

def attribute_as_index(dfs):
    main_df, feat_df = dfs
    return (main_df, feat_df.set_index('attribute'))

def nans_transformer(feat_df):
    def inner(row):
        missing_values = feat_df.loc[row.name, 'missing_or_unknown']
        checked = row.isin(missing_values)
        return row.where(~checked)
    return inner

def convert_missing_nans(dfs):
    main_df, feat_df = dfs
    return (main_df.apply(nans_transformer(feat_df)), feat_df)

def null_cols_drop(dfs):
    main_df, feat_df = dfs
    null_columns_count = main_df.isnull().sum() / len(main_df)
    above_40 = null_columns_count[null_columns_count > 0.4].index
    return (main_df.drop(above_40, axis=1), feat_df)

def null_rows_drop(dfs):
    main_df, feat_df = dfs
    null_rows_count = main_df.isnull().sum(axis=1)
    return (main_df[null_rows_count == 0], feat_df)

def re_encode_cats(dfs):
    main_df, feat_df = dfs
    categorical = feat_df.loc[:, 'type'] == 'categorical'
    categorical_cols = feat_df[categorical].index.tolist()
    dropped_cols = ['AGER_TYP', 'KK_KUNDENTYP', 'TITEL_KZ']
    clean_categorical = [x for x in categorical_cols if x not in dropped_cols]
    number_of_categorical = main_df[clean_categorical].apply(lambda x: x.nunique())
    multi_level_xs = number_of_categorical[number_of_categorical > 2].index.tolist()
    encode_binary = pd.get_dummies(main_df, columns=['OST_WEST_KZ'])
    encode_multi = pd.get_dummies(encode_binary, columns=multi_level_xs)
    return (encode_multi, feat_df)

def mixed_type_decade(dfs):
    main_df, feat_df = dfs
    main_df['DECADE'] = main_df['PRAEGENDE_JUGENDJAHRE'].map({
            1: 0,
            2: 0,
            3: 1,
            4: 1,
            5: 2,
            6: 2,
            7: 2,
            8: 3,
            9: 3,
            10: 4,
            11: 4,
            12: 4,
            13: 4,
            14: 5,
            15: 5
            })
    return (main_df, feat_df)

def mixed_type_movement(dfs):
    main_df, feat_df = dfs
    main_df['MOVEMENT'] = main_df['PRAEGENDE_JUGENDJAHRE'].map({
            1: 0,
            2: 1,
            3: 0,
            4: 1,
            5: 0,
            6: 1,
            7: 1,
            8: 0,
            9: 1,
            10: 0,
            11: 1,
            12: 0,
            13: 1,
            14: 0,
            15: 1
            })
    return (main_df, feat_df)

def cameo(pos):
    def inner(val):
        return (int(str(val)[0]), int(str(val)[1]))[pos]
    return inner
   
def mixed_type_cameo(dfs):
    main_df, feat_df = dfs
    main_df['WEALTH'] = main_df['CAMEO_INTL_2015'].apply(cameo(0))
    main_df['LIFE_STAGE'] = main_df['CAMEO_INTL_2015'].apply(cameo(1))
    return (main_df, feat_df)

def mixed_types_drop(dfs):
    main_df, _ = dfs
    return main_df.drop(['LP_LEBENSPHASE_FEIN', 
            'LP_LEBENSPHASE_GROB', 
            'PLZ8_BAUMAX', 
            'PRAEGENDE_JUGENDJAHRE',
            'CAMEO_INTL_2015'
            ], axis=1)

main = pipe([
        missing_to_list,
        attribute_as_index,
        convert_missing_nans,
        null_cols_drop,
        null_rows_drop,
        re_encode_cats,
        mixed_type_decade,
        mixed_type_movement,
        mixed_type_cameo,
        mixed_types_drop,
        ])


def load_dfs(dir_main, dir_feat, nrows):
    return (pd.read_csv(dir_main, delimiter=';', nrows=nrows),
            pd.read_csv(dir_feat, delimiter=';'))

if __name__ == '__main__':
    try:
        dfs = load_dfs('../inputs/Udacity_AZDIAS_Subset.csv', 
            '../inputs/AZDIAS_Feature_Summary.csv', 
            1000)
        main(dfs)
    except Exception as e:
        print(e)

