
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder

class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feature_to_drop=None):
        self.feature_to_drop = feature_to_drop if feature_to_drop is not None else []

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        if (set(self.feature_to_drop).issubset(df.columns)):
            df.drop(self.feature_to_drop, axis=1, inplace=True)
            return df
        else:
            print('Uma ou mais features para dropar não estão no DataFrame')
            return df

class OneHotEncodingNames(BaseEstimator, TransformerMixin):
    def __init__(self, OneHotEncoding=['Gender', 'family_history', 'FAVC', 'SMOKE', 'SCC', 'MTRANS']):
        self.OneHotEncoding = OneHotEncoding

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        if (set(self.OneHotEncoding).issubset(df.columns)):
            def one_hot_enc(df, OneHotEncoding):
                one_hot_enc = OneHotEncoder(handle_unknown='ignore')
                one_hot_enc.fit(df[OneHotEncoding])
                feature_names = one_hot_enc.get_feature_names_out(OneHotEncoding)
                df = pd.DataFrame(one_hot_enc.transform(df[OneHotEncoding]).toarray(),
                                  columns=feature_names, index=df.index)
                return df

            def concat_with_rest(df, one_hot_enc_df, OneHotEncoding):
                outras_features = [feature for feature in df.columns if feature not in OneHotEncoding]
                df_concat = pd.concat([one_hot_enc_df, df[outras_features]], axis=1)
                return df_concat

            df_OneHotEncoding = one_hot_enc(df, self.OneHotEncoding)
            df_full = concat_with_rest(df, df_OneHotEncoding, self.OneHotEncoding)
            return df_full
        else:
            return df

class OrdinalFeature(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_feature=['CAEC', 'CALC']):
        self.ordinal_feature = ordinal_feature

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        # Verifica se as colunas existem
        valid_features = [f for f in self.ordinal_feature if f in df.columns]
        if valid_features:
            ordinal_encoder = OrdinalEncoder()
            df[valid_features] = ordinal_encoder.fit_transform(df[valid_features])
            return df
        else:
            print('Features ordinais não encontradas no DataFrame')
            return df

class MinMaxWithFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, min_max_scaler_ft=['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']):
        self.min_max_scaler_ft = min_max_scaler_ft

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        if (set(self.min_max_scaler_ft).issubset(df.columns)):
            min_max_enc = MinMaxScaler()
            df[self.min_max_scaler_ft] = min_max_enc.fit_transform(df[self.min_max_scaler_ft])
            return df
        else:
            print('Uma ou mais features MinMax não estão no DataFrame')
            return df
