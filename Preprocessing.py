import pandas as pd
pd.set_option('display.expand_frame_repr', False)


def promo(x):
    if isinstance(x, str):
        if x.find("Y") != -1:
            return 1
        else:
            return 0
    else:
        return 0


def preprocessing(pred, df):
    if not pred:
        tmp = df[['WEEK', 'ACTUAL']].groupby('WEEK')[['ACTUAL']].sum()
        tmp.columns = ['Revenue']
        tmp['WEEK'] = tmp.index
        df['WEEK'] = df['WEEK'].apply(lambda x: int(x))
        df = pd.merge(df, tmp, on='WEEK')
        tmp = df[['WEEK', 'PROMO']].groupby('WEEK')[['PROMO']].sum()
        tmp['Promo'] = tmp['PROMO'].apply(lambda x: promo(x))
        tmp = tmp.drop('PROMO', 1)
        tmp['WEEK'] = tmp.index
        df = pd.merge(df, tmp, on='WEEK')
        df['Year'] = df['WEEK'].apply(lambda x: int(str(x)[1:3]))
        df['Date'] = df['WEEK'].apply(lambda x: int(str(x)[3:5] + str(x)[5:7]))
        df = df.drop('WEEK', 1)
        df = df.drop('SEG', 1)
        df = df.drop('ACTUAL', 1)
        df = df.drop('PROMO', 1)
        df = df.drop('BU', 1)
        df = df.drop_duplicates()
        df = df.sort_values(['Year', "Date"])
        if df.shape[0] < 15:
            f = open("seasonal.txt", 'a')
            f.write(df['PLN_LABEL'].iloc[0] + "|")
    else:
        df = df.drop('SEG', 1)
        tmp = df[['WEEK', 'PROMO']].groupby('WEEK')[['PROMO']].sum()
        tmp['Promo'] = tmp['PROMO'].apply(lambda x: promo(x))
        tmp = tmp.drop('PROMO', 1)
        df = pd.merge(df, tmp, on='WEEK')
        df['Year'] = df['WEEK'].apply(lambda x: int(str(x)[1:3]))
        df['Date'] = df['WEEK'].apply(lambda x: int(str(x)[3:5] + str(x)[5:7]))
        df = df.drop('PROMO', 1)
        df = df.drop_duplicates()
    return df
