# this sheet creates features and cleans data
## GOALS  of this research project 
### >anylize coeficaints on features which work best 
#### `pandas_ta` `candels` `homebrew`
### > create feature scripts that can run in shell_scripts for data downloaders


import pandas as pd
import os 
import seaborn as sns
#import matplotlib.pyplot as plt
#%matplotlib inline


path ='../../data/'

d = os.listdir(path)
pd.DataFrame(d,columns=['sheets'])

# selecgt sheet
## this should be iterable for the sheets in the dir

sheet = 0
for sheet in range(0,len(d)):
    d[sheet]

    # loading data
    ### this part will be replaced by a data loader

    df = pd.read_csv(path+d[sheet],index_col='time')
    df.index = pd.to_datetime(df.index,unit='s')

    df = df[['open','high','low','close']]
    df['date'] = df.index

    # candels

    import candle_sticks
    import daily_close_compare

    df = daily_close_compare.create_anna(df)

    candle_sticks.all_candels(df)

    df = df.dropna(axis=0)
    #sns.heatmap(df.isnull())

    cpath = 'clean_data/'
    if not os.path.exists(cpath):
        print('chayyyayyayya')
        os.mkdir(cpath)

    if 'date' in df.columns:
        df = df.drop('date',axis=1)

    name = d[sheet]
    name = name.split('.')[0].replace(' ','_').replace(',','') +'_candels.csv'
    name


    df.to_csv(cpath+name)
