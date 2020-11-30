

''' THIS HAS YET TO BE TESTED BUTIT SHOULD WORK....'''
import pandas as pd
import os 
#import seaborn as #sns
#import matplotlib.pyplot as plt
#%matplotlib inline
import pandas_ta as pta


#data always goes in data
path ='data/'

#d as in directory
d = os.listdir(path)
pd.DataFrame(d,columns=['sheets'])

d

# STANARD DATA LOADING 
#### `Open` `high` `low` `close` `volume` `date`

sheet = 0
d[sheet]

# loading data
### this part will be replaced by a data loader

df = pd.read_csv(path+d[sheet],index_col='time')
df.index = pd.to_datetime(df.index)#,unit='s')

df = df[['open','high','low','close','Volume']]
df['date'] = df.index.date
df

''' ADD FEATURES HERE '''
# From Here On Data Should Look Like that
### doesnt matter if its from downloader or what
#### [datetimeindex open 	high 	low 	close 	Volume 	date] 

# Technical aylisis features

import ta

df = ta.add_all_ta_features(df,'open','high','low','close','Volume',fillna=True)

# Candels features

df.index

#import candle_sticks
#import daily_close_compare

# adds all the candel patterns
#candle_sticks.all_candels(df)

#adds the daily_close_comparison class i made [which the day checker is not working btw]
#df = daily_close_compare.create_anna(df)

#plot empty values... candel sticks dont make any nul values
#sns.heatmap(df.isnull())

#pd.set_option('display.max_columns',None)
#df




''' FEATURES END HERE '''
print(len(df.columns))
df.head()

print('length of dataset is:',len(df))
print('length of features is:',len(df.columns))

# you are supposed to have 5 times as many rows as columns at a minmum for ml
five_times_col = len(df.columns)*5
print('maximum features:',five_times_col)

#standardizing data
df = df.rename(columns={'close':'Close',
               'open':'Open',
               'low':'Low',
               'high':'High',
               })
df

df = df.drop('date',axis=1)

from sklearn.preprocessing import StandardScaler

def scale(df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    sdf    = pd.DataFrame(scaled,columns = df.columns)
    sdf.index = df.index
    return sdf

# if there is nulls you want them to be at the beggining 
#sns.heatmap(df[:50].isnull())

#if there are nulls here SOMTHING IS WRONG and yourleaking data
#sns.heatmap(df[50:].isnull())

# Targets

import add_targets

#making the features data a copy so its unaffected by adding targets
fdf = df.copy()

#standard deviation based targets (default is .5,1,2)
tdf ,rhs,ts= add_targets.std_targs(df)
tdf ,rhs,ts= add_targets.std_targs(df,t1=3,t2=4,t3=5)
#tdf is for target_df
tdf

#adding % based targets
tdf = add_targets.add_targets(tdf)
tdf.columns

tdf = add_targets.ma_targets(tdf)

pd.set_option('display.max_columns',None)

tdf

### It should contain nulls

#sns.heatmap(tdf.isnull())

### Check the first 50

#sns.heatmap(tdf[:50].isnull())

### Check the last 50

tdf = tdf.dropna(axis=0)
print(len(tdf))
#sns.heatmap(tdf.isnull())
print(len(tdf))

#but now there are differnces in the legnths
print(len(tdf),len(fdf))



#so use the index as a cookie cutter template to have the exact same values
fdf = fdf.T[tdf.T.columns].T
print('target_len',len(tdf))

print( 'feature_len',len(fdf))

#sns.heatmap(tdf.isnull())


#sns.heatmap(fdf.isnull())

# scale

sdf = scale(fdf)


#sns.heatmap(sdf.isnull())

# Save Clean Parsed Data

#create the clean data_path if it doesnt already exist
cpath = 'clean_data/'
if not os.path.exists(cpath):
    os.mkdir(cpath)

#name em
tname = 'targets_'+d[sheet]
fname = 'features_'+d[sheet]
sname = 'scaled_'+d[sheet]


#save em
tdf.to_csv(cpath+tname)
fdf.to_csv(cpath+fname)
sdf.to_csv(cpath+sname)
