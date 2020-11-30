import pandas as pd
import numpy as np
import os
import scrape

## marriage function 

path = 'all_stars/marriage/'
mardir= os.listdir(path)
pd.DataFrame(mardir)

# select one of the married couples

couple = mardir[0].replace('.csv','')
couple

# Extract Model Names

long_model,short_model = couple.split('_')[1],couple.split('_')[3]
print('LONG:',long_model,'\nSHORT:',short_model)

#'''HERE'''
#DO A IF MODEL PATH NOT REAL THEN 
#>MOVER.MOVE_MODEL
# ALSO MAKE IT OVERWRITE A TXT FILE EVERYTIME WITH FEATURES

import mover
model,info,bt,was_long,was_scaled,up_mpl,dn_mpl = mover.load_model(long_model)

# [ data inputs ]

# this may change when you start moving the thing aroung
apath = 'all_stars/'

#params for urllib : get_some
ticker = 'DMTK'
url_param = 'historical-chart/1hour'


new_data_name = ticker + '_'+url_param.replace('/','_')+'.csv'

# {BEGIN DATA FUNCTIONS}



#find the unscaled training data
datapath=apath+'clean_data/'
dapali = os.listdir(datapath)
data   = [i for i in dapali if 'features' in i][0]
dfpath = datapath + data
old_df     = pd.read_csv(dfpath)#.set_index('Unnamed: 0')
old_df.tail()

## Pull Recent Data


#downloads recent data TAKES - url and ticker
price_data = scrape.get_some(url_param,ticker)
price_data

#load data from csv so its free of bullshit
dnld_path = 'downloaded_data/'
price_path = dnld_path+new_data_name
df = pd.read_csv(price_path,index_col='date')
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0',axis=1)
    
df.index = pd.to_datetime(df.index)
df['date'] = df.index.date
df = df[::-1]

df

# {these are the features}

'''
in the future i want the original feature creatation to
be called by a script with a standard name 
and then you can just call it here and ya done!
'''

#import candle_sticks
#import daily_close_compare

# Technical aylisis features

import ta

df = ta.add_all_ta_features(df,'open','high','low','close','volume',fillna=True)

# Candels features

df.index


# adds all the candel patterns
#candle_sticks.all_candels(df)

#adds the daily_close_comparison class i made [which the day checker is not working btw]
#df = daily_close_compare.create_anna(df)
df = df.drop('date',axis=1)

def jugjug_fucking_bowdown(df):
    for col in df.columns:
        df = df.rename(columns={col:col.lower()})
    return df

#make all columns lower case
df = jugjug_fucking_bowdown(df)
old_df= jugjug_fucking_bowdown(old_df)
df

if 'time' in old_df.columns:
    old_df = old_df.set_index('time')
old_df

# cheack lens of columns

print(len(df.columns),len(old_df.columns))

# this will get em in order
df = df[old_df.columns]

## mix em together

# make the indice datetime
df.index     = pd.to_datetime(df.index)
old_df.index = pd.to_datetime(old_df.index)

#last value on training data
last_value   = old_df.index[-1]
#create mask
df           = df[df.index>last_value]

#mix_old and new
df = old_df.append(df)

#import seaborn as sns
#sns.heatmap(df.isnull())

df

# {BEGIN MODEL FUNCTIONS}

## [inputs]

model_name = long_model
import mover #.................................................................................
model,info,bt,was_long,was_scaled,up_mpl,dn_mpl = mover.load_model(long_model)


## scale 

from sklearn.preprocessing import StandardScaler

def scale(df):
    scale = StandardScaler()
    scaled= scale.fit_transform(df)
    sdf   = pd.DataFrame(scaled,columns=df.columns)
    sdf.index = df.index
    return sdf


def get_pred(df,was_long,was_scaled):
    #scale input data if it was scaled
    if was_scaled:
        input_data = scale(df) 
        print('scaled')
    else:
        input_data = df.copy()
        print('not scaled')
    #name it long if it was long
    if was_long:
        name = 'long_'+model_name
    else:
        name = 'short_'+model_name

    # predictions

    pred = model.predict(input_data)
    pred

    name

    ## make a prediction dataFrame

    pdf = df[['open','close','high','low']]
    pdf[name] = pred
    pdf[name] = pdf[name].replace(True,1).replace(1,pdf.close)

    return [name,pred,pdf]



model_name = long_model
import mover #.................................................................................
model,info,bt,was_long,was_scaled,up_mpl,dn_mpl = mover.load_model(model_name)


model,info,bt,was_long,was_scaled,up_mpl,dn_mpl = mover.load_model(model_name)
name,pred,pdf = get_pred(df,was_long,was_scaled)

pdf

short_model

model_name = short_model

model,info,bt,was_long,was_scaled,up_mpl,dn_mpl = mover.load_model(model_name)
sname,spred,spdf = get_pred(df,was_long,was_scaled)

## mix scaled

pdf[sname] = spdf[sname]
pdf

import cufflinks as cf
cf.go_offline(connected=False)
def sola(df):
    return df.iplot(theme='solar',fill=True)


sola(pdf[[sname,'close',name]])


#highlight de boo-LEANS!
def hl(df):
    def highlight(boo):
        criteria = boo == True
        return['background-color: green'if i else '' for i in criteria]
    df = df.style.apply(highlight)
    return df

#prediction df for highlighted booleans
predf = df[['open','close','low','high']]
predf[name] = pred
predf[sname]= spred
#plot most recent data on top
hl(predf[::-1])

## on off plot and add to predf.. then ya done

# this creates a bionary plot thats on when long model says long
# until short model says short...
predf = predf.reset_index()
predf['bionary'] = False
for i in range(1,len(predf)):
    #name is long
    if predf[name][i] >0:
        predf['bionary'][i] = True
    #sname is short
    elif predf[sname][i] >0:
        predf['bionary'][i] = False
    else:
        predf['bionary'][i] = predf['bionary'][i-1]

predf = predf.set_index('index')

# create one that does the oposite
predf['short_bionary'] = predf['bionary']==False

#scale em
predf['bionary']       = predf['bionary'].replace(True,1).replace(1,predf.close)
predf['short_bionary'] = predf['short_bionary'].replace(True,1).replace(1,predf.close)
# plot em
bionary_predf = predf[['short_bionary','close','bionary']]


sola(bionary_predf)

ouput = [bionary_predf,pdf,]