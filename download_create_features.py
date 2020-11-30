

api = '26815f601e2c459e55a4510a897ea5dd'
try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen

import json

def get_jsonparsed_data(url):
    """
    Receive the content of ``url``, parse it as JSON and return the object.

    Parameters
    ----------
    url : str

    Returns
    -------
    dict
    """
    response = urlopen(url)
    data = response.read().decode("utf-8")
    return json.loads(data)


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def save_dat_data(dtype,ticker,df):
    sheet_name = (ticker+'_'+dtype.replace('/','_'))
    p = 'fundamentals_data/'
    if not os.path.exists(p):
        os.mkdir(p)
    df.to_csv(p+sheet_name+'.csv')
    
def get_some(dtype,ticker):
    url = ("https://financialmodelingprep.com/api/v3/{}/{}?apikey=26815f601e2c459e55a4510a897ea5dd").format(dtype,ticker)
    data = get_jsonparsed_data(url)
    df = pd.DataFrame(data)
    save_dat_data(dtype,ticker,df)
    return df 
#dtype = 'income-statement-growth'
#ticker = 'AAPL'
def get_quarter(dtype,ticker):
    url = ("https://financialmodelingprep.com/api/v3/{}/{}?period=quarter&apikey=26815f601e2c459e55a4510a897ea5dd").format(dtype,ticker)
    data = get_jsonparsed_data(url) #.......................^ thats all you change
    df = pd.DataFrame(data)
    save_dat_data(dtype,ticker,df)
    return df 

def get_news():
    url = ("https://financialmodelingprep.com/api/v3/stock-news?tickers=AAPL,FB,GOOG,AMZN&apikey=26815f601e2c459e55a4510a897ea5dd")#.format(ticker_list)
    data = get_jsonparsed_data(url) #.......................^ thats all you change
    df = pd.DataFrame(data)
    save_dat_data(dtype,ticker,df)
    return df 

def get_price(ticker):
    pdf = get_some('historical-price-full',ticker)
    price_dic = pdf.historical
    pcdf = pd.DataFrame(dict(price_dic)).T.set_index('date')
    dtype = 'historical-price-full'
    save_dat_data(dtype,ticker,pcdf)
    return pcdf

ticker = 'EVSI'
df = get_price(ticker)
df.head()

df.tail()

# lets lose non-usefull data

df = df.drop('label',axis=1)
df = df[::-1]

## create features

### riz-variations

buys = []
sells = []
def make_riz_targets(df,buy_thresh = 11,sell_thresh = 89):
    buythresh = 'below_buy_thresh:'+str(buy_thresh)
    sellthresh = 'above_sell_thresh:'+str(sell_thresh)
    df[buythresh] = df.riz < buy_thresh
    df[sellthresh] = df.riz > sell_thresh
    buy_corner = 'buy_corner:'+str(buy_thresh)
    sell_corner = 'sell_corner:'+str(sell_thresh)
    df[buy_corner] = (df[buythresh].shift() == True) & (df.riz > df.riz.shift())
    df[sell_corner] = (df[sellthresh].shift() == True) & (df.riz < df.riz.shift())
    buys.append(buy_corner)
    sells.append(sell_corner)
    #add touch eventually
    buy_touch = 'buy_touch:'+str(buy_thresh)
    sell_touch = 'sell_touch:'+str(sell_thresh)
    df[buy_touch] = (df[buythresh].shift()==False) & (df[buythresh]==True)
    df[sell_touch] = (df[sellthresh].shift()==False) & (df[sellthresh]==True)
    buys.append(buy_touch)
    sells.append(sell_touch)
    return df


df = df.rename(columns={'open':'Open','close':'Close','high':'High','low':'Low'})

import pandas_ta as pta

df['riz'] = pta.momentum.rsi(df.Close,length=2)
df.head()

#default val
df = make_riz_targets(df)
df = make_riz_targets(df,30,70)
df = make_riz_targets(df,20,80)
df = make_riz_targets(df,15,85)
df = make_riz_targets(df,40,60)
df.head()

## i can make my own vwap riz... sweet

import ta

ddf = df.copy

df['vwap_riz'] = pta.momentum.rsi(df['vwap'])
df['vwap_riz_ema']= pta.ema(df['vwap_riz'])
df['vwap_ab_avg'] = df['vwap_riz']>df['vwap_riz_ema']
df['vwap_bl_avg'] = df['vwap_riz']<df['vwap_riz_ema']

df

## im only going to do 2 different sto's b/c they kill so much of the df

df

df = df.join(pta.momentum.stoch(df.High,df.Low,df.Close))




multi = 2
df = df.join(pta.momentum.stoch(df.High,
                                      df.Low,
                                      df.Close,
                                      fast_k=14*multi,
                                     slow_k=5*multi,
                                     slow_d=3*multi))
multi = 3
df = df.join(pta.momentum.stoch(df.High,
                                      df.Low,
                                      df.Close,
                                      fast_k=14*multi,
                                     slow_k=5*multi,
                                     slow_d=3*multi))

len(df.columns)

df['trip_sto_bl30'] = df['STOCHFk_42']<30
df['trip_sto_ab50'] = df['STOCHFk_42']>50
df['doub_sto_ab50'] = df['STOCHFk_28']>50
df['sto_below30']   = df['STOCHFk_14']<30
df['trp_sto_ab_50n_short_below'] = (df['trip_sto_ab50']==True) & (df['sto_below30']==True)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


### sto_dif

def hl(df):
    def highlight(boo):
        criteria = (boo==True)
        return['background-color: green' if i else '' for i in criteria]
    df = df.style.apply(highlight)
    return df

hl(df)

## add targets

#first ill seperate the features
features = df.copy()

## saving the dirty raw check point

path = 'data/'
df.to_csv(path+ticker+'.csv')
os.listdir()





import seaborn as #sns
#sns.heatmap(df.isnull())
#df = df.dropna(axis=0).isnull()
#sns.heatmap(df.isnull())

len(df)

ticker



## and other pta shit
#### this may merge or become a seperate project but i want to know the coeficiants on the pta indicators - what do good models like? what are the favs
#### compare across securities
#### another project is to provide nothing but bools to the model and see if it can make use of it...to hard code models



### ok im dumb i didnt have to make a mask i could just transpose it and limit cols same way i match like this df.T = df.T[other_df.T.columns]

from datetime import datetime
clean_initiate= datetime.now()

info = {}
from datetime import datetime
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as #sns
import cufflinks as cf
pd.set_option('display.max_columns',1000)
pd.set_option('display.max_rows',10000)
cf.go_offline(connected=False)
stamp = str(datetime.now())
info['stamp'] = stamp
import os
path = 'data/'
info['path'] = path
sheets = os.listdir(path)
pd.DataFrame(sheets)

            
'''TARGETS!'''            
#STD TARGETS            
def std_targs(df=df,t1=0.5,t2=1,t3=2,roll_len=10,hi_rol_lo=10):
    #this reruns a data frame and targets with tuple unpacking...,
    #anything that could leak information has a:! in it...
    #roll_len = is the period of standard deviation
    #hi_rol_lo = is the rolling period of which to pull futur Highs and Lows
    df['STD'] = df['Close'].rolling(roll_len).std()
    #name the up targets
    ut1 = ('STDR'+str(hi_rol_lo)+':up_target:_'+str(t1)+'!')
    ut2 = ('STDR'+str(hi_rol_lo)+':up_target:_'+str(t2)+'!')
    ut3 = ('STDR'+str(hi_rol_lo)+':up_target:_'+str(t3)+'!')
    #name the down targets
    dt1 = ('STDR'+str(hi_rol_lo)+':down_target:_'+str(t1)+'!')
    dt2 = ('STDR'+str(hi_rol_lo)+':down_target:_'+str(t2)+'!')
    dt3 = ('STDR'+str(hi_rol_lo)+':down_target:_'+str(t3)+'!')
    #create the std values
    st1 = df['STD'] * t1
    st2 = df['STD'] * t2
    st3 = df['STD'] * t3
    
    #create the targets/stops
    df[ut1] = (df['High'] +  st1)
    df[dt1] = (df['Low'] -  st1)
    df[ut2] = (df['High'] +  st2)
    df[dt2] = (df['Low'] -  st2)
    df[ut3] = (df['High'] + st3)
    df[dt3] = (df['Low'] -  st3)

    #now create a count for each one with an upside down dataframe!!! yeet!
    df = df[::-1]
    rollingh = ('High_ahead!')#,str(hi_rol_lo)+'!')
    rollingl = ('Low_ahead!')#,str(hi_rol_lo)+'!')
    df[rollingh] = df.High.rolling(hi_rol_lo).max()
    df[rollingl] = df.Low.rolling(hi_rol_lo).min()
    df = df[::-1]
    #now we run bools on weather they are in range
    
    #name the new columns
    uth1 = ut1.replace('!','_HIT!')
    uth2 = ut2.replace('!','_HIT!')
    uth3 = ut3.replace('!','_HIT!')
    dth1 = dt1.replace('!','_HIT!')
    dth2 = dt2.replace('!','_HIT!')
    dth3 = dt3.replace('!','_HIT!')
    #make booleans of the action!
    df[dth3] = df[rollingl] < df[dt3]
    df[dth2] = df[rollingl] < df[dt2]
    df[dth1] = df[rollingl] < df[dt1]
    #i put Close in the middle for visual efffect evaluating the frame!
    #df['cClosee'] = df['Close']
    df[uth1] = df[rollingh] > df[ut1]
    df[uth2] = df[rollingh] > df[ut2]
    df[uth3] = df[rollingh] > df[ut3]
    
    #ts = [ut1,ut2,ut3,dt1,dt2,dt3]
    ts = ['Close',dt3,dt2,dt1,ut1,ut2,ut3]
    #ths = [uth1,uth2,uth3,dth1,dth2,dth3]
    ths = ['Close',dth3,dth2,dth1,uth1,uth2,uth3]
    print('these ARE targets!:\n',ts)
    return df,ths,ts

def add_targets(df):
    #df = df[['High','Low']]
    df['plus_10!'] = df['High'] + (df['High']* .1)
    df['minus_10!'] = df['Low'] - (df['Low']* .1)
    df['plus_20!'] = df['High'] + (df['High']* .2)
    df['minus_20!'] = df['Low'] - (df['Low']* .2)
    df['minus_5!'] = df['Low'] - (df['Low']* .05)
    df['plus_5!'] = df['High'] + (df['High']* .05)
    df['minus_5!'] = df['Low'] - (df['Low']* .05)
    df
    
    df = df[::-1]
    df['Highlast_10!'] = df['High'].rolling(10).max()
    df['Lowlast_10!'] = df['Low'].rolling(10).min()
    df  = df[::-1]

    df['uped_10!'] = df['plus_10!'] <= df['Highlast_10!']
    df['downed_10!'] = df['minus_10!'] >= df['Lowlast_10!']
    df['uped_20!'] = df['plus_20!'] <= df['Highlast_10!']
    df['downed_20!'] = df['minus_20!'] >= df['Lowlast_10!']
    df['uped_5!'] = df['plus_5!'] <= df['Highlast_10!']
    df['downed_5!'] = df['minus_5!'] >= df['Lowlast_10!']
    return(df)

## scale and plot functions
from sklearn.preprocessing import StandardScaler    
def scale(df):
    '''returns your data frame scaled!'''
    scale = StandardScaler()
    scaled = scale.fit_transform(df)
    sdf = pd.DataFrame(scaled,columns=df.columns)
    return sdf

'''
def sola(df,title=None):
    return df.iplot(theme='#solar',fill=True,title=title)
'''


#sns.heatmap(df.isnull())

#sns.heatmap(df.isnull())

#### SAVE TO CSV

df = df.reset_index()#.drop('index',axis=1)


df.head()




df.head()


feature_frame = df.copy()

# TARGETS

stdf = std_targs(df,t1=2,t2=3,t3=4)
df = df.merge(stdf[0])
targs = []
for i in df.columns:
    if '!' in i:
        targs.append(i)


#sola(df[stdf[2]])


ts = stdf[1]
targetframe = df[ts]
targetframe.replace(True,targetframe['Close'])

i = ts[2]
for i in targetframe.columns:
    targetframe[i] = targetframe[i].replace(True,targetframe['Close'])
    targetframe[i] = targetframe[i].replace(1,targetframe['Close'])

ups = []
downs=[]
for i in targetframe.columns:
    if 'up' in i:
        ups.append(i)
    if 'down' in i:
        downs.append(i)

#sola(targetframe)

pd.DataFrame(ts)

up = ts[2]
down = ts[-2]
#sola(targetframe[['Close',up,down]])

downs.append('Close')

ups.append('Close')

#sola(targetframe[ups])

#sola(targetframe[downs])

## i think i need some more rare targets...

stdf[0]

#sns.heatmap(df.isnull())

pd.DataFrame(targs)

## longer term targets

## IDEAS
### >5ma > 10ma
### >price > 5ma
### > price > 10ma


df['Close']

df



df = add_targets(df)

uped = []
downed = []
for i in df.columns:
    if 'uped' in i:
        uped.append(i)
    if 'downed' in i:
        downed.append(i)
print(uped)
print(downed)



df





scales = []
for i in df.columns:
    if 'uped' in i:
        df['scale_'+i] = df[i].replace(True,1).replace(1,df['Close'])
        scales.append('scale_'+i)
    if 'downed' in i:
        df['scale_'+i] = df[i].replace(True,1).replace(1,df['Close'])
        scales.append('scale_'+i)

pd.DataFrame(scales)

scales

scales.append('Close')


#sola(df[[scales[0],'Close',scales[3]]])

#sola(df[[scales[5],'Close',scales[4]]])

# long term targets

df

df['ma5'] = df['Close'].rolling(5).mean()
df['ma10']= df['Close'].rolling(10).mean()
df['ma20'] = df['Close'].rolling(5).mean()
df['ma40']= df['Close'].rolling(10).mean()
#sola(df[['ma5','ma10','Close']])

df['ma5_above_ma10'] = df['ma5'] > df['ma10']
df['ma5_below_ma10'] = df['ma5'] < df['ma10']
df['scale_ma5_above'] = df['ma5_above_ma10'].replace(True,1).replace(1,df.Close)
df['scale_ma5_below'] = df['ma5_below_ma10'].replace(True,1).replace(1,df.Close)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#sola(df[['scale_ma5_below','Close','scale_ma5_above']])

df['ma5_above_ma10_in5!']  = df['ma5_above_ma10'].shift(-5)   
df['ma5_below_ma10_in5!']  = df['ma5_below_ma10'].shift(-5) 
df['scale_ma5_above_in5!'] = df['scale_ma5_above'].shift(-5)
df['scale_ma5_below_in5!'] = df['scale_ma5_below'].shift(-5)

#sola(df[['ma5_above_ma10_in5!','ma5_below_ma10_in5!']])

#df['scale_ma5_above_in5!'] = df['ma5_above_ma10_in5!'].replace(True,1).replace(1,df.Close)
#df['scale_ma5_below_in5!'] = df['ma5_below_ma10_in5!'].replace(True,1).replace(1,df.Close)

#sola(df[['scale_ma5_below_in5!','Close','scale_ma5_above_in5!']])

df = df.dropna(axis=0)
#sns.heatmap(df.isnull())



start = df.index[0]
last =  df.index[-1]
print('starts:',start,'ends:',last)

mask = (feature_frame.index >= start) & (feature_frame.index <= last)
fedf = feature_frame[mask]

print('feature_frame is:',len(fedf))
print('target_frame is:',len(df))

df.head()

fedf.head()

# save em!

sdf = scale(fedf.drop('date',axis=1))
sdf['date'] = df['date']

sdf
#sola(sdf)

#og = sheet.split('/')[1]
og = ticker

path = 'clean_data/'
if not os.path.exists(path):
    os.mkdir(path)

sdf.to_csv(path+'scaled_features_'+og)
fedf.to_csv(path+'features_'+og)
df.to_csv(path+'targets_'+og)
scaled_feature_frame = sdf.copy()

clean_compleate = datetime.now()
notebook_runtime = clean_compleate - clean_initiate
print('notebook ran in: ',notebook_runtime)

df


