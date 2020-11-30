import pandas as pd
import numpy as np

import os

from fastquant import backtest

import pandas_ta as pta

pd.set_option('display.max_rows',None)



from sklearn.preprocessing import StandardScaler  
def scale(df):
    '''returns your data frame scaled!'''
    scale = StandardScaler()
    scaled = scale.fit_transform(df)
    sdf = pd.DataFrame(scaled,columns=df.columns)
    return sdf





path  ='PREDICTION_DATA/predictions/'
pdir = os.listdir(path)
pd.DataFrame(pdir)



def roll_baby(col,mdf,n):
    '''
    this function creates variations of potential strats or triggers,
    BUT COULD ALSO BE USED to give more in_depth features for ml_models
    on Coeficiants that they like... research is going to have to answer 
    these questions
    ==RESEARCH PROJECTS==
    1. what are the best ways to set up strats
    2. whatare the best features in all the strats
    3. does variations of features help the models better know whats going on?
    
    '''
    n_str    = str(n)
    sumname = (col+'_sum:'+n_str)
    meanname= (col+'_mean:'+n_str)
    summean = (col+'_summean:'+n_str)
    mdf[sumname] = mdf[col].rolling(n).sum()
    mdf[meanname]= mdf[col].rolling(n).mean()
    mdf[summean] = mdf[sumname].rolling(n).mean()


#print(i)
def mix_load_features(i):
    '''
    this function is a combination of a bunch of things to iterate through in a loop
    1.loads model predictions data
    2.merges price_data with predictions
    3. creates 9 variations per model
    '''
    sdf = pd.read_csv(path+i,index_col='date').drop('Unnamed: 0',axis=1)

    #extrat log model
    log    = [m for m in sdf.columns if 'Log' in m][0]
    tree   = [m for m in sdf.columns if 'Tree' in m][0]
    forest = [m for m in sdf.columns if 'forest' in m][0]
    models = [log,tree,forest]
    print(log,tree,forest)

    #set index 
    sdf.index = pd.to_datetime(sdf.index)

    df = sdf#[['Close',m]]
    df['closee'] = df['Close']
    df

    #price data
    #FIND FEATURES DATA
    features_data = [s for s in os.listdir('clean_data/') if 'features' in s][0]
    prdf = pd.read_csv('clean_data/'+features_data,index_col='time') #TODO: FIX THIS
    prdf.index.name = 'date'
    prdf.index = pd.to_datetime(prdf.index)
    prdf = prdf[['Open','High','Low','Close']]
    prdf['cclose'] = prdf['Close']
    prdf

    # use the index as a mask...
    ##### (which may not be nessisary now

    mask = df.index
    print('mask is:',len(mask))
    print('df is:',len(df))
    mask

    #create a mask the easy way
    prdft = prdf.T[mask]
    prdf  = prdft.T
    print('and now...\n price_df is:',len(prdf))
    prdf
    #mix the data (THA EASY WAY BECASUE JOIN IS A FUCK UP)
    for col in df.columns:
        prdf[col] = df[col]
    prdf
    print('you wnat to see the heat map TOTALLY white where the closes meet ')
    #sns.heatmap(df.corr())
    bullshit = ['cclose','closee']
    df = prdf.drop(bullshit,axis=1)
    df.tail()
    return df,models



def make_variations(df,models):
    #create models df
    mdf = df[models]

    mdf = mdf.replace(True,1)
    #hl(mdf)

    log
    #create variations for each strat...
    for m in models:
        roll_baby(m,mdf,3)
        #roll_baby(m,mdf,6)
        roll_baby(m,mdf,9)
        
        
    logs    = [m for m in mdf.columns if 'Log' in m]
    trees   = [m for m in mdf.columns if 'Tree' in m]
    forests = [m for m in mdf.columns if 'forest' in m]
    #logs.insert(0,'Close')
    #trees.insert(0,'Close')
    #forests.insert(0,'Close')
    modelz = [logs,trees,forests]
    ##sola(mdf[logs])
    ##sola(mdf[trees])
    ##sola(mdf[forests])
    
    
    df['Date'] = df.index
    df = df.merge(mdf)
    df = df.set_index('Date')
    return df.copy()
    


#i = pdir[0]

        
def loop_prep(i):
    '''
    takes i - which is a backtest csv from the prediction dir
    TODO:
    >make the loop identify if 'up' in df name
    > if 'up' not in name then reverse open/close/low/high
       >max_high = take the max from high
       > maxplus = add one to max_high
       > open/close/low/high = maxplus - oclh
       > BAM you have a reverse backtest that will run in the current arcitechure
    this does all the prepwork to run the backtest loop.
    the point of this function is for it so sit in a loop
    that runs every target dataframe in the prediction directory
    1. mix load features
    2. identify the target(to remove it later)
    3. ouputs a ready backtest df with models...
    '''
    #load tha thing
    df,models = mix_load_features(i)
    #check if the model is a downer and inverse price data if so
    if 'down' in i:
        print('its a downer')
        maxplus = df['High'].max() + 1
        #price columns
        pcols   = ['Open','High','Low','Close']
        for col in pcols:
            df[col] = maxplus - df[col] 
            print(maxplus)
    else:
        print('its upper')

    #identify the target TODO: save the target to results df
    target = [c for c in df.columns if '!' in c][0]
    df.head()
    models
    
    #create a bestest ready df
    btdf = df.drop(target,axis=True).replace(True,1)
    return df, models, btdf,target

## run terinary strat
#### which recieves +1 as a buy indication and -1 as sell indication

def back_test_things(m,to_plot=False,up_multiple=2,dn_multiple=2):
    '''
    this function does the backtest things
    m is for model
    '''
    df['atr'] = pta.atr(df.High,df.Low,df.Close)
    bdf = df[[m,'atr','High','Low','Close']].dropna(axis=0)
    #sns.heatmap(bdf.isnull())

    bdf['atr_up'] = (bdf['Close']+bdf['atr']*up_multiple)
    bdf['atr_down']=(bdf['Close']-bdf['atr']*dn_multiple)

    ##### later create a function that makes variations of ups and downs

    #### to_be an ATR -LOOP

    # STRAT FUNCTION
    ### starts here...

    '''llllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllll'''
    '''inputs'''
    buy = m#'Tree165473'
    t_up= 'atr_up'
    t_dn= 'atr_down'
    #SO this needs to go into a function and will be part of the backtest loop for each model
    # atr_targs and stop will be inputs so i can iterate through 
    if bdf.index.name == 'date':
        bdf = bdf.reset_index()

    bdf['trac'] = False
    bdf['targ'] = bdf[t_up]
    bdf['stop'] = bdf[t_dn]

    for i in range(1,len(bdf)):
        if (bdf[buy][i] == True)&(bdf['trac'][i-1]==False):
            bdf['trac'][i] = True
            bdf['targ'][i] = bdf[t_up][i]
            bdf['stop'][i] = bdf[t_dn][i]
        elif (bdf['trac'][i-1] == True) & ((bdf['Low'][i]<bdf['stop'][i-1])|(bdf['High'][i]>bdf['targ'][i-1])):
            bdf['trac'][i] = False
        else:
            bdf['trac'][i] = bdf['trac'][i-1]
            bdf['targ'][i] = bdf['targ'][i-1]
            bdf['stop'][i] = bdf['stop'][i-1]
    #hl(bdf)

    print(s)
    tit = (s + '_'+m)
    
    #if to_plot==True:
        #sola(bdf[['stop','Close','targ']],title=tit)

    ## todo- replace 'strat' with a varable


    bdf['strat'] = 0
    for i in range(1,len(bdf)):
        if (bdf['trac'][i-1]==False)&(bdf['trac'][i]==True):
            bdf['strat'][i] = 1
        elif (bdf['trac'][i-1]==True)&(bdf['trac'][i]==False):
            bdf['strat'][i] = -1


    #hl(bdf)

    bdf  =bdf.set_index('date')#.drop(['level_0','index'],axis=1)
    bdf

    #%matplotlib
    results, history = backtest('ternary', 
                                    bdf,
                                    custom_column='strat',
                                    init_cash=1000,
                                    plot=to_plot,
                                    verbose=False,
                                    return_history=True

                               )
    results['model'] = buy
    results['target']= target
    results['sheet'] = s
    results['up_multiple'] = up_multiple
    results['dn_multiple'] = dn_multiple
    #results = results.set_index('custom_column')
    #results['buy'] = buy
    #se
    results.T

    ### so now i need to rename the thing after the strat and atr paramaters

    #### 

    #####  this is ready to become a save_function... but 


    btpath = 'backtest_data/'
    if not os.path.exists(btpath):
        os.mkdir(btpath)
    fname = 'results.csv'
    if not os.path.exists(btpath+fname):
        print('dosnt exitst')
        rdf = results
        rdf.to_csv(btpath+fname)
        print('results.csv created')
    else:
        rdf = pd.read_csv(btpath+fname).drop('Unnamed: 0',axis=1)
        print('already exists')
        rdf = rdf.append(results)
        print('apending')
        rdf.to_csv(btpath+fname)
        print('results.csv UPDATED!')


## for de bugging im going to make s one item in directory , then run the things one at a time and find out where the problem is... sothing to do with the index name most liekly

pdir[0]

## ok so i see its pointing to where it loads the data and wants to name the index 'date'



# i Actually put this into a script 
####  and this step is not nessisarry after it runs the first time
#### but for some reason it doesnt plot right if you skip it
#### ... got to find a way to split this up right

# MAIN LOOP


# MAIN LOOP

#ready to run all in the thingy
for s in pdir:

    df,models,btdf,target = loop_prep(s)
    up_multiple = 2
    dn_multiple = 2
    for m in models:
        #in the future this could easily turn into a heatmap
        #by making up_multiple and dn_multiples into an array
        #shape output results at [col][row]
        back_test_things(m,False,2,2)
        back_test_things(m,False,2,1)
        back_test_things(m,False,3,2)
        back_test_things(m,False,4,2)
        back_test_things(m,False,3,3)
        back_test_things(m,False,4,1)
        #function will go here'''