import pandas as pd

df = pd.DataFrame()      
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
    uth1 = ut1.replace('!','TARGET!')
    uth2 = ut2.replace('!','TARGET!')
    uth3 = ut3.replace('!','TARGET!')
    dth1 = dt1.replace('!','TARGET!')
    dth2 = dt2.replace('!','TARGET!')
    dth3 = dt3.replace('!','TARGET!')
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

    df['TARGET:uped_10!'] = df['plus_10!'] <= df['Highlast_10!']
    df['TARGET:downed_10!'] = df['minus_10!'] >= df['Lowlast_10!']
    df['TARGET:uped_20!'] = df['plus_20!'] <= df['Highlast_10!']
    df['TARGET:downed_20!'] = df['minus_20!'] >= df['Lowlast_10!']
    df['TARGET:uped_5!'] = df['plus_5!'] <= df['Highlast_10!']
    df['TARGET:downed_5!'] = df['minus_5!'] >= df['Lowlast_10!']
    return(df)

def ma_targets(df,fast=5,slow=10,ahead_period=10):
    df['fastma'] = df['Close'].rolling(fast).mean()
    df['slowma'] = df['Close'].rolling(slow).mean()
    df['fastup']= df['fastma']>df['slowma']
    
    tname = 'TARGET:fast:'+str(fast)+'_up_slow:'+str(slow)+'in:'+str(ahead_period)
    dtname = 'TARGET:fast:'+str(fast)+'_down_slow:'+str(slow)+'in:'+str(ahead_period)
    df[tname] = df['fastup'].shift(-ahead_period)==True
    df[dtname]= df['fastup'].shift(-ahead_period)==False
    return df