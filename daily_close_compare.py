#extract the date...
#df['date'] = [thing.date() for thing in df['time']]


def create_anna(df):
    '''
    this creates the dailyclose comparison strat, as close as it can,
    be without repainting
    '''
    #identigy the first bar of the day, 
    df['next_day']= df['date']!= df['date'].shift()
    
    df = df.copy()
    df['last_close'] = None
    df['delta']      = None
    #on the first bar of the day we create our delta and deltapct
    for i in range(1,len(df)):
        if df['next_day'][i]==True:
            df['last_close'][i] = df['close'][i-1]
            df['delta'][i]      = (df['close'][i]-df['last_close'][i])/df['last_close'][i]

        else:
            df['last_close'][i] = df['last_close'][i-1]
            df['delta'][i]      = df['delta'][i-1]
    df = df.copy()
    #.................................................................................................
    
    #df['pred_up'] = df['prediction']>0
    df['delta_up']= df['delta']     >0
    return df
#df = create_anna(df)    
#df = df[['close','next_day','prediction','delta','date']]
#hl(df)
