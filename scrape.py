#https://financialmodelingprep.com/developer/docs/


api = '26815f601e2c459e55a4510a897ea5dd'
from urllib.request import urlopen
import json
import os
import pandas as pd

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

def save_dat_data(dtype,ticker,df):
    sheet_name = (ticker+'_'+dtype.replace('/','_'))
    p = 'downloaded_data/'
    if not os.path.exists(p):
        os.mkdir(p)
    df.to_csv(p+sheet_name+'.csv')
    
def get_some(dtype,ticker):
    url = ("https://financialmodelingprep.com/api/v3/{}/{}?apikey=26815f601e2c459e55a4510a897ea5dd").format(dtype,ticker)
    data = get_jsonparsed_data(url)
    df = pd.DataFrame(data)
    save_dat_data(dtype,ticker,df)
    return df 

def get_quarter(dtype,ticker):
    url = ("https://financialmodelingprep.com/api/v3/{}/{}?period=quarter&apikey=26815f601e2c459e55a4510a897ea5dd").format(dtype,ticker)
    data = get_jsonparsed_data(url) #.......................^ thats all you change
    df = pd.DataFrame(data)
    save_dat_data(dtype,ticker,df)
    return df 