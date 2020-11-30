import pandas as pd
import shutil
import os

def pull_info(model):
    #info df
    ipath = 'PREDICTION_DATA/model_preformance.csv'
    idf = pd.read_csv(ipath,index_col='index')
    #spesific model
    return pd.DataFrame(idf[model])

#model mover

def move_model(model_name):
    def pull_info(model):
        #info df
        ipath = 'PREDICTION_DATA/model_preformance.csv'
        idf = pd.read_csv(ipath,index_col='index')
        #spesific model
        return pd.DataFrame(idf[model])


    print(model_name)
    #model info
    model_info = pull_info(model_name)


    #backtest results
    btpath         = 'backtest_data/results.csv'
    btdf           = pd.read_csv(btpath,index_col='Unnamed: 0')
    back_test_data = btdf[btdf['model']==model_name]

    #pickle
    pickle = 'pickles/'+model_name

    #data_dir
    clean_data_path = 'clean_data/'

    # CREATE DIRECTORYS

    all_star    = 'all_stars/'
    if not os.path.exists(all_star):
        os.mkdir(all_star)
        print('CREATING ALLSTARS')
    #create model_directory
    model_dir   = all_star+model_name+'/'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        print('creating model directory')


    # MOVE EVERYTHING OVER 




    # Model_info

    model_info.to_csv(model_dir+'info.csv')
    model_info
    print('saved info')
    # Backtest Data

    back_test_data.to_csv(model_dir+'backtest_data.csv')
    back_test_data
    print('saved backtestdata')

    # Pickle



    src = pickle
    dst = model_dir+'pickle'
    shutil.copy2(src,dst)
    print('moved pickle')
    # Original Gangsta Data

    new_data_path = 'all_stars/'+clean_data_path

    #create data directory
    if not os.path.exists(new_data_path):
        shutil.copytree(clean_data_path,new_data_path)
        print("moved data")
        
    print('ALL DONE!!')


def load_model(model_name,path='all_stars/'):
    '''
    this function returns a tuple with the model and any relivent data.
    RETURNS:
    1. model_pickle
    2. info
    3. backtest results
    4. was it long?
    5. was it scaled? 
    6. best up atr multiple (for sharpe ratio)
    7. best down atr multiple (for sharpe ratio)


    '''
    
    
    path = path+model_name+'/'

    import pickle
    #load model pickle 
    model_path = path+'pickle'
    model = pickle.load(open(model_path,'rb'))

    #load info 
    info_path = path+'info.csv'
    info = pd.read_csv(info_path,index_col='index').dropna(axis=0)
    print(info[model_name]['model_name'])
    
    #load backtest data
    btpath = path+'backtest_data.csv'
    backtest_data = pd.read_csv(btpath).drop('Unnamed: 0',axis=1).sort_values('sharperatio',ascending=False)
    up_multiple,dn_multiple = backtest_data['up_multiple'][0], backtest_data['dn_multiple'][0]
    
    #long or short?
    was_long   = 'up' in info[model_name]['model_name']
    if was_long:
        print('TYPE  : long')
    else:
        print('TYPE  : short')

    #was scaled?
    was_scaled = 'scaled' in info[model_name]['sheet']
    if was_scaled:
        print('SCALED:',was_scaled)
    return [model,info,backtest_data,was_long,was_scaled,up_multiple,dn_multiple]

