from datetime import datetime
clean_initiate= datetime.now()


info = {}
from datetime import datetime
import math
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as #sns
#import cufflinks as cf
#cf.go_offline(connected=False)
stamp = str(datetime.now())
info['stamp'] = stamp
import os
path = 'clean_data/'
info['path'] = path
sheets = os.listdir(path)
pd.DataFrame(sheets)


# seperate diff types of sheets and grab first one
scaled_features = [s for s in sheets if 'scaled' in s][0]
target_sheet    = [s for s in sheets if 'targets'in s][0]
features        = [s for s in sheets if 'features'in s][0]
print('scaled:',scaled_features,'\ntargets:',target_sheet,'\nfeatures',features)

#features in fdf
sheet = path+features#sheets[2] 
fdf = pd.read_csv(sheet)
fdf = fdf.rename(columns={'time':'date'})

#TARGETS GO IN DF
df = pd.read_csv(path+target_sheet)  #sheets[0])
df = df.rename(columns={'time':'date'})
#makeing copys in case fuck-ups
ddf = df.copy()
dfdf = fdf.copy()
df.head()

# REMOVING THE FUCKING INDEXES...

#bullshit = ['Unnamed: 0','index']
#fdf = fdf.drop(bullshit,axis=1)
#df = df.drop(bullshit,axis=1)
fdf.head()

df.head()

# targets

#CREATING LIST OF TARGETS!
targs = []
for i in df.columns:
    if '!' in i:
        targs.append(i)
tardf = pd.DataFrame(targs,columns=['targets'])
tardf

target  = targs[22]
dtarget = targs[23]
info['target'] = df[target]
print('target...............:     ',target, '\nanti-target..........:',dtarget)
print('rows:',len(df))

# split here and load in sheets..

#SPLIT BASED ON PERCENT
split_per = 0.3 * len(df)
split     = len(df) - int(split_per)

#MANUAL SPLIT
#split = 1000
split

print('lenth:',len(fdf),len(df))

print(split)

fe_first = fdf[:split]
df_first  = df[:split]

#second 
fe_second = fdf[split:]
df_second = df[split:]
#goingto have to do curent differently, by saving the columns this will be easy
#current= fdf[:-100]

#  functions

from tqdm import trange
def sheet_n_sec():
    #this saves the sheet and name to the refrence dictionary 'info'
    info['sheet'] = sheet
    info['sec'] = sheet.replace('.csv','')
    
def save_timeframe():
    #saves the timeframe to the refrence dictionary 'info' 
    tfr = str(pd.Timedelta(df.index[0])-pd.Timedelta(df.index[1]))
    tfr = tfr.split('.')[0]
    info['time_frame'] = tfr    

    
def save_start_stop():
    #this saves the start and stop dates to the dic 'info'
    start = str(df['Date'][0])
    end = str(df['Date'][-1:])
    info['start'] = start
    info['end'] = end    

def df_split(df,split_pct_fmEnd=.2):
    #splits a dataframe into 'first'&'second' based on a given percentage
    split = int(math.ceil(split_pct_fmEnd*len(df)))
    split = int(len(df) - forcast_out*split_pct_fmEnd)
    print('df_len',len(df))
    print('split is',split)
    first = df[:split]
    second = df[split:]
    print('first_ends:',first.index[-1])
    print('second_starts:',second.index[0])
    return first, second

def ttsplit(x,y):
    #train_test_split simply saves some trouble, of typeing
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)
    return x_train,x_test,y_train,y_test

'''MACHINE LEARNING MODELS'''
#model dictionary save attribtes and results of each model to be saved later
modeldic = {}
def lin_reg_fit(x,y,name=' ',notes=''):
    # i didnt add any dtargs to the metrics....i you think of any, DO IT DO IT DO IT!!!!!!!
    '''#ouputs the model, model_id, predictions & coef_df in a list, while saving all parmas to 'modeldic'
    runs a linear model and save the paramaters, outputs,
    the model,model_id, prediction, and a coeficiant dataframe '''
    indic = {}
    mstamp = str(datetime.now())
    mtype = 'LinearRegression'
    sn = mstamp.split('.')[1]
    modelid = mtype+sn
    #adding paramaters
    indic['model_id'] = modelid
    indic['model_name'] = ltarget+modelid #this is striclty beinf used for nameing
    indic['model_notes'] = 'fit_intercept=False' #notes
    indic['stamp'] = mstamp
    indic['model_type'] = mtype
    indic['features'] = str(x.columns)
    indic['target']   = ltarget
    #setting up model
    lm = LinearRegression(fit_intercept=False)
    lm.fit(x_train,y_train)
    pred = lm.predict(x_test)
    #jplt = #sns.jointplot(y_test,pred)
    coef = lm.coef_ 
    
    codf = pd.DataFrame(coef,x.columns,columns=['coefs'])
    #this would be if you wanted to rank them ....
    #for i in trange(0,len(codf)):
    #    if codf['coefs'][i] < 0:
    #        codf['coefs'][i] = codf['coefs'][i] * -1
    codf = codf.T
    
    codf#.iplot(kind='bar',theme='#solar')
    r2 = r2_score(y_test,pred)
    mae = mean_absolute_error(y_test,pred)
    indic['r2_score'] = r2
    indic['MAE'] = mae
    print('r2 score',r2)
    print('mean_absolute_er',mae)
    codf
    indic['coefs'] = codf.T['coefs'].values
    modeldic[modelid] = indic
    return [lm, modelid, pred,codf]#,codf.T#.iplot(kind='bar',theme='#solar')
def log_reg_fit(x,y,dy,name=' ',notes=''):
    #ouputs the model, model_id, predictions & coef_df in a list, while saving all parmas to 'modeldic'
    #runs a Logistic model and save the paramaters, outputs,
    #the model,model_id, prediction, and a coeficiant dataframe 
    
    indic = {}
    mstamp = str(datetime.now())
    '''model'''
    mtype = 'LogisticRegression'
    sn = mstamp.split('.')[1]
    modelid = mtype+sn
    #adding paramaters
    indic['model_id'] = modelid
    indic['model_name'] = target+modelid #this is striclty beinf used for nameing
    indic['model_notes'] = 'max_iter is on 4000'#notes
    indic['stamp'] = mstamp
    indic['model_type'] = mtype
    indic['features'] = str(x.columns)
    #setting up model
    '''object name'''
    lr = LogisticRegression(max_iter=4000)
    lr.fit(x_train,y_train)
    pred = lr.predict(x_test)
    #dont think this will work on 
    coef = lr.coef_ 
    
    codf = pd.DataFrame(coef,columns=[x.columns])#,x.columns)#,columns=['coefs'])
    #this would be if you wanted to rank them ....
    #for i in trange(0,len(codf)):
    #    if codf['coefs'][i] < 0:
    #        codf['coefs'][i] = codf['coefs'][i] * -1
    #codf = codf.T
    
    codf#.iplot(kind='bar',theme='#solar')
    
    #metrics
    clr = classification_report(y_test,pred)
    dclr= classification_report(dy_test,pred)
    print('~~~~~~~~~~~~~~~~~~~~~~~HITTNG THE TARGET~~~~~~~~~~~~~~~~~~~~~~~~~\n',clr)
    print('..............................vs................................')
    print('~~~~~~~~~~~~~~~~~~~~~~~ THE ANTI-TARGET~~~~~~~~~~~~~~~~~~~~~~~~~\n',dclr)
    comx = confusion_matrix(y_test,pred)
    print("~~~~~~~~~~~~~~~~~~~~~~[CONFUSION MATRIX:]~~~~~~~~~~~~~~~~~~~~~~~" )
    print("                 ......///",comx[0], "\\\........." )
    print("                 ...../// ",comx[1], " \\\........." )
    #classification report
    clr = classification_report(y_test,pred,output_dict=True)
    clrdf = pd.DataFrame(clr)
    indic['true_positive']=clr['True']['precision']
    indic['true_negitive']=clr['False']['precision']
    indic['weighted avg'] =clr['weighted avg']['precision']
    indic['f1_score']     =clr['weighted avg']['f1-score']
    codf
    modeldic[modelid] = indic
    return [lr, modelid, pred,clrdf,codf] #,codf.T#.iplot(kind='bar',theme='#solar')

def tree_fit(x,y,dy,name=' ',notes=''):
    indic = {}
    mstamp = str(datetime.now())
    '''model'''
    mtype = 'Tree'
    sn = mstamp.split('.')[1]
    modelid = mtype+sn
    #adding paramaters
    indic['model_id'] = modelid
    indic['model_name'] = target+modelid #this is striclty beinf used for nameing
    indic['model_notes'] = 'default-tree '#notes
    indic['stamp'] = mstamp
    indic['model_type'] = mtype
    indic['features'] = str(x.columns)
    indic['target']   = target
    #setting up model
    '''object name'''
    tree = DecisionTreeClassifier('entropy')
    tree.fit(x_train,y_train)
    pred = tree.predict(x_test)
    #metrics
    clr = classification_report(y_test,pred)
    dclr= classification_report(dy_test,pred)
    print('~~~~~~~~~~~~~~~~~~~~~~~HITTNG THE TARGET~~~~~~~~~~~~~~~~~~~~~~~~~\n',clr)
    print('..............................vs................................')
    print('~~~~~~~~~~~~~~~~~~~~~~~ THE ANTI-TARGET~~~~~~~~~~~~~~~~~~~~~~~~~\n',dclr)
    comx = confusion_matrix(y_test,pred)
    print("~~~~~~~~~~~~~~~~~~~~~~[CONFUSION MATRIX:]~~~~~~~~~~~~~~~~~~~~~~~" )
    print("                 ......///",comx[0], "\\\........." )
    print("                 ...../// ",comx[1], " \\\........." )
    #classification report
    clr = classification_report(y_test,pred,output_dict=True)
    clrdf = pd.DataFrame(clr)
    #indic['true_positive']=clr['True']['precision']
    #indic['true_negitive']=clr['False']['precision']
    #indic['weighted avg'] =clr['weighted avg']['precision']
    #indic['f1_score']     =clr['weighted avg']['f1-score']
    modeldic[modelid] = indic
    return [tree, modelid, pred,clrdf] #,codf.T#.iplot(kind='bar',theme='#solar')

def forest_fit(x,y,dy,name=' ',notes=''):
    indic = {}
    mstamp = str(datetime.now())
    '''model'''
    mtype = 'forest'
    sn = mstamp.split('.')[1]
    modelid = mtype+sn
    #adding paramaters
    indic['model_id'] = modelid
    indic['model_name'] = target+modelid #this is striclty beinf used for nameing
    indic['model_notes'] =  'running 200 estimators'#notes
    indic['stamp'] = mstamp
    indic['model_type'] = mtype
    indic['features'] = str(x.columns)
    indic['target']   = target
    #setting up model
    '''object name'''
    forest = RandomForestClassifier(200,'entropy')
    forest.fit(x_train,y_train)
    pred = forest.predict(x_test)
    #metrics
    clr = classification_report(y_test,pred)
    dclr= classification_report(dy_test,pred)
    print('~~~~~~~~~~~~~~~~~~~~~~~HITTNG THE TARGET~~~~~~~~~~~~~~~~~~~~~~~~~\n',clr)
    print('..............................vs................................')
    print('~~~~~~~~~~~~~~~~~~~~~~~ THE ANTI-TARGET~~~~~~~~~~~~~~~~~~~~~~~~~\n',dclr)
    comx = confusion_matrix(y_test,pred)
    print("~~~~~~~~~~~~~~~~~~~~~~[CONFUSION MATRIX:]~~~~~~~~~~~~~~~~~~~~~~~" )
    print("                 ......///",comx[0], "\\\........." )
    print("                 ...../// ",comx[1], " \\\........." )
    #classification report
    clr = classification_report(y_test,pred,output_dict=True)
    clrdf = pd.DataFrame(clr)
    indic['true_positive']=clr['True']['precision']
    indic['true_negitive']=clr['False']['precision']
    indic['weighted avg'] =clr['weighted avg']['precision']
    indic['f1_score']     =clr['weighted avg']['f1-score']
    modeldic[modelid] = indic
    return [forest, modelid, pred,clrdf] #,codf.T#.iplot(kind='bar',theme='#solar')


def lin_validate(lm,x):
    pred = lm[0].predict(x)
    predf[lm[1]] = pred
    redf = predf[['Close',lm[1]]]
    redf['time'] = ddf['Date']
    redf.set_index('time',inplace=True)
    r2 = r2_score(y,pred)
    mae = mean_absolute_error(y,pred)
    modeldic[lm[1]]['r2_score'] = r2
    modeldic[lm[1]]['MAE'] = mae
    #modeldic[lm[1]]['target'] = ltarget
    print('r2 score',r2)
    print('mean_absolute_er',mae)
    #codf
    #modeldic[lm[1]]['coefs'] = codf.T['coefs'].values
    #modeldic[lm[1]] = indic
    #sola(redf)
    #sns.jointplot(pred,df_second[ltarget])

def binary_validate(mdl,x):
    pred = mdl[0].predict(x)
    predf[mdl[1]] = pred
    redf = predf[[target,'Close',mdl[1]]]
    redf['time'] = ddf['Date']
    redf.set_index('time',inplace=True)
    sdf = scale(redf)
    sdf['time'] = ddf['Date']
    sdf.set_index('time',inplace=True)
    #metrics
    clr = classification_report(y,pred)
    dclr= classification_report(dy_test,pred)
    print('~~~~~~~~~~~~~~~~~~~~~~~HITTNG THE TARGET~~~~~~~~~~~~~~~~~~~~~~~~~\n',clr)
    print('..............................vs................................')
    print('~~~~~~~~~~~~~~~~~~~~~~~ THE ANTI-TARGET~~~~~~~~~~~~~~~~~~~~~~~~~\n',dclr)
    comx = confusion_matrix(y,pred)
    print("~~~~~~~~~~~~~~~~~~~~~~[CONFUSION MATRIX:]~~~~~~~~~~~~~~~~~~~~~~~" )
    print("                 ......///",comx[0], "\\\........." )
    print("                 ...../// ",comx[1], " \\\........." )
    #the real price plot
    scale2close= mdl[1]+'$scale'
    redf[scale2close] = redf[mdl[1]].replace(True,1)
    redf[scale2close] = redf[scale2close].replace(1,redf['Close'])
    #sola(redf[['Close',scale2close]])
    #scaled close
    #sola(sdf.drop(target,1))
    #scaled bionary
    #sola(sdf[[target,mdl[1]]])
    clr = classification_report(y,pred,output_dict=True)
    clrdf = pd.DataFrame(clr)
    modeldic[mdl[1]]['true_positive']=clr['True']['precision']
    modeldic[mdl[1]]['true_negitive']=clr['False']['precision']
    modeldic[mdl[1]]['weighted avg'] =clr['weighted avg']['precision']
    modeldic[mdl[1]]['f1_score']     =clr['weighted avg']['f1-score']
    modeldic[mdl[1]]['f1_score']
    #modeldic[mdl[1]] = dic
    return redf,pred



 
    
def scale(df):
    '''returns your data frame scaled!'''
    scale = StandardScaler()
    scaled = scale.fit_transform(df)
    sdf = pd.DataFrame(scaled,columns=df.columns)
    return sdf


'''
def #sola(df):
    return df#.iplot(theme='#solar',fill=True)
'''

def mix_info(modeldic,info):
    #takes the model dict and ouputs dataFrame
    # with the global data from 'info' dict
    modf = pd.DataFrame(modeldic)
    modf = modf.T
    for i in info.keys():
        modf[i] = info[i]
    modf = modf.T    
    modf.index.name = 'index'
    return modf

def save_info(csvname='model_preformance'):
    cname = (csvname+'.csv')
    info_path = 'PREDICTION_DATA/'
    if not os.path.exists(info_path):
        os.mkdir(info_path.replace('/',''))
    yn = 'y'#input('do you want to save these results y/n?')
    if yn == 'y':
        yyn = 'loopy'#input('is there anything you want to add?')
        if yyn == 'n':
            pass
        else:
            #add_stuff = input('type some')
            add_some = yyn
            info['after_thoughts'] = add_some
            info['sheet']          = sheet
        #forming the dataframe
        modf = mix_info(modeldic,info)
        if cname in os.listdir(info_path):
            print('its here ill update it')
            mpdf = pd.read_csv(info_path+cname,index_col='index')
            for i in modf.columns:
                mpdf[i] = modf[i]
                mpdf.to_csv(info_path+cname)
            #modf.to_csv(info_path+cname)
            print('cool i saved it!')
        else:
            print('its not there...')
            modf.to_csv(info_path+cname)#,index_label='index')
            print('..................now it is motha fucka!')
            


'''splits the date to a train/testset,validateset'''
def df_splitt(split):
    first = df[:split]
    second = df[split:]
    return first , second

'''get rid of anything that leaks data of the future, marked by an exlamtion mark '''
def lose_giveaways(df):
    gaways = []
    for i in df.columns:
        if '!' in i:
            gaways.append(i)
    return df.drop(gaways,axis = 1)

pd.set_option('display.max_columns',500)
pd.set_option('display.max_rows',10000)

#highlight de boo-LEANS!
def hl(df):
    def highlight(boo):
        criteria = boo == True
        return['background-color: green'if i else '' for i in criteria]
    df = df.style.apply(highlight)
    return df





# [[[[{{{{{{{{{{{ MACHINE LEARNING ZONE! }}}}}}}}}}]]]

##### .................. ////////////////////////////////////////SPLIT\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\...............

## <<<<<<<<<<<<<<<<<<< `LINEAR MODEL` >>>>>>>>>>>>>>>>>>>>>

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error

df.columns

#CREATING LIST OF TARGETS!
targs = []
for i in df.columns:
    if '!' in i:
        targs.append(i)
tardf = pd.DataFrame(targs,columns=['targets'])
tardf

# ---------------{define your TARGET!}--------------------

ltarget = targs[6]
ltarget

fe_first = fe_first.rename(columns={'time':'date'})


#SEPERATE FEATURES FROM TARGET

x = fe_first.drop('date',axis=1)
lx = x
y = df_first[ltarget]
#saving features to the thing
#info['target'] = df[target]

x_train,x_test,y_train,y_test = ttsplit(lx,y) #train_test_split simply saves some trouble, of typeing

#ouputs the model, model_id, predictions and coeficiant dataframe in a list, while saving all parmas to 'modeldic'
lm = lin_reg_fit(lx,y,notes='this is test 4')

lm[3].T

coedf = lm[3].T

coedf.sort_values('coefs',ascending=False)



# lets build a log reg model function ...

import tqdm
#help(tqdm)

tardf

tarli = [8,9,10,11,12,13,22,23,24,25,26,27]
#tarli = [26,27,22,23,24,25,34,35,11,10,9,12,8,13]
for ml_target in tarli:
    print(df[targs[ml_target]].name)
    


for ml_target in tarli:
    try:
        target  = targs[ml_target]

        if 'up' in target:
            dtarget = target.replace('up','down')
        elif 'down' in target:
            dtarget = target.replace('down','up')

        elif 'above' in target:
            dtarget = target.replace('above','below')
        elif 'below' in target:
            dtarget = target.replace('below','above')
        print('target',target, '-->',dtarget)


        #dtarget = targs[13]

        print('target:     ',target, '\nanti-target:',dtarget)

        info['target'] = df[target]

        ## <<<<<<<<<<<<<<<<< `LOGISTIC MODEL` >>>>>>>>>>>>>>>>>>>>
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report, confusion_matrix,r2_score

        ### -------------------------------({bionary} target selection)----------------------------------

        ##sns.heatmap(df.isnull())

        # ==============this is where i need to add d target to one log_reg_fit!


        x = fe_first
        x = x.drop(['date'],axis=1)
        #x = scale(x)
        y = df_first[target]
        dy = df_first[dtarget]



        x_train,x_test,y_train,y_test = ttsplit(x,y)
        dx_train,dx_test,dy_train,dy_test = ttsplit(x,dy)



        x

        def log_reg_fit(x,y,dy,name=' ',notes=''):
            #ouputs the model, model_id, predictions & coef_df in a list, while saving all parmas to 'modeldic'
            #runs a Logistic model and save the paramaters, outputs,
            #the model,model_id, prediction, and a coeficiant dataframe 

            indic = {}
            mstamp = str(datetime.now())
            '''model'''
            mtype = 'LogisticRegression'
            sn = mstamp.split('.')[1]
            modelid = mtype+sn
            #adding paramaters
            indic['model_id'] = modelid
            indic['model_name'] = target+modelid #this is striclty beinf used for nameing
            indic['model_notes'] = 'max_iter is on 4000'#notes
            indic['stamp'] = mstamp
            indic['model_type'] = mtype
            indic['features'] = str(x.columns)
            indic['target']   = target
            #setting up model
            '''object name'''
            lr = LogisticRegression(max_iter=4000)
            lr.fit(x_train,y_train)
            pred = lr.predict(x_test)
            #dont think this will work on 
            coef = lr.coef_ 

            codf = pd.DataFrame(coef,columns=[x.columns])#,x.columns)#,columns=['coefs'])
            #this would be if you wanted to rank them ....
            #for i in trange(0,len(codf)):
            #    if codf['coefs'][i] < 0:
            #        codf['coefs'][i] = codf['coefs'][i] * -1
            #codf = codf.T

            codf#.iplot(kind='bar',theme='#solar')

            #metrics
            clr = classification_report(y_test,pred)
            dclr= classification_report(dy_test,pred)
            '''
            print('~~~~~~~~~~~~~~~~~~~~~~~HITTNG THE TARGET~~~~~~~~~~~~~~~~~~~~~~~~~\n',clr)
            print('..............................vs................................')
            print('~~~~~~~~~~~~~~~~~~~~~~~ THE ANTI-TARGET~~~~~~~~~~~~~~~~~~~~~~~~~\n',dclr)
            comx = confusion_matrix(y_test,pred)
            print("~~~~~~~~~~~~~~~~~~~~~~[CONFUSION MATRIX:]~~~~~~~~~~~~~~~~~~~~~~~" )
            print("                 ......///",comx[0], "\\\........." )
            print("                 ...../// ",comx[1], " \\\........." )
            '''
            #classification report
            clr = classification_report(y_test,pred,output_dict=True)
            clrdf = pd.DataFrame(clr)
            indic['true_positive']=clr['True']['precision']
            indic['true_negitive']=clr['False']['precision']
            indic['weighted avg'] =clr['weighted avg']['precision']
            indic['f1_score']     =clr['weighted avg']['f1-score']
            codf
            modeldic[modelid] = indic
            return [lr, modelid, pred,clrdf,codf] #,codf.T#.iplot(kind='bar',theme='#solar')


        lr = log_reg_fit(x_train,y_train,dy_train)


        lr[4].T.reset_index().sort_values(0,ascending=False)



        os.listdir()

        def save_log_coefs():
            name = lr[1]
            coefs = lr[4]
            codf = coefs.T
            codf = codf.reset_index().rename(columns={'level_0':'features',0:'coef'})
            codf['true_coef'] = codf['coef']
            for i in trange(0,len(codf)):
                if codf['coef'][i] < 0:
                    codf['coef'][i] = codf['coef'][i] * -1
            p = 'PREDICTION_DATA/'
            if not os.path.exists(p):
                os.mkdir(p)
            csname = ('coefs_'+target+'_split:'+str(split)+'.csv')
            codf.to_csv(p+csname)
            return codf.sort_values('coef',ascending=False)
        save_log_coefs()



        # Saving the Coefs Sheet







        # SWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEET
        ## .............................................TODO

        ### OK THIS IS THE TEMPLATE TO APPLY TO THE REST OF THE BIONARY MODELS





        lr[3]

        ## <<<<<<<<<<<<<<<<<<<<<<< `TREE` >>>>>>>>>>>>>>>>>>>>>>>>>>

        from sklearn.tree import DecisionTreeClassifier

        tree = tree_fit(x,y,dy_train)

        tree

        tree[3]

        #from sklearn.tree import plot_tree

        #plt.figure(figsize=(20,10))
        #plot_tree(tree[0],filled=True,max_depth=5)
        #plt.show()

        lr[4].T.reset_index()

        ## <<<<<<<<<<<<<<<<<<<< `FOREST` >>>>>>>>>>>>>>>>>>>>>>>

        from sklearn.ensemble import RandomForestClassifier



        forest = forest_fit(x,y_train,dy_train)

        forest[3]

        modf = pd.DataFrame(modeldic)
        modf

        #     [[[[[[[[{{{{{{{{{{{ ~VALIDATE~ ! }}}}}}}}}}]]]]]]]]]]]

        os.listdir(path)

        ddf['Date'] = ddf['date']
        #predf = second.copy()
        df.head()

        ### LIN TARGET

        x = fe_second[lx.columns]
        y = df_second[ltarget]


        #sola(x)

        ## <<<<<<<<<<<<<<<<<<<< `LINEAR` >>>>>>>>>>>>>>>>>>>>>>>

        predf = df_second[['Close',target]] 
        lmp = lin_validate(lm,x)



        ## <<<<<<<<<<<<<<<<<<<< `LOGISTIC` >>>>>>>>>>>>>>>>>>>>>>>

        ### BIONARY TARGET

        ### TODO
        #### #add the date to result data frame
        #### #add a plot of price to scale on the real close price
        #### >add classification report with oposite values
        #### # save metrics to dict...
        ####  #true metrics vs pred plot,(just bionarys )
        #### > add title & save image params to #sola function  
        #### > add a prediction only function


        def binary_validate(mdl,x,dy):
            pred = mdl[0].predict(x)
            predf[mdl[1]] = pred
            redf = predf[[target,'Close',mdl[1]]]
            redf['time'] = ddf['Date']
            redf.set_index('time',inplace=True)
            sdf = scale(redf)
            sdf['time'] = ddf['Date']
            sdf.set_index('time',inplace=True)
            #metrics
            clr = classification_report(y,pred)
            dclr= classification_report(dy,pred)
            '''
            print('~~~~~~~~~~~~~~~~~~~~~~~HITTNG THE TARGET~~~~~~~~~~~~~~~~~~~~~~~~~\n',clr)
            print('..............................vs................................')
            print('~~~~~~~~~~~~~~~~~~~~~~~ THE ANTI-TARGET~~~~~~~~~~~~~~~~~~~~~~~~~\n',dclr)
            comx = confusion_matrix(y,pred)
            print("~~~~~~~~~~~~~~~~~~~~~~[CONFUSION MATRIX:]~~~~~~~~~~~~~~~~~~~~~~~" )
            print("                 ......///",comx[0], "\\\........." )
            print("                 ...../// ",comx[1], " \\\........." )
            '''
            #the real price plot
            scale2close= mdl[1]+'_$scale'
            redf[scale2close] = redf[mdl[1]].replace(True,1)
            redf[scale2close] = redf[scale2close].replace(1,redf['Close'])
            print(sdf[target].name)
            #sola(redf[['Close',scale2close]])
            #scaled close
            ##sola(sdf.drop(target,1))
            #scaled bionary
            ##sola(sdf[[target,mdl[1]]])
            clr = classification_report(y,pred,output_dict=True)
            clrdf = pd.DataFrame(clr)
            dclr = classification_report(dy,pred,output_dict=True)
            dclrdf = pd.DataFrame(dclr)
            modeldic[mdl[1]]['target']      = target
            modeldic[mdl[1]]['true_positive']=clr['True']['precision']
            modeldic[mdl[1]]['true_negitive']=clr['False']['precision']
            modeldic[mdl[1]]['weighted avg'] =clr['weighted avg']['precision']
            modeldic[mdl[1]]['f1_score']     =clr['weighted avg']['f1-score']
            modeldic[mdl[1]]['dtrue_positive']=dclr['True']['precision']
            modeldic[mdl[1]]['dtrue_negitive']=dclr['False']['precision']
            modeldic[mdl[1]]['dweighted avg'] =dclr['weighted avg']['precision']
            modeldic[mdl[1]]['df1_score']     =dclr['weighted avg']['f1-score']
            modeldic[mdl[1]]['true_dtrue_dif']=clr['True']['precision'] - dclr['True']['precision']
            modeldic[mdl[1]]['wt_dwt_avg_dif'] =clr['weighted avg']['precision']-dclr['weighted avg']['precision']
            modeldic[mdl[1]]['f1_score']
            #modeldic[mdl[1]] = dic
            return redf,pred



        print(len(fe_second),len(df_second))

        print(len(df_second[target]))
        print(len(fe_second))

        x = fe_second
        x = x.drop(['date'],axis=1)
        y = df_second[target]#.replace(1,True).replace(0,False)
        dy = df_second[dtarget]#.replace(1,True).replace(0,False)
        x.head()



        lrp = binary_validate(lr,x,dy)


        # Predf is a dataframe ofall the predictions?

        predf



        ## <<<<<<<<<<<<<<<<<<<< `TREE` >>>>>>>>>>>>>>>>>>>>>>>

        treep = binary_validate(tree,x,dy)

        redf = treep[0]
        redf[dtarget] = df[dtarget]








        ## <<<<<<<<<<<<<<<<<<<< `FOREST` >>>>>>>>>>>>>>>>>>>>>>>

        forestp = binary_validate(forest,x,dy) 

        forestp[0]

        pd.DataFrame(modeldic)



        modf = pd.DataFrame(modeldic)

        ## try adding more bars here

        



        # [[[[[[[[{{{{{{{{{{{ SAVE RESULTS ZONE! }}}}}}}}}}]]]]]]]]]]]

        
         ### save predictions

        predf['date'] = df['date']

        pppath = 'PREDICTION_DATA/predictions/'
        if not os.path.exists(pppath):
            os.mkdir(pppath)
        pppname = (pppath+target+'_split:'+str(split)+'.csv')
        predf.to_csv(pppname)


        fin = 'fin:'+str(datetime.now()) 

        
        
        fin = 'fin:'+str(datetime.now()) 

        import pickle

        

        def pickle_suprise(model):
            # save the model to disk
            modell = model[0]
            pickle_path = ('pickles/')
            if not os.path.exists(pickle_path):
                os.mkdir(pickle_path)
            filename = (pickle_path+model[1])
            pickle.dump(modell, open(filename, 'wb'))

            # some time later...

            # load the model from disk
            #loaded_model = pickle.load(open(filename, 'rb'))
            #result = loaded_model.score(x_test, y_test)
            #print(result)
            #print(filename)

        save_info()
        #eventually these should go into the save_info function, but this works nive ads a hard stop
        pickle_suprise(lm)
        pickle_suprise(lr)
        pickle_suprise(tree)
        pickle_suprise(forest)
    
    except IndexError:
        pass
    except KeyError:
        pass
    except ValueError:
        pass
    

# Common Errors

#except IndexError:
    #   pass
    #except KeyError:
    #    pass
    #except ValueError:
    #    pass
    

modf.T[['true_dtrue_dif','wt_dwt_avg_dif']]#.iplot(kind='bar',theme='#solar')

modeldic[lm[1]].keys()

clean_compleate = datetime.now()
notebook_runtime = clean_compleate - clean_initiate
print('notebook ran in: ',notebook_runtime)


mpdf = pd.read_csv('PREDICTION_DATA/model_preformance.csv',index_col='index')
mpdf.T[['true_dtrue_dif','wt_dwt_avg_dif']]#.iplot(theme='#solar',fill=True)
mpdf.T[['true_dtrue_dif','wt_dwt_avg_dif']]#.iplot(theme='#solar',kind='bar')
mpdf.T[['true_positive','weighted avg','true_negitive','f1_score']]#.iplot(theme='#solar',kind='bar')

# you have to add notes or it wont save csv_name
### odly enough true_negitive is more important than true negitive...makes sence

mpdf

#!>models.md
stamp


print('notebook runtime:\n'
      ,str(datetime.now() -pd.to_datetime(stamp)).split(' ')[2].split('.')[0])