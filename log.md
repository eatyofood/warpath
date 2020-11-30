# TODO:
steptwo_scaled
steptwo_features
COPY_ALL!!!!
THAT WAY YOU CAN ITERATE THROUGH ALL OF THEM !!!!!!


# STEP II - reworking
added scale_steptwo.py

> seperating features scaled and targets when selecting sheets

- now apply to script
- make feature script
- make scaled script


# stepthree.py
# steptwo.py

# stepone_nofeatures.py
>just for trading view data or data that camein with features

## ADDED STEP 5 

### stepfive.py
>pulls most recent predictions on married models into note books

import stepfive
>gets most recent data and plots predictions

from stepfive import predf,hl
>hl(predf[::-1]) - works exactly the way you would want it to


### ADDED STEP 6 - pick up sticks
>plots models
### scrape.py

# TODO:
>stepone_tech_n_cans
candles are turned off

# ADDED
import scrape
scrape.load(data_type,ticker)
>downloads data from urllib


### mover.py

import mover

>mover.move_model(model_name)
moves and saves the model in its own directory

>mover.load_model(model_name)
recalls everything that ever had to do with that model


# 11.2.2020

#stepone datetime error

>fixed it by blocking out )#,unit='s')

### TODO
> make a class that standardizes trading view data coming in




#adding template folder
for anything you want to get carried over to next version...

#clear_data.py clear_models.py copy_dir.py
all ways of maniging work flow really nicely 

# 10.30.2020
the long models are not coming through for some reason
>   NOPEi think its because they are not saving
>   NOPEcould be becuase targets are wrong
>    IT WAS B/C I RAN THE SAME SPLIT TWICE IN THE SAME DIR...

## todo
>   CHECKchange the split settings- rerun it
>   CHECKfix the data cleaner
> 


#10.16.2020

## STEP - 2 
### Save the Date to Prediction csv
line336 = predf['date'] = df['date']

### Eliminate 'index' columns from features
>for some reason when i try to remove it at the begining it doesnt create model dic 
> MUST BE - b/c its erroring somwhere 
>AND I HAVE ALL THE ERRORS TURNED OFF...
*YUP - having errors off is an issue...


###REMEMBER
###STEP - 2 - logs

THE COPY OF ITER IS TRASH BULLSHIT!!!

#save into
###funcions
def save_info(csvname='model_preformance'):
    cname = (csvname+'.csv')
    info_path = 'PREDICTION_DATA/'
    if not os.path.exists(info_path):
        os.mkdir(info_path.replace('/',''))
    yn = 'y'#input('do you want to save these results y/n?')
    if yn == 'y':
        yyn = 'loopy'#input('is there anything you want to add?')
        if yyn == 'n':

### iter loop
i add this to the bottom so that it skips the ones that dont work 

it did indeed go all the way to the end so i think this shold be the new version but also might be import to leave these out in some cases wh
when you want to see what is not working

except IndexError:
        pass
    except KeyError:
        pass
    except ValueError:
        pass


### well this is an attempt at logging my changes to the system

### loading the dataset
i am loading the tradingview data... 

#### replaced
df = get_price(ticker)
> the directory where is saves...?

>#df = df.drop('label',axis=1)

df = pd.read_csv('fundamentals_data/BATS HYLN, 15.csv')
if 'time' in df.columns:
    df['time'] = pd.to_datetime(df['time'],unit='s')
    df = df.rename(columns={'time':'date',
                            'open':'Open',
                            'high':'High',
                            'low':'Low',
                            'close':'Close'})
                            
                            
### lets lose non-usefull
 
#df = df.drop('label',axis=1)
#df = df[::-1]


###i can make my own vwapiz..sweet
#df['vwap_riz'] = pta.momentum.rsi(df['vwap'])
#df['vwap_riz_ema']= pta.ema(df['vwap_riz'])
#df['vwap_ab_avg'] = df['vwap_riz']>df['vwap_riz_ema']
#df['vwap_bl_avg'] = df['vwap_riz']<df['vwap_riz_ema']



