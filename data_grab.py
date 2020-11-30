



import pandas as pd
import os 

path  = '../data/'
dirli = os.listdir(path)
pd.DataFrame(dirli)

# INDEX
i = 0
sheet = dirli[i]
df = pd.read_csv(path+sheet)
df

# CONDITIONAL INDEX

if 'time' in df.columns.str.lower():
    print('its a unicode index ')
    df = df.set_index('time')
    df.index = pd.to_datetime(df.index,unit='s')
    df.index = df.index - pd.Timedelta(hours=4)
elif 'datetime' in df.columns.str.lower():
    print('index is datetime')
    df = df.set_index('datetime')
elif 'date' in df.columns.str.lower():
    print('index is date')
    df = df.set_index('date')
else:
    print('dont know where the index is...')

#STANDARDISING INDEX - were not ready for that yet...its a big project
#df.index.name = 'datetime'

df.head()
destpath = 'data/'
if not os.path.exists(destpath):
    os.mkdir(destpath)
    print('directory created!')

df.to_csv(destpath+sheet)
print(df)
print('all done')
'''

import shutil

#this grabs data from a specifyed directory: here we grab from one folder back called fdata
dpath = '../data'
dest  = 'data/'
shutil.copytree(dpath,dest)

#this case does not require it but there will also be preprocessing being done on theis script

'''