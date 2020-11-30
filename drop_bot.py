import shutil
import os

'''
this is a script to copy bots when they are ready


'''
dirs = [
    'all_stars/',
    'downloaded_data/'
]

files = [
    'candle_sticks.py',
    'daily_close_compare.py',
    'download_create_features.py',
    'drop_bot.py',
    'mover.py',
    'scrape.py',
    'version.txt',
    'stepfive.py'
]
with open('version.txt') as file:
    version = file.readline().split('-')[1]
    new_name = str(float(version)+0.1).split('0')[0]
    new_version = ('version - '+new_name)
    print(new_version)

copy_name = 'BOT_WARPATH_V'+new_name


copy_path = '../'+copy_name +'/'

#create path
if not os.path.exists(copy_path):
    os.mkdir(copy_path)


#COPY DIRECORYS
for d in dirs:
    src = d
    dest= copy_path+d
    shutil.copytree(src,dest)


#COPY FILES:
for f in files:
    src = f
    dest= copy_path+f
    shutil.copy2(src,dest)

