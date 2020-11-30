import shutil
import os

# IF I MAKE A VERSION THAT I WANT TO COPY DIRECTORYS I JUST THROW THE 
# try except statment in reverse!!


deldirs = [
    'backtest_data',
    'clean_data',
    'pickles',
    'PREDICTION_DATA'
]

def deleter(path):
    if os.path.exists(path):
        shutil.rmtree(path)

with open('version.txt') as file:
    version = file.readline().split('-')[1]
    new_name = str(float(version)+0.1).split('0')[0]
    new_version = ('version - '+new_name)
    print(new_version)


with open('version.txt','wt') as file:
    file.write(new_version)

copy_name = 'ML_WARPATH-V'+new_name
#i want this to see what version its on and add a + .1 to the end
#by looking at the directory its in obviously and chnage copy_name

curdir = os.listdir()

copy_path = '../'+copy_name

# cheack if the directory exists
if not os.path.exists(copy_path):
    #create directory
    os.mkdir(copy_path)
    for i in curdir:
        try:
            if i not in deldirs:
                shutil.copy2(i,(copy_path+'/'+i))
        except IsADirectoryError:
            pass
# IF I MAKE A VERSION THAT I WANT TO COPY DIRECTORYS I JUST THROW THE 
# try except statment in reverse!!

'''COPYING DIRECTORYS'''


#she works
source = 'templates/'
destination = copy_path+'/'+source
shutil.copytree(source,destination)



# this is the old delete shit thing
'''
for path in curdir:
    if path not in deldirs:




for path in deldirs:
    deleter(path)
'''
