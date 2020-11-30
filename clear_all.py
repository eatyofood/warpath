import shutil
import os


deldirs = [
    'backtest_data',
    'clean_data',
    'pickles',
    'PREDICTION_DATA',
    'all_stars',
    'data'
]

def deleter(path):
    if os.path.exists(path):
        shutil.rmtree(path)


for path in deldirs:
    deleter(path)

    