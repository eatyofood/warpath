# welcome to the instructions

### WORKFLOW
#### the workflow consists of 5 steps
> STEP 0. - get the data, and TODO:make sure its in the right format
> STEP 1  - create A.features B.targets and then C.clean/parse/save
> STEP 2  - run the ml_models
> STEP 3  - run predictions through backtests
> STEP 4  - [UNDER_CONSTRUNTION] evaluate the modesl
> STEP 5  - take the best ones out for tea (under conseption)

### BASIC EXAMPLE
#### lets say you already have some data in a directory outside warpaths current directory...

> STEP 0 = data_grab.py       
- puts the data in 'data/' directory
> STEP 1 = stepone_candels.py 
- adds features targest and drops nulls
> STEP 2 = NOTEBOOK[STEP 2 -ITERATE_ALL.ML-MODELS-Multi-Model ]- v4.0
-trains and tests dozens of ml models with dozens of targets and saves the results
> STEP 3 = NOTEBOOK[STEP 3 - Back Testing Predictions Evaluation -v5]
runs hundreds of backtest's and saves the best ones
> STEP 4 = todo: 

> LAST STEP = copy_dir.py
adds a .1 to version and copys the files and 'template/' dir





# FRESH START
### delete all data (except data in 'data')

clear_data.py
> with this you start over from STEP-I


clear_models.py
>clears all model data so you can start over from STEP - II


copy_dir.py
>copys the directory and updates version


# STEP ONE: PREPROSING, TARGETS, FEATURES

## scripts:
>stepone_candels.py
>stepone_tech.py
>stepone_tech_n_cans.py

## NOTEBOOKS
> template_folder ... pick one

TODO:DATA comes in standard format this should be pre_step one actually 

# STEP THREE: backtest & evaluation
>one option
> soon to be split...into just backtest

### saving models

mover.move_model(model_name)
>moves and saves the model in its own directory

# STEP V
this notebook turns your bot into a script to be called form another notebook

### stepfive.py
>pulls most recent predictions on married models into note books

import stepfive
>gets most recent data and plots predictions

from stepfive import predf,hl
>hl(predf[::-1]) - works exactly the way you would want it to


there are differnt version here for differnt things

# for each project the first notebook [feature creation and data_cleaning]
will be a little differnt each time


#run time
first run:
step1 : 2mins to get to step II
stepII: 3MINS
stepIII: 

