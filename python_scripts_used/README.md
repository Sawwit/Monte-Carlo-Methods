# Description of files
This folder, as mentioned, contains all python scripts ultimately used within the project. It also contains the output of the fine tuning procedures. Since it was very useful to have them within the same folder as the data analysis  Below a brief description of the files will follow

### Cluster.py

This file is the pythons script that ultimately made the implementations run. Please note the comments within this file. 

### HMC_finetuning.py

The file contains the  fine tuning procedure for the HMC method

### HMC_scalar.py

This file contains the implemenation of the HMC algorithm

### Heatbath.py 

This file contains the implemenation of MH from the lectures. NOTE: The naming has gone wrong due to a error within our understanding which method was which, as explained within the readme of the whole folder

### data_analysis.py

This file contains the python script with the code used for the data analysis. 

### Analysis_Notebook.ipynb

The notebook contains the code that is also within data_analysis.py but then within an interactive notebook

### heatbath_finetuner.py

This file contains the finetuning procedure of the MH method. Again note the naming mistake

### real_heatbath.py

This file contains the implemenation of the heatbath algorithm of the assignments. 

### The JSON files

The JSON files within this subfolder are the output of the fine tuning procedures, they are present within this folder since the implementations (within cluster.py) use them, and originally assumed that these JSON files were within the same folder
