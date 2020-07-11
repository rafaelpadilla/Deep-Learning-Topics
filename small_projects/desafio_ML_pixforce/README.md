# Pixforce challenge

In order to replicate the experiments, follow the steps below:

1) Install the anaconda environment: `conda env create -f pixforce.yml`;

2) Manuall add the following files containing the samples in the folder `desafio_ML_pixforce/data/`:  
   _1_win50_trainX.npy_  
   _1_win50_trainY.npy_  
   _2_win50_trainY.npy_  
   _2_win50_trainX.npy_  
   _4_win50_trainX.npy_  
   _4_win50_trainY.npy_  
   _3_win50_trainX.npy_  
   _3_win50_trainY.npy_  
   _5_win50_trainX.npy_  
   _5_win50_trainY.npy_  
   _6_win50_trainX.npy_  
   _6_win50_trainY.npy_  
   
2) Run the notebook `notebooks/2_separando_amostras.ipynb` to generate the datasets;

3) Run the script `code/experiment_1_random_forest.py` to generate the classification approach using the **Random Forest** classifier;

4) Run the script `experiment_2_DL_models.py` to generate the classification approach using some **Deep Learning** models;
