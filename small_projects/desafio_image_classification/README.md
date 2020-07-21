# Image classification challenge

In order to replicate the experiments, follow the steps below:

1) Install the anaconda environment: `conda env create -f environment.yml`;

2) Manuall add the 12 .npy files containing the samples in the folder `/data/`;
  
3) Run the notebook `notebooks/2_separando_amostras.ipynb` to generate the datasets;

4) Run the script `code/experiment_1_random_forest.py` to generate the classification approach using the **Random Forest** classifier;

5) Run the script `experiment_2_DL_models.py` to generate the classification approach using some **Deep Learning** models;
