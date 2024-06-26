# HOW TO USE THE CODE

Our code is split into 2 projects. The ARTS++ folder contains the code for generating the ARTS as well as the ARTS++ sets. Within the ARTS++/data/2014 folder you find the 
(target-opinion word extracted) input data for the 2014 laptop and restaurant datasets, as well as the generated ARTS and ARTS++ sets for both datasets which can be found in the merged.json in the output folders. 
Furthermore, the ontology part of HAABSA++ is contained within the ARTS++ folder.

# ARTS++ project

- Main_og_ARTS.py/Main.py are used to run ARTS and ARTS++ respectively. The corresponding strategies and utils used are found in strategies_ARTS.py/strategies.py and utils_ARTS.py/utils.py,
respectively. 
- extractCategories.py is used to extract dictionaries linking aspect terms to their corresponding categories and vice verse using a labeled dataset such as SemEval 2016 Restaurant. 
- Main_Ont.py runs the ontology reasoner
- Sentiment_prediction.py is used in the ARTS++ generation process to transform sentences into the format expected by the neural network of HAABSA++
- BERTembeddings.py is used to create BERT embeddings for training and testing purposes in HAABSA++ neural network
- The remaining files contain some additional data manipulation/analyzing tasks

# HAABSA++ project

- Tuning.py is used to tune the hyperparameters of the model
- Train.py is used to train the model
- Test.py is used to test the model on a variety of dataset

# Requirements

The two separate projects/folders require different versions of python, as well as different package versions. The ARTS++ project requires python 3.9, whereas HAABSA++ requires python 3.11. Some packages must be installed via conda, otherwise they will not work. Thus, it is recommended to create two separate virtual environments, one for ARTS++ with python 3.9 and one for HAABSA++ with python 3.11. The specific package requirements are listed in the requirements.txt files in ARTS++ and HAABSA++ folders separately. 

# Acknowledgements

In this project we use code from the original aspect-robustness test set https://github.com/zhijing-jin/ARTS_TestSet as well as a keras implementation for HAABSA++ from https://github.com/MiladAgha/rHAABSA-pp.
  
