# HOW TO USE THE CODE

Our code is split into 2 projects. The ARTS++ folder contains the code for generating the ARTS as well as the ARTS++ sets. Within the ARTS++/data/2014 folder you find the 
(target-opinion word extracted) input data for the 2014 laptop and restaurant datasets, as well as the generated ARTS and ARTS++ sets for both datasets. Furthermore, the ontology part of
HAABSA++ is contained within the ARTS++ folder.

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

  
