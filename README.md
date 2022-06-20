# What is this repository?

This repository contains an example baseline script for the ML competition. Besides the scripts, it is included a model trained on the first 70% samples of the training set, validated on the next 20% samples, and evaluated on the last 10%, as well as the beam prediction results obtained with the model.

In particular, a description for each file follows:

- ```position_only_baseline.py``` - the main script for reading position data using a csv, and for training and evaluating a simple ML model. 
- ```position_only_baseline_func.py``` - contains auxiliary functions, like computation utilities, the neural network and training and testing procedures.
- ```trained_model_example.pkl``` - the trained model object saved using Pickle. 
- ```beam_prediction_results_example.csv``` - the predicted beam results from applying the aforementioned model to the last 10% of data in the training set.

# About the code

Requires: PyTorch, CUDA, utm and tqdm.

*Besides changing the csv_path variable, the code should work as is.*

During training, some changes should occur in the folder with the main script: a) a checkpoint folder is created to save the machine learning models in each epoch (and returning the one with the highest validation accuracy at the end); b) a pickled model should be created.

During testing, a csv file with the results should be generated, and performance metrics are printed, including the *competition score*. Furthermore, there is a small histogram plot included. 

See the main script for further details and examples on how to use the scripts.

# Expected Results

When testing the example model on last 10% of the data in the training dataset, these should be the obtained results:

|            | Accuracy [%] |       |       |
|:----------:|:------------:|:-----:|:-----:|
|            |     Top-1    | Top-3 | Top-5 |
| Validation |     31.87    | 63.64 | 77.24 |
|    Test    |     28.61    | 57.49 | 70.22 |			

Competition score (in 10% of training set): 0.674260

# About the Neural Network

The model used to predict beams from positions is presented in: (arxiv link)

The only difference between the model presented here and the model in the paper is that the former uses a single label for model training, while the latter backpropagates with all labels available in the dataset. In this competition we provide a partial dataset that is already divided into sequences, so the first approach is not available.

