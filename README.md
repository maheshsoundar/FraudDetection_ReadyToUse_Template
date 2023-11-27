# FraudDetection_ReadyToUse_Template

This repo is a ready to use template for simple Unsupervised fraud detection. It trains isolation forest as well as Autoencoder on data 
and provides accessto both models. User can then choose to use both models however they want. 

1.Set up virtual env from root directory using python -m venv .venv. Then activate venv by running the activate.bat file in .venv/Scripts.
2.Run pip install -r requirements.txt and make sure all dependencies are installed in .venv/Scripts folder.
3.Make sure the data from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud is downloaded and is available in the data folder before running the scripts. But any other dataset can be added to that folder. Or the model training will fail. 
4. Make sure to update the target column name in ain.py before running the script. 
5.Run main.py using "python main.py" from a terminal (root directory of repo)

Training step chooses the best contamination via Gridsearch and provides the best isolation forest model. Although separate parameter grid can be provided as input to isolation forest if needed. Similarly for autoencoder best model and threshold with max auroc is already chosen. Template can be used to download the models after sufficient training.