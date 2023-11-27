import os
from sklearn.model_selection import train_test_split

from src.utils import *
from src.autoencoder import *
from src.iso_forest import *

TARGET_COL = 'Class' #Set the name of the target column according to the file uploaded. 

if __name__== '__main__':
    #Load data
    data_dir = os.path.join(os.getcwd(),'data') #The data should be present inside folder 'data' as csv
    data = DataUtil.load_data(data_dir)

    #Separate data and target
    data, target = DataUtil.data_target_split(data,TARGET_COL)

    #Train test split
    x_train,x_test,y_train,y_test = train_test_split(data,target,stratify=target,test_size=0.2,random_state=5)

    #Normalize data
    x_train,normalizer = DataUtil.normalize_data(x_train)

    #Train isolation forest and autoencoder.
    #Training step chooses the best contamination via Gridsearch and provides the best isolation forest model
    #Similarly for autoencoder best model and threshold with max auroc is already chosen.
    #Although separate parameter grid can be provided as input to isolation forest if needed. 
    iso_forest =  IsolationForestModel()
    iso_forest.train(x_train,y_train)

    auto_enc = AutoEncoderModel(x_train.shape[1])
    history = auto_enc.run(x_train,epochs=30)

    #Normalize test data before prediction 
    x_test,_ = DataUtil.normalize_data(x_test,normalizer)

    pred_iso_forest = iso_forest.predict(x_test)
    auc_score_iso = roc_auc_score(y_test,pred_iso_forest)
    print("AUROC for isolation forest {}".format(auc_score_iso))

    pred_auto_enc = auto_enc.predict(x_test)
    auc_score_enc = roc_auc_score(y_test,pred_auto_enc)
    print("AUROC for auto encoder {}".format(auc_score_enc))
