import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score,make_scorer
from sklearn.model_selection import GridSearchCV

class IsolationForestModel:
  def __init__(self,param_grid:dict=None):
    self.iso_model = IsolationForest(random_state=5)
    if(param_grid is None):
      self.param_grid = self.__get_param_grid()
    else:
      self.param_grid = param_grid

  def  __get_param_grid(self):
      return {'contamination': [0.02, 0.04, 0.06, 0.08, 0.1]}

  def __grid_search(self,x_train,y_train):
    grid = GridSearchCV(self.iso_model,self.param_grid,
                                                 scoring=make_scorer(roc_auc_score, average='micro'),
                                                 refit=True,
                                                 cv=5,
                                                 return_train_score=True)
    grid.fit(x_train,y_train)
    return grid.best_estimator_

  def train(self,x_train,y_train):
    self.iso_model = self.__grid_search(x_train,y_train)

  def predict(self,test):
    pred = self.iso_model.predict(test)
    return np.where(pred == 1, 0, 1)

  def get_model(self):
    return self.iso_model