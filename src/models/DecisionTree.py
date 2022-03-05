from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,roc_auc_score, f1_score, precision_score, recall_score


import pandas as pd

class DecisionTree:
    """
    A decision tree classifier.
    """

    def __init__(self,criterion:str = 'entropy', class_weight:str=None):
        # min_samples_split:int=2, min_samples_leaf:int=1, min_impurity_decrease:float=0.0, max_depth:int=None 
        """
        Initializes the decision tree classifier.
        Args:
            criterion: measure of disorder
            class_weight: weight of the classes e.g. "balanced"
        """
        self.criterion = criterion
        self.class_weight = class_weight
        self.model = DecisionTreeClassifier(criterion=self.criterion, class_weight=self.class_weight)
    
    def fit(self, X:pd.DataFrame, y:pd.Series)->None:
        """
        Fits the model to the data.
        Args:
            X: The data to fit.
            y: The labels to fit.
        """
        self.model.fit(X, y)
        self.accuracy = self.model.score(X, y)
        # accuracy will be 1, that means the model is 100% overfitted
    
    def predict(self, X:pd.DataFrame,y_test:pd.DataFrame)->pd.Series:
        """
        Predicts the labels for the given data.
        Args:

            X: The data to predict.
            y_test: The labels to predict.
        Returns:
            The predicted labels.
        """
        self.y_pred = self.model.predict(X)
        self.f1 = f1_score(y_test, self.y_pred)
        self.acc = accuracy_score(y_test, self.y_pred)
        self.recall = recall_score(y_test, self.y_pred)
        self.auc = roc_auc_score(y_test, self.y_pred)
        return self.y_pred
            

    def get_accuracy(self)->float:
        """
        Returns the accuracy of the model.
        Returns:
            The accuracy of the model.
        """
        return self.acc
    
    def get_f1(self)->float:
        """
        Returns the f1 score of the model.
        Returns:
            The f1 score of the model.
        """
        return self.f1

    def get_recall(self)->float:
        """
        Returns the recall score of the model.
        Returns:
            The recall score of the model.
        """
        return self.recall
    
    def get_auc(self)->float:
        """
        Returns the auc score of the model.
        Returns:
            The auc score of the model.
        """
        return self.auc
        