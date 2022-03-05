from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,roc_auc_score, f1_score, precision_score, recall_score,confusion_matrix
from sklearn.model_selection import cross_validate
import pandas as pd
import sys,os


class RandomForest:
    """
    Random Forest classifier.
    """
    
    def __init__(self, n_estimators:int=10, max_depth:int=None, **kwargs):
        """
        Initializes the random forest classifier.
        Args:
            n_estimators: number of trees
            max_depth: maximum depth of the tree
            min_samples_split: minimum number of samples required to split an internal node
            min_samples_leaf: minimum number of samples required to be at a leaf node
            max_features: maximum number of features to consider for splitting at each node
            bootstrap: bootstrap samples when building trees
            overbose: verbose output
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.__dict__.update(kwargs)
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth)
    
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
        self.precision = precision_score(y_test, self.y_pred)
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
    
    def get_precision(self)->float:
        """
        Returns the precision score of the model.
        Returns:
            The precision score of the model.
        """
        return self.precision
    
    def get_confusion_matrix(self,y_test:pd.Series)->pd.DataFrame:
        """
        Returns the confusion matrix of the model.
        Returns:

            The confusion matrix of the model.
        """
        return confusion_matrix(y_test, self.y_pred)
    
    def get_all_metrics(self)->pd.DataFrame:
        """
        Returns all the metrics of the model.
        Returns:
            
                The all the metrics of the model
        """
        return pd.DataFrame({'accuracy':[self.accuracy],'f1':[self.f1],'recall':[self.recall],'precision':[self.precision]})
    
    def get_feature_importances(self)->pd.DataFrame:
        """
        Returns the feature importance of the model.
        Returns:
            The feature importance of the model.
        """
        return pd.DataFrame(index=self.data.columns[:-1], data = self.model.feature_importances_,
        columns = ["Feature Importance"] ).sort_values("Feature Importance",ascending=False)

    def cross_validate(self, X:pd.DataFrame, y:pd.Series, n_splits:int=10)->pd.DataFrame:
        """
        Returns the cross validation scores of the model.
        Args:
            X: The data to cross validate.
            y: The labels to cross validate.
            n_splits: The number of splits to use.
        Returns:
            The cross validation scores of the model.
        """
        
        
        scores = cross_validate(self.model, X, y, cv=n_splits, scoring = ["accuracy", "precision", "recall", "f1"])
        return pd.DataFrame(scores, index = range(1, n_splits+1))

    def cross_validate_mean(self, X:pd.DataFrame, y:pd.Series, n_splits:int=10)->pd.DataFrame:
        """
        Returns the cross validation mean of the model.
        """
        return pd.DataFrame(self.cross_validate( X, y, n_splits).mean()[2:]).transpose()


    


if __name__ == '__main__':
    
    args = sys.argv[1:]

    data_prep_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    + '/data_preparation/')
    sys.path.append(data_prep_dir)
    from DataPreparator import DataPreparator



    tmp = DataPreparator(args[0][1:])

    tmp.one_hot_encode()
    tmp.train_test_split('DEATH_EVENT')
    tmp.scale()

    X_train, X_test, y_train, y_test = tmp.X_train, tmp.X_test, tmp.y_train, tmp.y_test

    model = RandomForest(data=tmp.data)
    model.fit(X_train, y_train)
    model.predict(X_test, y_test)
    print("Accuracy on train dataset: ", model.accuracy)    
    print("Accuracy on test dataset: ", model.acc)      
    print("F1 score: ", model.f1)
    print("Recall score: ", model.recall)
    print("Precision score: ", model.precision)
    print("Confusion matrix: \n", model.get_confusion_matrix(y_test))
    print(model.get_feature_importances())
    print(model.cross_validate_mean(X_train, y_train))

