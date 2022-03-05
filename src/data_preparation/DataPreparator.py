from operator import methodcaller
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler,RobustScaler

class DataPreparator:
    """
    Prepares the data for the model.
    """
    

    def __init__(self, data_path:str):
        self._data_path = data_path
        self._data = pd.read_csv(self.data_path)
    
    @property
    def data_path(self):
        return self._data_path
    
    @data_path.setter
    def data_path(self, path):
        self._data_path = path
    
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, data):
        self._data = data
    
    



    def one_hot_encode(self,specify_columns:bool = False, columns:list[str]=None)->None:
        """
        One-hot encodes the given column.
        Args:
            columns: The columns to one-hot encode.
        
    
        """
        if specify_columns:
            self.data = pd.get_dummies(self.data, columns=columns, drop_first=True)
        else:    
            self._data = pd.get_dummies(self._data, drop_first=True)


    def train_test_split(self, target:str, test_size:float=0.15)->None:
        """
        Splits the data into train and test sets.
        Args:
            test_size: The size of the test set.
            target: The column to use as the target.

        """
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data.drop(target, axis=1), self.data[target],test_size=test_size,random_state=42)

    def scale(self, scaler:str='standard')->None:
        """
        Scales the data.
        Args:
            scaler: The scaler to use. [standard, minmax, robust]
        """
        if scaler == 'standard':
            self.scaler = StandardScaler()
        elif scaler == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError('Scaler not recognized.')
        
        if hasattr(self, 'X_train') or hasattr(self, 'X_test'):
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
        else:
            raise ValueError('No data to scale.')

        
        


