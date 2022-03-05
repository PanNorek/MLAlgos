from src.data_preparation.DataPreparator import DataPreparator
from src.models.DecisionTree import DecisionTree


tmp = DataPreparator('data/heart_failure_clinical_records_dataset.csv')

tmp.one_hot_encode()
tmp.train_test_split('DEATH_EVENT')
tmp.scale()

X_train, X_test, y_train, y_test = tmp.X_train, tmp.X_test, tmp.y_train, tmp.y_test

model = DecisionTree()
model.fit(X_train, y_train)
print(model.accuracy)      
model.predict(X_test, y_test)
print(model.acc)      
print(model.f1)
print(model.recall)
print(model.auc)


