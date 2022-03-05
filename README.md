# About data

https://www.kaggle.com/andrewmvd/heart-failure-clinical-data



# Setup


<li>Create a virtual environment using <code> virtualenv venv </code>
<li>Activate the virtual environment by running <code> venv/bin/activate </code>
<li>On Windows use <code> venv\Scripts\activate.bat </code>
<li>Install the dependencies using <code> pip install -r requirements.txt </code>
<li>Check possibilities <code> python main.py -h </code> or run default <code> python main.py </code>

# Limitations 

1. Dataset must be preprocessed and cleaned (no missing values, no outliers, no duplicates). 
2. Dataset must be in csv format.
3. Models solve the classification problem.
4. Available models: Decision Tree, Random Forest K-Nearest Neighbors


# Future work
1. Add models: Logistic Regression, Support Vector Machine, Naive Bayes, K-Means, Linear SVM, and Gradient Boosting Classifier.
2. Add model manipulation: Grid Search, Cross Validation and Hyperparameter Tuning.
3. Add visualization.