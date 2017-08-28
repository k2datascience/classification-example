## Imports

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

from sklearn.naive_bayes import BernoulliNB
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, f1_score

## ML Pipeline Functions

def clean_data(data_file, pickle_file):
    print("Starting to clean bank data")
    print("---------------------------")
    bank_data = pd.read_csv(data_file, sep=';', na_values=['NA'])

    # Make target binary
    bank_data = bank_data.replace(['yes', 'no'], [1, 0])

    # Imputation
    bank_data.replace(['unknown'], [np.nan], inplace=True)
    bank_data['pdays'].replace([999], [np.nan], inplace=True)
    bank_data = bank_data.fillna(bank_data.median())

    # Get dummy variables for categorical features
    columns = ['job', 'marital', 'education', 'contact', 'month', 'day_of_week', 'poutcome']
    bank_data = pd.get_dummies(bank_data, columns=columns)

    print("Saving processed data to pickle file")
    print("---------------------------")
    pickle.dump(bank_data, open(pickle_file, 'wb'))

def bernoulli_model(pickle_file):
    bank_data = pickle.load(open(pickle_file, 'rb'))
    del bank_data['duration']

    # Stratified train test split
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(bank_data, bank_data["y"]):
        strat_train_set = bank_data.loc[train_index]
        strat_test_set = bank_data.loc[test_index]

    bank = strat_train_set.drop("y", axis=1)
    bank_labels = strat_train_set["y"].copy()

    # Set numerical and categorial
    num_attribs = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx',
           'cons.conf.idx', 'euribor3m', 'nr.employed']

    cat_attribs = list(set(num_attribs).symmetric_difference(bank.columns))

    # Pipeline
    num_pipeline = Pipeline([
            ('selector', DataFrameSelector(num_attribs)),
            ('std_scaler', StandardScaler()),
        ])

    cat_pipeline = Pipeline([
            ('selector', DataFrameSelector(cat_attribs)),
        ])

    full_pipeline = FeatureUnion(transformer_list=[
            ("num_pipeline", num_pipeline),
            ("cat_pipeline", cat_pipeline),
        ])

    bank_prepared = full_pipeline.fit_transform(bank)
    bank_bernoulli = bank_prepared[:, 9:]

    # Grid search with a stratified KFold
    parameter_grid = {
        'alpha': [0.001, 0.01, 0.1, 0.5, 1, 2, 5]
    }

    cross_validation = StratifiedKFold(n_splits=10)

    gs = GridSearchCV(BernoulliNB(),
                      param_grid=parameter_grid,
                      cv=cross_validation)

    gs.fit(bank_bernoulli, bank_labels)
    print('Best score: {}'.format(gs.best_score_))
    print('Best parameters: {}'.format(gs.best_params_))

    X_test = strat_test_set.drop("y", axis=1)
    y_test = strat_test_set["y"].copy()

    X_test_prepared = full_pipeline.transform(X_test)

    # Play around with decision thresholds
    y_labels = BernoulliNB().fit(bank_bernoulli, bank_labels).predict(X_test_prepared[:, 9:])
    y_probas = BernoulliNB().fit(bank_bernoulli, bank_labels).predict_proba(X_test_prepared[:, 9:])[:, 1]

    # Higher recall
    new_labels = [update_labels(y, threshold=0.05) for y in list(y_probas)]

    # Pretty confusion matrix
    confmat = confusion_matrix(y_true=y_test, y_pred=new_labels)
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.savefig('./reports/figures/confusion.png', dpi=300)
    print("Saved confusion matrix")
    print("---------------------------")

    # Serialize finalized model
    full_y = bank_data['y'].copy()
    full_bank = full_pipeline.fit_transform(bank_data.drop("y", axis=1))
    full_bank_bernoulli = full_bank[:, 9:]
    clf = gs.best_estimator_
    clf.fit(full_bank_bernoulli, full_y)
    filename = './models/finalized_model.sav'
    pickle.dump(clf, open(filename, 'wb'))
    print("Serialized final Bernoulli Naive Bayes Model")
    print("---------------------------")


## Helper Classes Functions

# Create a class to select numerical or categorical columns
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

# New labels
def update_labels(y, threshold=0.5):
    if y > threshold:
        label = 1
    else:
        label = 0

    return label


## Variables
data_file = './data/raw/bank-additional-full.csv'
pickle_file = './data/processed/bank_data.pkl'

## Main Block
if __name__ == "__main__":
    clean_data(data_file, pickle_file)
    bernoulli_model(pickle_file)
