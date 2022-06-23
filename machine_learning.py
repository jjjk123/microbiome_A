import pandas
import numpy
import sklearn.linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, average_precision_score, precision_score, \
    recall_score, make_scorer, confusion_matrix
from sklearn.model_selection import train_test_split


class MachineLearning():

    def __init__(self):
        pass

    def prepare_data(self, df_merged: pandas.DataFrame):
        # Total training data
        self.X = (df_merged.loc[:, df_merged.columns != 'CLASS'])

        # Target/labels array
        self.y = (df_merged.loc[:, df_merged.columns == 'CLASS'])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, stratify=self.y, test_size=0.3)

    @staticmethod
    def get_metrics(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        aps = average_precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        prs = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        return "Accuracy: %.3f, Precision: %.3f, Recall: %.3f, APS: %.3f, F1: %.3f, MCC: %.3f, CM: %s" % (acc, prs, recall, aps, f1, mcc, cm)


class FLLogisticRegression(MachineLearning):

    def __init__(self):
        self.intercepts = []
        self.coefs = []
        self.glob_intercept = 0
        self.glob_coef = []
        self.ml = None

    def fit(self):
        lr = LogisticRegression()
        lr.fit(self.X_train, self.y_train)
        self.ml = lr
        self.coefs.append(lr.coef_)
        self.intercepts.append(lr.intercept_)
        self.glob_intercept = self.ml.intercept_
        self.glob_coef = self.ml.coef_

    def benchmark(self, X=None, y=None, check_them_all=False) -> str:
        if X is None and y is None:
            #print(self.X_test)
            #print(self.y_test)
            #print("### Params")
            #print(self.ml.intercept_)
            #print(self.ml.coef_)
            if check_them_all:
                y_pred = self.ml.predict(self.X)
                bench = MachineLearning.get_metrics(self.y, y_pred)
            else:
                y_test_pred = self.ml.predict(self.X_test)
                bench = MachineLearning.get_metrics(self.y_test, y_test_pred)
        else:
            y_test_pred = self.ml.predict(self.X)
            bench = MachineLearning.get_metrics(self.y, y_test_pred)
        return bench

    def get_params(self):
        return self.glob_intercept, self.glob_coef

    def set_params(self, intercept, coef, collection=True):
        if collection:
            self.intercepts.append(intercept)
            self.coefs.append(coef)
        else:
            self.glob_intercept = intercept
            self.glob_coef = [coef]

    def aggragate_params(self):
        self.glob_intercept = [sum(self.intercepts) / len(self.intercepts)]
        self.glob_coef = []

        t = self.coefs

        for outer in range(0, len(t[0])):
            s = 0
            for inner in range(0, len(t)):
                s += t[inner][outer]
            self.glob_coef.append(s / len(t))

        return self.glob_intercept, self.glob_coef

    def build_new_model(self):
        new_lr = LogisticRegression()
        new_lr.coef_ = numpy.array(self.glob_coef)
        new_lr.intercept_ = self.glob_intercept
        new_lr.classes_ = numpy.array([0, 1])
        #print("I've a new model!!!")
        #print(new_lr.intercept_)
        #print(new_lr.coef_)
        self.ml = new_lr
