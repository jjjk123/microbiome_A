import pandas
import numpy as np
from imblearn.metrics import specificity_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, average_precision_score, precision_score, \
    recall_score, make_scorer, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split, ParameterGrid
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, KFold


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
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        return "Acc: %.3f, Sens: %.3f, Spec: %.3f, Precision: %.3f, Recall: %.3f, APS: %.3f, F1: %.3f, MCC: %.3f, TN: %d, FP: %d, FN: %d, TP: %d" \
               % (acc, sensitivity, specificity, prs, recall, aps, f1, mcc, tn, fp, fn, tp)


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

    def build_new_model(self, warm_start=True):
        new_lr = LogisticRegression()
        new_lr.coef_ = np.array(self.glob_coef)
        new_lr.intercept_ = self.glob_intercept
        new_lr.classes_ = np.array([0, 1])
        #print("I've a new model!!!")
        #print(new_lr.intercept_)
        #print(new_lr.coef_)
        self.ml = new_lr

    def refit_model(self):
        self.ml.coef_ = np.array(self.glob_coef)
        self.ml.intercept_ = self.glob_intercept
        self.ml.classes_ = np.array([0, 1])
        self.ml.fit(self.X, self.y)

    def get_model(self):
        return self.ml

    def fit_cv(self):
        TrainTest_SplitRatio = 0.1
        random_seed = 123
        CV_folds = 2
        MC_runs = 1

        X = self.X
        y = self.y

        MonteCarloRuns_dict = {
            'F1': [],
            'Precision': [],
            'Recall': [],
            'Specificity': [],
            'Accuracy': [],
            'MCC': [],
            'Conf_matrix': [],

            'TEST_Conf_matrix': [],

            'RandomForestOptimum_params': [],
            'RF_opt_model': [],
            'RF_opt_PredictedProbabilities': [],

            'RF_fitted_opt': [],

            'FPR_MC': [],
            'TPR_MC': [],
            'Thresholds_MC': [],

            'GeneratedRandomNumbersForReproducibility': [],
            'MC_test_accuracy': [],
            'MC_test_specificity': [],
            'MC_test_recall': [],
            'MC_test_precision': [],
            'MC_test_f1': [],
            'MC_test_ConfMat': [],

            'Overall_Test_DataLabels': [],
            'Overall_Training_DataLabels': [],

            'Overall_Test_Data': []
        }

        for MonteCarlo_i in range(0, MC_runs):
            cross_val_MCC_lst = []
            cross_val_f1_score_lst = []
            cross_val_precision_lst = []
            cross_val_recall_lst = []
            cross_val_specificity_lst = []
            cross_val_accuracy_lst = []
            cross_val_ConfMatrix_lst = []

            random_seed = 1234  # random.randint(1,100000000)
            MonteCarloRuns_dict['GeneratedRandomNumbersForReproducibility', MonteCarlo_i] = random_seed
            ###############################################################################
            # NOTES: shuffle here is True
            X_train1, X_test, y_train1, y_test = train_test_split(X, y,
                                                                  shuffle=True,
                                                                  stratify=y,
                                                                  test_size=TrainTest_SplitRatio,
                                                                  random_state=random_seed)
            MonteCarloRuns_dict['Overall_Training_DataLabels', MonteCarlo_i] = y_train1
            MonteCarloRuns_dict['Overall_Test_DataLabels', MonteCarlo_i] = y_test
            MonteCarloRuns_dict['Overall_Test_Data', MonteCarlo_i] = X_test

            # print('**************** ----- ****************')
            # print("shape of Training Data is {}".format(X_train1.shape))
            # print("shape of Test Data is {}".format(X_test.shape))
            # print("shape of Training Data Labels is {}".format(y_train1.shape))
            # print("shape of Testing Data Labels is {}".format(y_test.shape))

            # print('Missing values in training data: ', np.sum(np.sum(np.isnan(X_train1))))
            # print('Missing values in testing data:  ', np.sum(np.sum(np.isnan(X_test))))

            # Make sure all values are finite
            # print(np.where(~np.isfinite(X_train1)))
            # print(np.where(~np.isfinite(X_test)))

            # print('*************** training/test classes distribution ***************')
            # print('Training \n', pd.value_counts(y_train1['CLASS']))
            # print('Test \n', pd.value_counts(y_test['CLASS']))

            kf = StratifiedKFold(n_splits=CV_folds,
                                 shuffle=True,
                                 random_state=random_seed
                                 )

            # Create the parameter grid
            # pm_grid = {
            #     'n_estimators':      [10, 25, 50, 100],
            #     'max_features':      ['auto', 'sqrt', 'log2'],
            #     'max_depth':         [10, 20, 30, 40],
            #     'min_samples_split': [5, 10, 15, 20],
            #     'min_samples_leaf':  [2, 5, 10, 15]
            # }
            # param_grid = ParameterGrid(pm_grid)
            # Create the parameter grid
            pm_grid = {
                'penalty': ['l1', 'l2'],
                'C': [100, 10, 1.0, 0.1, 0.01, 0.001],
                'solver': ['liblinear'],
                'max_iter': [1000, 3000, 5000]
            }
            param_grid = ParameterGrid(pm_grid)

            # print('************************* Monte Carlo Run====> ', MonteCarlo_i)
            ParamRun = 1

            Mean_CV_MCC = []
            Mean_CV_F1 = []
            Mean_CV_Precision = []
            Mean_CV_Recall = []
            Mean_CV_Specificity = []
            Mean_CV_Accuracy = []
            Sum_CV_ConfMat = []
            for params in param_grid:
                # print(params)
                # print("\r%i/%i" % (ParamRun, len(param_grid)), end="")
                # print('*************** Current Parameters Combination Run====> ', ParamRun, ' ====Total: ', len(param_grid))

                fld = 1
                for train_index_ls, validation_index_ls in kf.split(X_train1.values, y_train1.values):
                    # keeping validation set apart and oversampling in each iteration using SMOTE
                    train, validation = X_train1.iloc[train_index_ls].values, X_train1.iloc[validation_index_ls].values
                    target_train, target_val = y_train1.iloc[train_index_ls].values, y_train1.iloc[
                        validation_index_ls].values

                    # ----- no oversampling training/validation data ----
                    X_train = train
                    y_train = target_train

                    # % Logistic Regression training/validation
                    scikit_log_reg = LogisticRegression(**params)

                    # ----- LR on not-oversampled CV training data ---------------------
                    scikit_log_reg.fit(X_train, y_train.ravel())
                    validation_preds = scikit_log_reg.predict(validation)  # testing on one fold of validation set

                    cross_val_MCC_lst.append(matthews_corrcoef(target_val, validation_preds))
                    cross_val_accuracy_lst.append(accuracy_score(target_val, validation_preds))
                    cross_val_specificity_lst.append(specificity_score(target_val, validation_preds))
                    cross_val_recall_lst.append(recall_score(target_val, validation_preds))
                    cross_val_precision_lst.append(precision_score(target_val, validation_preds))
                    cross_val_f1_score_lst.append(f1_score(target_val, validation_preds))

                    cm = confusion_matrix(target_val, validation_preds)
                    cross_val_ConfMatrix_lst.append(cm)

                    fld += 1  # for train_index_ls, validation_index_ls in kf.split(X_train1.values, y_train1.values)

                # ----- save mean CV results of not-overasmpling method
                Mean_CV_MCC.append(np.mean(cross_val_MCC_lst))
                Mean_CV_F1.append(np.mean(cross_val_f1_score_lst))
                Mean_CV_Precision.append(np.mean(cross_val_precision_lst))
                Mean_CV_Recall.append(np.mean(cross_val_recall_lst))
                Mean_CV_Specificity.append(np.mean(cross_val_specificity_lst))
                Mean_CV_Accuracy.append(np.mean(cross_val_accuracy_lst))
                Sum_CV_ConfMat.append(np.sum(np.array(cross_val_ConfMatrix_lst), 0))

                ParamRun += 1  # for params in param_grid

            # ----- save the mean CV results of not-oversampling to the dictionary struct -----
            MonteCarloRuns_dict['MCC', MonteCarlo_i] = Mean_CV_MCC
            MonteCarloRuns_dict['F1', MonteCarlo_i] = Mean_CV_F1
            MonteCarloRuns_dict['Precision', MonteCarlo_i] = Mean_CV_Precision
            MonteCarloRuns_dict['Recall', MonteCarlo_i] = Mean_CV_Recall
            MonteCarloRuns_dict['Specificity', MonteCarlo_i] = Mean_CV_Specificity
            MonteCarloRuns_dict['Accuracy', MonteCarlo_i] = Mean_CV_Accuracy
            MonteCarloRuns_dict['Conf_matrix', MonteCarlo_i] = Sum_CV_ConfMat

            # Find the best CV model parameters based on the maximum F1 score
            # BestIndex = np.where(MonteCarloRuns_dict['F1', MonteCarlo_i] == np.amax(MonteCarloRuns_dict['F1', MonteCarlo_i])
            BestIndex = np.where(
                MonteCarloRuns_dict['MCC', MonteCarlo_i] == np.amax(MonteCarloRuns_dict['MCC', MonteCarlo_i]))

            # Best model parameters combination----> not-oversampling case
            LRopt_params = param_grid[int(BestIndex[0][0])]
            lr_opt = LogisticRegression(**LRopt_params)
            LR_fitted_opt = lr_opt.fit(X_train1.values, (y_train1.values).ravel())
            lr_opt_pred_prob = LR_fitted_opt.predict_proba(X_test.values)

            # ----------- Evaluation metrics on testing data
            Test_preds = LR_fitted_opt.predict(X_test.values)

            metrics = self.get_metrics(y_test, Test_preds)

            MonteCarloRuns_dict['MC_test_accuracy'].append(accuracy_score(y_test, Test_preds))
            MonteCarloRuns_dict['MC_test_specificity'].append(specificity_score(y_test, Test_preds))
            MonteCarloRuns_dict['MC_test_recall'].append(recall_score(y_test, Test_preds))
            MonteCarloRuns_dict['MC_test_precision'].append(precision_score(y_test, Test_preds))
            MonteCarloRuns_dict['MC_test_f1'].append(f1_score(y_test, Test_preds))
            MonteCarloRuns_dict['MC_test_ConfMat'].append(confusion_matrix(y_test, Test_preds))

            # -------------- Save LRopt_Params and lr_opt -----------------------------
            MonteCarloRuns_dict['LROptimum_params', MonteCarlo_i] = LRopt_params
            MonteCarloRuns_dict['LR_opt_model', MonteCarlo_i] = lr_opt
            MonteCarloRuns_dict['LR_opt_PredictedProbabilities', MonteCarlo_i] = lr_opt_pred_prob
            MonteCarloRuns_dict['LR_fitted_opt', MonteCarlo_i] = LR_fitted_opt

            fpr, tpr, thresholds = roc_curve(y_test.values, lr_opt_pred_prob[:, 1])
            roc_auc = auc(fpr, tpr)

            # Save FPR, TPR, thresholds
            MonteCarloRuns_dict['FPR_MC', MonteCarlo_i] = fpr
            MonteCarloRuns_dict['TPR_MC', MonteCarlo_i] = tpr
            MonteCarloRuns_dict['Thresholds_MC', MonteCarlo_i] = thresholds

            # -------------- Plot ROC curves -------------------------------------------
            fig = plt.figure()
            lw = 2
            plt.plot(fpr, tpr, lw=lw, label='ROC (area = %0.2f)' % (roc_auc))
            pred_vals = np.zeros(lr_opt_pred_prob.shape[0])
            for r in range(lr_opt_pred_prob.shape[0]):
                if lr_opt_pred_prob[r, 0] > lr_opt_pred_prob[r, 1]:
                    pred_vals[r] = 0
                elif lr_opt_pred_prob[r, 0] < lr_opt_pred_prob[r, 1]:
                    pred_vals[r] = 1

            Test_CM = confusion_matrix(y_test.values, pred_vals)
            # print(Test_CM)
            MonteCarloRuns_dict['TEST_Conf_matrix', MonteCarlo_i] = Test_CM

            plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
                     label='Random Guessing')
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC curves')
            plt.legend(loc="lower right")
            plt.show()

            # -------------- Plot Precision-Recall curves ------------------------------
            # predict probabilities
            lr_probs = lr_opt_pred_prob

            # keep probabilities for the positive outcome only
            lr_probs = lr_probs[:, 1]

            # predict class values
            yhat = LR_fitted_opt.predict(X_test.values)
            lr_precision, lr_recall, _ = precision_recall_curve(y_test.values, lr_probs)
            lr_f1, lr_auc = f1_score(y_test.values, yhat), auc(lr_recall, lr_precision)
            averagePrecision = average_precision_score(y_test.values, lr_probs)
            lr_MCC = matthews_corrcoef(y_test.values, yhat)

            # plot the precision-recall curves
            no_skill = len(y_test.values[y_test.values == 1]) / len(y_test.values)
            plt.figure()
            plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='k', label='No Skill')
            plt.plot(lr_recall, lr_precision, color='b', label='LR (AP = %0.2f)' % (averagePrecision))
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend()
            plt.title('Precision-Recall curves')
            plt.legend(loc='best')
            plt.show()

        self.ml = LR_fitted_opt
        self.coefs.append(LR_fitted_opt.coef_)
        self.intercepts.append(LR_fitted_opt.intercept_)
        self.glob_intercept = self.ml.intercept_
        self.glob_coef = self.ml.coef_