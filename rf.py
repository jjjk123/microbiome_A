import pandas as pd
import numpy as np
import warnings
from scipy.stats import randint
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, average_precision_score, precision_score, recall_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")
res = []

# create individual
estimators = []
classes = np.array([0, 1])
rnd_params = {
    'n_estimators': randint(low=50, high=801),
    'max_depth': randint(low=2,high=81),
    'min_samples_split': randint(low=2, high=8),
    'min_samples_leaf': randint(low=1, high=8)
}


def miss_val_tbl(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Print some summary information
    tmp = df.shape[1] - mis_val_table_ren_columns['Missing Values'].astype(bool).sum(axis=0)
    """print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(tmp) +
          " columns that have no missing values.")"""

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


def data_proc(df_input):
    # Count the number of times each unique value appears in column CLASS
    c1 = pd.value_counts(df_input['health_status'])

    # Drop 'sample' column since it does not provide any meaningful information
    df_input = df_input.drop(columns=['sample'])

    # Drop 'secondary_sample_accession' column since it does not provide any meaningful information
    df_input = df_input.drop(columns=['secondary_sample_accession'])

    # Drop 'sel_Beghini_2021' column since it does not provide any meaningful information
    df_input = df_input.drop(columns=['sel_Beghini_2021'])

    # Drop 'study_accession' column since it does not provide any meaningful information
    df_input = df_input.drop(columns=['study_accession'])

    # Drop 'sample_accession' column since it does not provide any meaningful information
    df_input = df_input.drop(columns=['sample_accession'])

    # Drop 'instrumemt_platform' column since it does not provide any meaningful information
    df_input = df_input.drop(columns=['instrument_platform'])

    # Drop 'instrumemt_model' column since it does not provide any meaningful information
    df_input = df_input.drop(columns=['instrument_model'])

    # Drop 'library_layout' column since it does not provide any meaningful information
    df_input = df_input.drop(columns=['library_layout'])

    # Drop 'sample_alias' column since it does not provide any meaningful information
    df_input = df_input.drop(columns=['sample_alias'])

    # Drop 'country' column since it does not provide any meaningful information
    df_input = df_input.drop(columns=['country'])

    # Drop 'individual_id' column since it does not provide any meaningful information
    df_input = df_input.drop(columns=['individual_id'])

    # Drop 'timepoint' column since it does not provide any meaningful information
    df_input = df_input.drop(columns=['timepoint'])

    # Drop 'body_site' column since it does not provide any meaningful information
    df_input = df_input.drop(columns=['body_site'])

    # Drop 'host_phenotype' column since it does not provide any meaningful information
    df_input = df_input.drop(columns=['host_phenotype'])

    # Drop 'host_subphenotype' column since it does not provide any meaningful information
    df_input = df_input.drop(columns=['host_subphenotype'])

    # Drop 'class' column since it does not provide any meaningful information
    df_input = df_input.drop(columns=['class'])

    # Drop 'to_exclude' column since it does not provide any meaningful information
    df_input = df_input.drop(columns=['to_exclude'])

    # Drop 'low_read' column since it does not provide any meaningful information
    df_input = df_input.drop(columns=['low_read'])

    # Drop 'low_map' column since it does not provide any meaningful information
    df_input = df_input.drop(columns=['low_map'])

    # Drop 'excluded' column since it does not provide any meaningful information
    df_input = df_input.drop(columns=['excluded'])

    # Drop 'excluded_comment' column since it does not provide any meaningful information
    df_input = df_input.drop(columns=['excluded_comment'])

    # Drop 'mgp_sample_alias' column since it does not provide any meaningful information
    df_input = df_input.drop(columns=['mgp_sample_alias'])

    # Drop 'westernised' column since it does not provide any meaningful information
    df_input = df_input.drop(columns=['westernised'])

    # Drop 'body_subsite' column since it does not provide any meaningful information
    df_input = df_input.drop(columns=['body_subsite'])

    # Drop 'HQ_clean_read_count' column since it does not provide any meaningful information
    df_input = df_input.drop(columns=['HQ_clean_read_count'])

    # Drop 'gut_mapped_read_count' column since it does not provide any meaningful information
    df_input = df_input.drop(columns=['gut_mapped_read_count'])

    # Drop 'oral_mapped_read_count' column since it does not provide any meaningful information
    df_input = df_input.drop(columns=['oral_mapped_read_count'])


    # Get one hot encoding of column 'gender'
    one_hot_gender = pd.get_dummies(df_input['gender'])
    # Drop column 'gender' as it is now encoded
    df_input = df_input.drop('gender', axis = 1)
    # Join the encoded dataframe
    df_input = df_input.join(one_hot_gender)

    return(df_input)


def data_checks(df_merged):
    # The elements of CLASS/health_status become as: minority H--> 1,  P--> 0
    df_merged['CLASS'] = pd.factorize(df_merged['health_status'])[0]

    # Drop 'health_status' column since it does not provide any meaningful information
    df_merged = df_merged.drop(columns=['health_status'])

    # Contains two columns: [1] total missing values, [2] % of missing values
    MissValsTable = miss_val_tbl(df_merged)

    # Type of each column/feature
    cols_type_df = df_merged.dtypes

    cols = df_merged.dtypes  # Type of each column/feature
    MissValsTable = pd.concat([MissValsTable, cols], axis=1)

    # Contains three columns: [1] total missing values, [2] % of missing values, [3] type of each column/feature
    MissValsTable.columns = ['missing_values', 'perc_miss_vals', 'data_type']

    sorted_MissValsTable = MissValsTable.sort_values('missing_values', ascending=False)
    plt.figure()
    plt.plot(sorted_MissValsTable['missing_values'].to_numpy())
    plt.title('(sorted) missing values per column in MIMIC')

    return sorted_MissValsTable, df_merged


def read_data(c : str) -> tuple:
    df_anno = pd.read_csv("../data/%s/anno.csv" % c)
    df_exp = pd.read_csv("../data/%s/exp.csv" % c)
    df_merged = pd.merge(df_anno, df_exp.rename(columns={'Unnamed: 0': 'sample'}), on='sample', how='left')

    df_merged = data_proc(df_merged)

    # Total training data
    X = (df_merged.loc[:, df_merged.columns != 'CLASS'])

    # Target/labels array
    y = (df_merged.loc[:, df_merged.columns == 'CLASS'])

    if c == "CHN":
        sorted_MissValsTable, df_merged = data_checks(df_merged)
        df_merged['bmi'] = df_merged['bmi'].fillna(0)
        df_merged['age'] = df_merged['age'].fillna(0)
    elif c == "FRA":
        sorted_MissValsTable, df_merged = data_checks(df_merged)
        df_merged['bmi'] = df_merged['bmi'].fillna(0)
    elif c == "ITA":
        sorted_MissValsTable, df_merged = data_checks(df_merged)
        df_merged['bmi'] = df_merged['bmi'].fillna(0)
    elif c == "USA":
        sorted_MissValsTable, df_merged = data_checks(df_merged)
        df_merged['bmi'] = df_merged['bmi'].fillna(0)
    else:
        sorted_MissValsTable, df_merged = data_checks(df_merged)

    # Total training data
    X = (df_merged.loc[:, df_merged.columns != 'CLASS'])

    # Target/labels array
    y = (df_merged.loc[:, df_merged.columns == 'CLASS'])

    return X, y

def custom_scorer(y_true, y_pred, actual_scorer):
    score = np.nan
    try:
      score = actual_scorer(y_true, y_pred)
    except Exception:
      pass
    return score

#def get_data(path: str) -> tuple:
#    data = np.array(pandas.read_csv(path, delimiter=","))
#    X = data[:, :-1]
#    y = data[:,-1:]
#    y = y[:,0]
#    return X, y

def get_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    aps = average_precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    prs = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return "Accuracy: %.3f, Precision: %.3f, Recall: %.3f, APS: %.3f, F1: %.3f, MCC: %.3f" % (acc, prs, recall, aps, f1, mcc)


mcc_scorer = make_scorer(custom_scorer, actual_scorer = matthews_corrcoef)

countries = ["AUT", "CAN", "CHN", "FRA", "IND", "ITA", "JPN", "USA"]

for c in countries:
    warnings.filterwarnings("ignore", category=DataConversionWarning)
    X, y = read_data(c)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
    clf = RandomForestClassifier(max_depth=2)
    #clf_cv = RandomizedSearchCV(clf, rnd_params, random_state=0, n_jobs=-1, n_iter=200, scoring=mcc_scorer, cv=3)
    clf_cv = RandomizedSearchCV(clf, param_distributions=rnd_params, n_jobs=-1, n_iter=100, scoring="f1", cv=3)
    clf_cv.fit(X_train.values, y_train.values)
    clf = clf_cv.best_estimator_
    #clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)
    estimators.extend(clf.estimators_)
    res.append("%s= %s" % (c, get_metrics(y_test, y_pred_test)))
    print("%s= %s" % (c, get_metrics(y_test, y_pred_test)))

print("Federated")
glb_clf = RandomForestClassifier()
glb_clf.estimators_ = estimators
glb_clf.classes_ = classes
glb_clf.n_classes_ = 2
glb_clf.n_outputs_ = 1

print(res)
for c in countries:
    X, y = read_data(c)
    y_pred = glb_clf.predict(X)
    print("%s= %s" % (c, get_metrics(y, y_pred)))
