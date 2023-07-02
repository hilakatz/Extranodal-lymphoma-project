import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mode
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_curve, plot_roc_curve, plot_confusion_matrix, roc_auc_score, auc


def hist_by_feature(df: pd.core.frame.DataFrame) -> None:
    for feature in df:
        df[feature].hist()
        plt.title(feature)
        plt.show()
        plt.close()


def heatmap(df: pd.core.frame.DataFrame) -> None:
    corrmat = df.drop('Unnamed: 0', axis='columns').corr()
    f, ax = plt.subplots()
    sns.heatmap(corrmat, vmax=.8, square=True, cmap="YlGnBu",annot=True)
    plt.show()


def split_pos_and_neg(df: pd.core.frame.DataFrame):
    y_pos = df.apply(lambda x: x['Number of Extranodal Sites'] == 1,axis='columns')
    y_neg = df.apply(lambda x: x['Number of Extranodal Sites'] == 0,axis='columns')
    for feature in df.drop('Number of Extranodal Sites',axis='columns').select_dtypes(exclude=['object']).columns:
        not_na = -df[feature].isna()
        pos = df[feature][not_na][y_pos]
        neg = df[feature][not_na][y_neg]
        plt.hist(pos, color= 'r', alpha= 0.4, label='Positive', density=True)
        plt.hist(neg, color= 'b', alpha= 0.4, label='Negative', density=True)
        plt.legend()
        plt.title(feature)
        plt.show()
        plt.close()


def feature_exploration(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    # change binary features to 0\1
    df['Included in Survival Analysis'] = df['Included in Survival Analysis'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['Number of Extranodal Sites'] = df['Number of Extranodal Sites'].apply(lambda x: 1 if x > 0 else 0) #turn label to binary
    df['LDH Ratio'] = df['LDH Ratio'].apply(lambda x: np.nan if x == -1 or x== 0 else x)
    return df


def create_dummy_variables(categorial_sereies: pd.core.frame.DataFrame, prefix:str = '') -> pd.core.frame.DataFrame:
    dummy_df = pd.get_dummies(categorial_sereies)
    dummy_df = dummy_df.add_prefix(prefix)
    dummy_df.columns = dummy_df.columns.str.replace(' ', '_')
    return dummy_df


def fill_missing_values_by_mice(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """
    This type of imputation works by filling the missing data multiple times.
    Multiple Imputations (MIs) are much better than a single imputation as it measures
    the uncertainty of the missing values in a better way. The chained equations approach is also very
    flexible and can handle different variables of different data types (ie., continuous or binary) as well as
    complexities such as bounds or survey skip patterns. For more information on the algorithm mechanics,
    you can refer to the Research Paper - https://www.jstatsoft.org/article/view/v045i03/v45i03.pdf
    """
    from impyute.imputation.cs import mice


    # start the MICE training
    imputed_training = mice(df.values)
    return pd.DataFrame(imputed_training, columns=df.columns, index=df.index)


def fill_missing_values_by_KNN(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """
    It creates a basic mean impute then uses the resulting complete list to construct a KDTree.
    Then, it uses the resulting KDTree to compute nearest neighbours (NN). After it finds the k-NNs,
    it takes the weighted average of them.
    """
    import sys
    from impyute.imputation.cs import fast_knn
    df.astype('float64').dtypes
    sys.setrecursionlimit(100000)  # Increase the recursion limit of the OS

    # start the KNN training
    imputed_training = fast_knn(df.values, k=15)
    return pd.DataFrame(imputed_training, columns=df.columns, index=df.index)


def train_test_split_and_rename(data: pd.core.frame.DataFrame, labels: pd.core.frame.DataFrame, test_size=0.30, random_state=42):
    try:
        data = data.drop(['Number of Extranodal Sites'],axis=1)
        labels = pd.DataFrame(labels,columns=['Number of Extranodal Sites'],index=data.index)
    except:
        pass
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=test_size, random_state=random_state)
    return X_train, X_val, y_train, y_val


def run_model_and_draw_ROC(clf, X_train: pd.core.frame.DataFrame, y_train: pd.core.frame.DataFrame,
                           X_val: pd.core.frame.DataFrame, y_val:pd.core.frame.DataFrame):
    model_name = str(type(clf)).split('.')[-1].strip('>').replace("'",'')
    clf.fit(X_train,y_train)
    plot_roc_curve(clf, X_val,y_val)
    fpr, tpr, thresholds = roc_curve(y_val, clf.predict_proba(X_val)[:, 1])
    clf_auc = auc(fpr, tpr)
    plt.title(model_name)
    plt.show()
    plt.close()
    return clf_auc


def open_file_and_save(name, dic):
    f = open(name+'.txt',"w+")
    f.write(f"{name} results: {dic}")
    f.close()


def fill_missing_values_traditional_way(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    df = df.fillna(value=np.nan)
    df = df.astype('float64')
    features_median = ['LDH Ratio', 'Age']
    for feature in features_median:
        mead = df[feature].median()
        df[feature].fillna(mead, inplace=True)
    features_mode = ['Ann Arbor Stage', 'IPI Group', 'ECOG Performance Status']
    for feature in features_mode:
        feat_mode = mode(df[feature])[0][0]
        df[feature] = df[feature].fillna(feat_mode)
    return df


def plot_multiple_ROC_curves(X_train, y_train, X_val, y_val,classifiers, relevant_features):
    # Train the models and record the results
    for cls in classifiers:
        model_name = str(type(cls)).split('.')[-1].strip('>').replace("'", '')
        temp_X_train = X_train[relevant_features[cls]]
        temp_X_val = X_val[relevant_features[cls]]
        print(f'started working on {model_name}')
        model = cls.fit(temp_X_train, y_train)
        yproba = model.predict_proba(temp_X_val)[::, 1]
        fpr, tpr, _ = roc_curve(y_val, yproba)
        auc = roc_auc_score(y_val, yproba)
        plt.plot(fpr,
                 tpr,
                 label=f"{cls.__class__.__name__},AUC= {auc:.3f}")


    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Flase Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size': 13}, loc='lower right')

    plt.show()


def change_IPIGroup_col(df: pd.core.frame.DataFrame):
    df['IPI Group'].replace({'Low': 1,
                             'Intermediate': 2,
                             'High': 3}, inplace=True)
    return df


def change_Gender_col(df: pd.core.frame.DataFrame):
    df['Gender'].replace({'F': 0,
                          'M': 1}, inplace=True)
    return df


def preprocessing_dataframe(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    print("num of rows in the start: ", len(df))
    print("num of rows with na in more than 4 features: ", df.iloc[df[(df.isnull().sum(axis=1) >= 4)].index][
        'dbGaP submitted subject ID'].count())  # num of rows with nulls in more than x features
    df.dropna(subset=['Number of Extranodal Sites'], inplace=True)
    print("num of rows after deleting null lables: ", len(df))
    df_check = df[df.isnull().sum(axis=1) >= 4]
    print("num of rows with na in more than 4 features after deleting rows with no labels: ",
          len(df_check['dbGaP submitted subject ID']))  # num of rows with nulls in more than x features
    df = feature_exploration(df)
    df = change_IPIGroup_col(df)
    df = change_Gender_col(df)
    df.drop(['Diagnosis',
             'Treatment__',
             'Biopsy Type', 'Status at Follow_up_ 0 Alive_ 1 Dead',
             'Follow_up Time _yrs', 'Progression_Free Survival _PFS_ Status_ 0 No Progressoin_ 1 Progression',
             'Progression_Free Survival _PFS_ Time _yrs', 'Included in Survival Analysis'], axis=1, inplace=True)
    ## change all categorial and not ordinal features to dummy variables
    dbGaP_accession_dummy = create_dummy_variables(df['dbGaP accession'], 'dbGaP_accession_')
    Gene_Expression_Subgroup_dummy = create_dummy_variables(df['Gene Expression Subgroup'],
                                                               'Gene_Expression_Subgroup_')
    Genetic_Subtype_dummy = create_dummy_variables(df['Genetic Subtype'], 'Genetic_Subtype_')
    ## delete all original columns and concat the dummies
    features_to_remove = ['dbGaP accession', 'Gene Expression Subgroup', 'Genetic Subtype']
    all_features_df = pd.concat([df, dbGaP_accession_dummy, Gene_Expression_Subgroup_dummy, Genetic_Subtype_dummy],
                                axis=1)
    numeric_df = all_features_df.drop(features_to_remove, axis=1)
    numeric_df = numeric_df.set_index(['dbGaP submitted subject ID'])
    curr_df = numeric_df.drop(['Unnamed: 0'], axis=1)
    curr_df['LDH Ratio'] = np.log(curr_df['LDH Ratio'])

    return curr_df


def KfoldPlot(X, y, clf, k):
    X = np.array(X)
    y = np.array(y)
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    # kf = ShuffleSplit(n_splits=k, random_state=42, test_size=0.1, train_size=None)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure()
    i = 1
    for train_index, val_index in kf.split(X,y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        clf.fit(X_train, y_train)
        prob_prediction = clf.predict_proba(X_val)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_val, prob_prediction)
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        plt.plot(fpr, tpr, lw=2, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, auc(fpr, tpr)))  # plotting current fold
        i += 1
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='random-guess')
    mean_tpr /= k
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for ' + str(clf)[:str(clf).find("(")])
    plt.plot(mean_fpr, mean_tpr, color='red', linestyle='-', label='Mean ROC (area = %0.3f)' % mean_auc)
    plt.plot([0], [0], color='#D3D3D3', linestyle='-', label='K-folds')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()
    plt.close()


def combine_train_set_and_val_set(X_train: pd.core.frame.DataFrame, X_val: pd.core.frame.DataFrame,
                                            y_train: pd.core.frame.DataFrame,y_val: pd.core.frame.DataFrame):
    X = pd.concat([X_train,X_val])
    y = pd.concat([y_train,y_val])
    return X,y


def plot_train_test_ROC_curves(X_train, y_train, X_val, y_val,cls):
    result_table = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc'])

    # Train the models and record the results
    # fit on train predict val
    model = cls.fit(X_train, y_train)
    yproba1 = model.predict_proba(X_val)[::, 1]
    fpr1, tpr1, _ = roc_curve(y_val, yproba1)
    auc = roc_auc_score(y_val, yproba1)

    result_table = result_table.append({'classifiers': f'{cls.__class__.__name__}_validation',
                                        'fpr': fpr1,
                                        'tpr': tpr1,
                                        'auc': auc}, ignore_index=True)

    #fit on train predict train
    #model2 = cls.fit(X_train, y_train)
    model2 = model
    yproba2 = model2.predict_proba(X_train)[::, 1]
    fpr2, tpr2, _ = roc_curve(y_train, yproba2)
    auc2 = roc_auc_score(y_train, yproba2)

    result_table = result_table.append({'classifiers': f'{cls.__class__.__name__}_train',
                                            'fpr': fpr2,
                                            'tpr': tpr2,
                                            'auc': auc2}, ignore_index=True)

    # Set name of the classifiers as index labels
    result_table.set_index('classifiers', inplace=True)

    fig = plt.figure(figsize=(8, 6))

    for i in result_table.index:
        plt.plot(result_table.loc[i]['fpr'],
                 result_table.loc[i]['tpr'],
                 label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))

    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Flase Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size': 13}, loc='lower right')

    plt.show()

def run_multiple_kfolds(X, y, models, k):
    for clf in models:
        KfoldPlot(X, y, clf, k)

def run_grid_search(clf, param_grid: dict, X_train, y_train, X_val, y_val):
        model_name = str(type(clf)).split('.')[-1].strip('>').replace("'",'')
        print(f'strated procces {model_name}')
        gs = GridSearchCV(clf, param_grid, cv=10, scoring="roc_auc").fit(X_train, y_train)
        fpr, tpr, thresholds = roc_curve(y_val, gs.predict_proba(X_val)[:, 1])
        print(f'model name: {model_name}\nBest Params: {gs.best_params_}\nBest Params: {gs.best_score_}\nThe AUC is: {auc(fpr, tpr)}\ntested params: {param_grid}')
        data = f'model name: {model_name}\nBest Params: {gs.best_params_}\nBest Params: {gs.best_score_}\nThe AUC is: {auc(fpr, tpr)}\ntested params: {param_grid}'
        open_file_and_save(model_name,data)
        print(f'Finished with {model_name}')
        return gs.best_params_


def run_random_search_and_gridsearch(X_train, y_train, X_val, y_val):

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(X_train, y_train)
    best = rf_random.best_params_
    print(best)
    return rf_random.best_estimator_


def cross_validation(k,X,y,clf):
    X = np.array(X)
    y = np.array(y)
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    classifier = clf
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure()
    i = 1
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1.5, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='random-guess', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Cross-Validation ROC of {clf.__class__.__name__}')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()
    return mean_auc


def get_reduced_dimensions_df(train_df: pd.core.frame.DataFrame, val_df: pd.core.frame.DataFrame,
                              n_components=0.95) -> pd.core.frame.DataFrame:
    pca = PCA(n_components)
    pca.fit(train_df)
    val_pca = pd.DataFrame(pca.transform(val_df), index=val_df.index)
    train_pca = pd.DataFrame(pca.transform(train_df), index=train_df.index)
    return train_pca, val_pca


def load_data_and_transform(path,name_data):
    data_path = path
    data_xls = pd.read_excel(data_path, index_col=None, dtype=str)
    if not os.path.isfile(f"./{name_data}.csv"):
        data_xls.to_csv(f"./{name_data}.csv", index=False)
    data_csv = rf"./{name_data}.csv"
    df = pd.read_csv(data_csv)
    return df


def join_genomics(genomic_df,X_train,y_train,X_val=None,y_val=None):
    print(genomic_df.shape)
    print(genomic_df.head())
    genomic_df = genomic_df.drop(['Gene_ID', 'Unnamed: 0', 'Accession'], axis=1)
    genomic_df_T = genomic_df.T
    genomic_df_T.columns = genomic_df_T.iloc[0]
    genomic_df_T = genomic_df_T.drop(['Gene'], axis=0)
    genomic_df_T.rename_axis('dbGaP submitted subject ID', inplace=True)
    print(genomic_df_T.head())
    print(genomic_df_T.shape)
    X_train = pd.merge(X_train, genomic_df_T, on='dbGaP submitted subject ID')
    y_train = pd.merge(y_train, X_train, on='dbGaP submitted subject ID', how='right')['Number of Extranodal Sites']
    X_val = pd.merge(X_val, genomic_df_T, on='dbGaP submitted subject ID')
    y_val = pd.merge(y_val, X_val, on='dbGaP submitted subject ID', how='right')['Number of Extranodal Sites']
    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    return X_train, y_train, X_val, y_val


def draw_multiple_histograms(df: pd.core.frame.DataFrame, number_of_columns: int, number_of_rows: int):
    ncols = number_of_columns
    nrows = int(np.ceil(len(df.columns) / (1.0 * ncols)))
    ncols = 3
    nrows = 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

    # Lazy counter so we can remove unwated axes
    counter = 0
    for i in range(nrows):
        for j in range(ncols):

            ax = axes[i][j]

            # Plot when we have data
            if counter < len(df.columns):

                ax.hist(df[df.columns[counter]].dropna(), bins=10, color='blue', alpha=0.5,
                        label='{}'.format(df.columns[counter]))
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                # ax.set_ylim([0, 5])
                leg = ax.legend(loc='upper left')
                leg.draw_frame(False)

            # Remove axis when we no longer have data
            else:
                ax.set_axis_off()

            counter += 1

    plt.show()