import pandas as pd
from tqdm import tqdm
import utilities as ut
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, plot_confusion_matrix, roc_auc_score, classification_report,\
    accuracy_score, precision_score

if __name__ == '__main__':

    # =================================================================================
    # ==========================load prognostics data==================================
    data_path = r"/ML2/project/extra_cell_lymphoma_dev/prognostics_dev.xlsx"  # enter path
    df = ut.load_data_and_transform(data_path, "prognostics")

    # =================================================================================
    # ===========================data exploration&label handling=======================
    # we saw that their are patient w/o label.
    # in addition we saw that there is corr between missing labels and na in more than 4 features.
    # thus we decided to delete rows w/o labels.
    print(df.isnull().sum(axis=0))
    print(df.describe())
    for i in range(2,27,9): # features distribution
        ut.draw_multiple_histograms(df[df.columns[i:i+9]], 2, 2)
    ut.heatmap(df)
    ut.split_pos_and_neg(df)
    # we decided to delete number of features according to their distributions:
    # 1. Diagnosis - obviously all the patients has the same diagnosis so this is redundant
    # 2. Treatment - half of the data is missing and the other half has the same treatment.
    # 3. Biopsy Type - the major of the feature is pre-treatment
    # these features from the future thus can't be part of the model
    # 1. Included in survival analysis
    # 2. all progressions
    # 3. follow ups
    curr_df = ut.preprocessing_dataframe(df)
    print(curr_df.describe())
    # ==================================================================================================================
    # ============================= baseline model imputed missing values by traditional way ===========================
    baseline_df_trad = curr_df.copy()
    baseline_df_trad = ut.fill_missing_values_traditional_way(baseline_df_trad)
    baseline_df_trad = baseline_df_trad.round(
        {'Age': 0, 'IPI Group': 0, 'Ann Arbor Stage': 0, 'ECOG Performance Status': 0})
    X_train_trad, X_val_trad, y_train_trad, y_val_trad = ut.train_test_split_and_rename(baseline_df_trad,
                                                                                        baseline_df_trad[
                                                                                            'Number of Extranodal Sites'
                                                                                        ])

    rf0 = RandomForestClassifier()
    log_r0 = LogisticRegression()
    knn0 = KNeighborsClassifier()
    mlp0 = MLPClassifier()
    models = [rf0, log_r0, knn0, mlp0]
    for clf in models:
        mean_auc = ut.cross_validation(10,X_train_trad,y_train_trad,clf)
        if clf == rf0:
            baseline_auc = mean_auc

    # ==================================================================================================================
    # =============================== baseline model with missing values imputed by KNN ================================
    curr_df = ut.fill_missing_values_by_KNN(curr_df)
    curr_df = curr_df.round({'Age': 0, 'IPI Group': 0, 'Ann Arbor Stage': 0, 'ECOG Performance Status': 0})
    # split train labels and validation
    X_train, X_val, y_train, y_val = ut.train_test_split_and_rename(curr_df, curr_df['Number of Extranodal Sites'])
    rf = RandomForestClassifier()
    log_r = LogisticRegression()
    knn = KNeighborsClassifier()
    mlp = MLPClassifier()
    models = [rf, log_r, knn, mlp]
    for clf in models:
        ut.cross_validation(10,X_train,y_train,clf)
    # =====================================================================================================
    # ============================================grid_search==============================================
    # rf = RandomForestClassifier()
    # rf_param_grid = {
    #      'max_depth': [10,20,50,80,110],
    #      'max_features': ['auto', 'sqrt'],
    #      'min_samples_leaf': [1,2,6,10],
    #      'min_samples_split': [2,6,10],
    #      'n_estimators': [10,100,600,1000]}
    # train_size = 0.8
    # # In the first step we will split the data in training and remaining dataset
    # X_train_grid, X_rem, y_train_grid, y_rem = X_train_trad, X_val_trad, y_train_trad, y_val_trad
    #
    # # Now since we want the valid and test size to be equal (10% each of overall data).
    # # we have to define valid_size=0.5 (that is 50% of remaining data)
    # test_size = 0.5
    # X_val_grid, X_test_grid, y_val_grid, y_test_grid = ut.train_test_split(X_rem, y_rem, test_size=0.5)
    #
    # best_param = ut.run_grid_search(rf, rf_param_grid, X_train_grid, y_train_grid, X_val_grid, y_val_grid)
    #
    # # rf_grid = RandomForestClassifier(**best_param)
    # # rf_grid.fit(X_train_grid,y_train_grid)
    # # plot_roc_curve(rf_grid, X_test_grid,y_test_grid)
    # # plt.show()
    # ==========================================================================================================
    # the two Imputing methods yield the same results, thus we will continue with the traditional baseline
    # In addition, after gridsearch we found the best parameters are the default

    X_train, X_val, y_train, y_val = X_train_trad, X_val_trad, y_train_trad, y_val_trad
    # ==========================================================================================================
    # ===================================load genetic data======================================================
    data_path = r"/ML2/project/extra_cell_lymphoma_dev/genomics_dev.xlsx"  # enter  path
    genomic_df = ut.load_data_and_transform(data_path, "genomics")
    X_train, y_train, X_val, y_val = ut.join_genomics(genomic_df, X_train, y_train, X_val, y_val)
    # ==========================================================================================================
    # ===================================drop features by domain knowledge =====================================
    X_train = X_train.drop(['ECOG Performance Status', 'IPI Range', 'dbGaP_accession_phs000178',
                            'dbGaP_accession_phs001444', 'Gene_Expression_Subgroup_ABC', 'Gene_Expression_Subgroup_GCB',
                            'Gene_Expression_Subgroup_Unclass', 'Genetic_Subtype_BN2', 'Genetic_Subtype_EZB',
                            'Genetic_Subtype_MCD', 'Genetic_Subtype_N1', 'Genetic_Subtype_Other'], axis=1)
    X_val = X_val.drop(['ECOG Performance Status', 'IPI Range', 'dbGaP_accession_phs000178',
                        'dbGaP_accession_phs001444', 'Gene_Expression_Subgroup_ABC', 'Gene_Expression_Subgroup_GCB',
                        'Gene_Expression_Subgroup_Unclass', 'Genetic_Subtype_BN2', 'Genetic_Subtype_EZB',
                        'Genetic_Subtype_MCD', 'Genetic_Subtype_N1', 'Genetic_Subtype_Other'], axis=1)
    # =========================================================================================================
    # ===================================run basic model with genomic data=====================================
    print("=================run basic model with all genomic data=================")
    rf2 = RandomForestClassifier()
    auc = ut.run_model_and_draw_ROC(rf2, X_train, y_train, X_val, y_val)
    print(f"for the baseline model with all the genomic data the AUC is: {auc}")

    # =========================================================================================================
    # ===================================we will try reducing dimension by using pca===========================
    print("=================run model with reduced dimensions by PCA=================")
    X_train_pca, X_val_pca = ut.get_reduced_dimensions_df(X_train, X_val)
    y_train_pca = y_train.reindex(X_train_pca.index)
    y_val_pca = y_val.reindex(X_val_pca.index)
    print(f"Reduced dimensions to: {X_train_pca.shape}")
    rf3 = RandomForestClassifier()
    auc = ut.run_model_and_draw_ROC(rf3, X_train_pca, y_train_pca, X_val_pca, y_val_pca)
    print(f"for the model with reduced dimensions by PCA the AUC is: {auc}")

    # =========================================================================================================
    # ===================================We will try feature selection by Domain knowledge====================
    print("=================run model with features selected by Domain knowledge=================")
    relevant_features = ['Gender', 'Age', 'Ann Arbor Stage', 'LDH Ratio',
                         'IPI Group', 'JAM2', 'JAM3', 'MYC', 'POU2AF1', 'EZH2',
                         'BCL2', 'AQP9',
                         'LMBR1L', 'FGF20', 'TANK', 'CRP', 'ORM1', 'JAK1', 'BACH1', 'MTCP1', 'IFITM1', 'TNFSF10',
                         'FGF12', 'RFX5', 'LAP3']
    print(f"The selected features are:{relevant_features}")
    X_train_fs = X_train[relevant_features]
    X_val_fs = X_val[relevant_features]

    rf4 = RandomForestClassifier()
    auc = ut.run_model_and_draw_ROC(rf4, X_train_fs, y_train, X_val_fs, y_val)
    print(f"for the model with features selected by Domain knowledge the AUC is: {auc}")

    # =========================================================================================================
    # =================================== lets try feature selection using forward selection ==================
    def forward_selection(data, target, X_val, y_val,auc, significance_level=0.01):
        initial_features = data.columns.tolist()
        best_features = ['Gender', 'Age', 'Ann Arbor Stage', 'LDH Ratio',
                         'IPI Group']
        scores = [auc,auc,auc]
        best_features_and_scores = {}
        while (len(initial_features) > 0):
            print(best_features_and_scores)
            remaining_features = list(set(initial_features) - set(best_features))
            new_pval = pd.Series(index=remaining_features)
            for new_column in tqdm(remaining_features):
                new_pval[new_column] = 0
                for i in range(3):
                    model = RandomForestClassifier().fit(data[best_features + [new_column]], target)
                    new_pval[new_column] += roc_auc_score(y_val,
                                                          model.predict_proba(X_val[best_features + [new_column]])[:,
                                                          1])
                new_pval[new_column] = new_pval[new_column] / 3
                print(f"{new_column}:", new_pval[new_column])
            max_score = new_pval.max()
            if (max_score - scores[-1] > significance_level):
                best_features.append(new_pval.idxmax())
                scores.append(max_score)
                best_features_and_scores[new_pval.idxmax()] = max_score
            else:
                best_features.append(new_pval.idxmax())
                scores.append(scores[-1])
                if scores[-1] == scores[-2] and scores[-1] == scores[-3]:
                    best_features = best_features[:len(best_features) - 2]
                    print(f"scores: {scores}")
                    break
        return best_features, best_features_and_scores

    # =========================================================================================================
    # ===================3 experiments for the feature selection===============================================
    for experiment in range(1,4):
        best_features, best_features_and_scores = forward_selection(X_train, y_train, X_val, y_val, baseline_auc)
        print(f"best features : {best_features}")
        print(f"best features for exp. {experiment}: {best_features_and_scores}")
        ut.open_file_and_save(f"exp.{experiment}", best_features_and_scores)

    # after finding the best features plot roc curves that compares the baseline model to the one
    # with the best genetics features

    print("=================run model with features selected by Forward Selection=================")

    print(f"best features: {best_features}")
    rf_basic = RandomForestClassifier()
    rf_genetic = RandomForestClassifier()
    models = [rf_basic, rf_genetic]
    relevant_features = {rf_basic: ['Gender', 'Age', 'Ann Arbor Stage', 'LDH Ratio', 'IPI Group'],
                         rf_genetic: best_features}
    ut.plot_multiple_ROC_curves(X_train, y_train, X_val, y_val, models, relevant_features)
    X_train = X_train[best_features]
    X_val = X_val[best_features]

    # =========================================================================================================
    # =====================================CM and scores=============================================
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    plot_confusion_matrix(rf, X_val, y_val)
    plt.show()
    y_pred_test = rf.predict(X_val)
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_val, y_pred_test)
    print(f"Accuracy: {accuracy}")
    # precision tp / (tp + fp)
    precision = precision_score(y_val, y_pred_test)
    print(f"Precision: {precision}")
    # recall: tp / (tp + fn)
    recall = recall_score(y_val, y_pred_test)
    print(f"Recall: {recall}")
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_val, y_pred_test)
    print(f"F1 score: {f1}")
    # View the classification report for test data and predictions
    print(classification_report(y_val, y_pred_test))
    ut.plot_train_test_ROC_curves(X_train, y_train, X_val, y_val, rf)
