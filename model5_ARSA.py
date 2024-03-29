### add packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC, SVR
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
#from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score, accuracy_score, classification_report, roc_auc_score
from sklearn.decomposition import PCA
from scipy.stats import pearsonr


###ClinvarSet1Processed and ClinvarSet2Processed have different Y label meaning, Benign, Pathogenic, likely Benign, and likely Pathogenic
Dtrain = pd.read_excel("ClinvarSet2Processed.xlsx")

Dpred_ = pd.read_excel("ArsaProcessed.xlsx")
Dpred = pd.read_excel("ArsaProcessed.xlsx")

### choose categorical features and numerical features
categorical_features = [x for x in Dtrain.select_dtypes(exclude=['number']).columns.values if x != "Name"]
numeric_features = [x for x in Dtrain.select_dtypes(include=['number']).columns.values if x != "Name"]

#print(categorical_features)
#check the share columns between Dtrain and Dpred
ShareColumns = (np.intersect1d(Dtrain.columns, Dpred_.columns))
ShareColumns = (np.intersect1d(ShareColumns, numeric_features))

ShareColumns = ShareColumns.tolist()
ShareColumns = np.array(ShareColumns)

ShareColumns_noY = ShareColumns
ShareColumns = np.append(ShareColumns,'Clinical_significance')
print (len(ShareColumns_noY), len(ShareColumns))
# reduce down to numeric columns for training
Dtrain=Dtrain[ShareColumns]

Dpred=Dpred[ShareColumns_noY]

Dpred.fillna(Dtrain.mean(), inplace=True)
Dtrain.fillna(Dtrain.mean(), inplace=True)

def cal_AUC(model_name, X_Train, Y_Train, X_Test, Y_Test):
    if model_name == 'LogisticRegression':
        model = LogisticRegression()
    if model_name == 'Perceptron':
        model = Perceptron()
    if model_name == 'SVC':
        model = SVR()  
    if model_name == 'KNN':
        model = KNeighborsRegressor()   
    if model_name == 'DecisionTree':
        model = DecisionTreeRegressor()
    if model_name == 'RandomForest':
        model = RandomForestRegressor()
    if model_name == 'GaussianNB':
        model = GaussianNB()
    if model_name == 'NeuralNetwork':
        model = MLPRegressor(hidden_layer_sizes=(15,), random_state=1, max_iter=5000)
    model.fit(X_Train, Y_Train)
    if model_name in ['Perceptron', 'SVC']:
        Y_train_pred = model.predict(X_Train)
        Y_test_pred = model.predict(X_Test)

        AUC_train = roc_auc_score(Y_Train, Y_train_pred)
        AUC_test = roc_auc_score(Y_Test, Y_test_pred)
    else:
        Y_train_pred = model.predict(X_Train)
        Y_test_pred = model.predict(X_Test)

    R2_train = pearsonr(Y_Train, Y_train_pred)[0]
    R2_test = pearsonr(Y_Test, Y_test_pred)[0]
 
    return R2_train, R2_test

def cal_AUC_no_test(model_name, X_Train, Y_Train, Dpred_Zscore_X, Dpred_):
    if model_name == 'LogisticRegression':
        model = LogisticRegression()
    if model_name == 'Perceptron':
        model = Perceptron()
    if model_name == 'SVC':
        model = SVR()  
    if model_name == 'KNN':
        model = KNeighborsRegressor()   
    if model_name == 'DecisionTree':
        model = DecisionTreeRegressor()
    if model_name == 'RandomForest':
        model = RandomForestRegressor(n_estimators=100, max_features=15, max_depth=None, min_samples_split=2, min_samples_leaf=1, bootstrap=True, random_state = 70)
    if model_name == 'GaussianNB':
        model = GaussianNB()
    if model_name == 'NeuralNetwork':
        model = MLPRegressor(hidden_layer_sizes=(15,), random_state=1, max_iter=5000)
    model.fit(X_Train, Y_Train)
    if model_name in ['Perceptron', 'SVC']:
        Y_train_pred = model.predict(X_Train)
        Y_Dpred_Zscore = model.predict(Dpred_Zscore_X)
        df_Y_Dpred_Zscore = pd.DataFrame({'aa_substitution': Dpred_.ProteinChange,'score': Y_Dpred_Zscore,'sd':1, 'comments':''})   
    else:
        Y_train_pred = model.predict(X_Train)
        Y_Dpred_Zscore = model.predict(Dpred_Zscore_X)
        df_Y_Dpred_Zscore = pd.DataFrame({'aa_substitution': Dpred_.ProteinChange,'score': Y_Dpred_Zscore,'sd':1, 'comments':'NA'})   
    
    R2_train = pearsonr(Y_Train, Y_train_pred)[0]

    df_Y_Dpred_Zscore.to_csv('Team3_model5.tsv', index=False, sep='\t')
    
    return R2_train, None #R2_test #None #AUC_train, AUC_test

def cal_AUC_no_test_RF(model_name, X_Train, Y_Train, Dpred_Zscore_X, Dpred_Zscore_Y): #, Dpred_):
    for n_estimators in [200]: #100, 200, 300]:
        for max_features in [15]: #5,10,15,'auto']:
            for max_depth in [None]: #None, 10, 20]:
                for min_samples_split in [5]: #2, 5, 8]:
                    for min_samples_leaf in [1]: #, 4, 7]:
                        for bootstrap in [True]:
                            for random_state in [70]:
                                model = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, random_state=random_state)
        
                                model.fit(X_Train, Y_Train)
                                Y_train_pred = model.predict(X_Train)
                                Y_Dpred_Zscore = model.predict(Dpred_Zscore_X)
                                
                                R2_train = pearsonr(Y_Train, Y_train_pred)[0]
                                R2_test = pearsonr(Dpred_Zscore_Y, Y_Dpred_Zscore)[0]
                                print (n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf, bootstrap, random_state, R2_train, R2_test)
    return R2_train, R2_test #AUC_train, AUC_test

models = ['RandomForest']#, 'NeuralNetwork', 'KNN', 'DecisionTree'] #, 'LogisticRegression' , 'Perceptron', , 'GaussianNB', 'SVC', 
def run_pred(models, Dtrain, ShareColumns_noY, Dpred):
    for model_name in models:
        print (model_name)
        AUC_train_sets, AUC_test_sets = [], []
        pca_AUC_train_sets, pca_AUC_test_sets = [], []
        pca = PCA(n_components=27)
        
        ss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
        
        Dtrain_X = Dtrain[ShareColumns_noY]

        Dtrain_Y = Dtrain["Clinical_significance"]

        Dtrain_Y = Dtrain_Y.replace({'Benign': 0})
        Dtrain_Y = Dtrain_Y.replace({'Likely_benign': 0.33})
        Dtrain_Y = Dtrain_Y.replace({'Benign/Likely_benign': 0.33})
        Dtrain_Y = Dtrain_Y.replace({'Pathogenic/Likely_pathogenic': 0.66})
        Dtrain_Y = Dtrain_Y.replace({'Likely_pathogenic': 0.66})
        Dtrain_Y = Dtrain_Y.replace({'Pathogenic': 1})

        Dpred_X = Dpred[ShareColumns_noY]

        AUC_train, AUC_test = cal_AUC_no_test(model_name, Dtrain_X, Dtrain_Y, Dpred_X, Dpred_)
        print (AUC_train, AUC_test)

run_pred(models, Dtrain, ShareColumns_noY, Dpred)
