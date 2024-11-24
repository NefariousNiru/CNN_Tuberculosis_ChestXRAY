from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from util import data_util

def run_rf(train_feature_file, train_label_file, test_feature_file, test_label_file):
    train_features, train_labels, test_features, test_labels, classes = (
        data_util.feature_pre_processing(train_feature_file, train_label_file, test_feature_file, test_label_file, smote=True))

    rf = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf=3, min_samples_split=8, random_state=42, n_jobs=4)
    rf.fit(train_features, train_labels)

    data_util.f_statistics(rf, test_features, test_labels, classes)