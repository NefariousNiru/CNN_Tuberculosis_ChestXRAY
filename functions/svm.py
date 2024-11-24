from sklearn.svm import SVC
from util import data_util

def run_svm(train_feature_file, train_label_file, test_feature_file, test_label_file):
    train_features, train_labels, test_features, test_labels, classes = (
        data_util.feature_pre_processing(train_feature_file, train_label_file, test_feature_file, test_label_file, smote=True))

    svm = SVC(kernel='poly', degree=3, gamma=0.4, C=266)
    svm.fit(train_features, train_labels)

    data_util.f_statistics(svm, test_features, test_labels, classes)
