from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from util import data_util

def run_rf(train_feature_file, train_label_file, test_feature_file, test_label_file):
    train_features, train_labels, test_features, test_labels, classes = (
        data_util.feature_pre_processing(train_feature_file, train_label_file, test_feature_file, test_label_file, smote=True))

    param_dist = {
        'n_estimators': randint(100, 300),
        'max_depth': [10, 20],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 5),
    }

    rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    # Perform Randomized Search
    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_dist,
        n_iter=20,
        scoring='f1_weighted',
        cv=2,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    random_search.fit(train_features, train_labels)

    print("Best Params")
    print(random_search.best_params_)

    data_util.f_statistics(random_search.best_estimator_, test_features, test_labels, classes)