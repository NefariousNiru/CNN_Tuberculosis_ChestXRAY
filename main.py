from functions.customCNN import run_custom_cnn
from functions.svm import run_svm
from functions.random_forest import run_rf
if __name__ == "__main__":
    train_feature_file = "models/features/train_features.npy"
    train_label_file = "models/features/train_labels.npy"
    test_feature_file = "models/features/test_features.npy"
    test_label_file = "models/features/test_labels.npy"

    args = [train_feature_file, train_label_file, test_feature_file, test_label_file]

    # run_custom_cnn()
    # run_svm(*args)
    run_rf(*args)

