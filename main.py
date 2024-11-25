from functions.customCNN import run_custom_cnn
from functions.fine_tune_customCNN import run_fine_tune_customCNN
from functions.grad_cam import run_grad_cam
from functions.pca import run_pca
from functions.pretrained_cnn import run_pretrained_model
from functions.svm import run_svm
from functions.random_forest import run_rf
from functions.tsne import run_tsne
from functions.umap_viz import run_umap

if __name__ == "__main__":
    train_feature_file = "models/features/train_features.npy"
    train_label_file = "models/features/train_labels.npy"
    test_feature_file = "models/features/test_features.npy"
    test_label_file = "models/features/test_labels.npy"
    existing_model_path = "models/custom_model_v1.pth"
    fine_tuned_existing_model_path = "models/fine_tuned_custom_model_v1.pth"
    pre_trained_attention_model_path = "models/pre_trained_v1.pth"
    test_dir = "dataset/chest_XRAY/test"

    args = [train_feature_file, train_label_file, test_feature_file, test_label_file]

    run_custom_cnn(predict=True)
    run_fine_tune_customCNN(existing_model_path)
    run_pretrained_model()
    run_svm(*args)
    run_rf(*args)
    run_pca(*args)
    run_tsne(*args)
    run_umap(*args)
    run_grad_cam(test_dir, pre_trained_attention_model_path)
