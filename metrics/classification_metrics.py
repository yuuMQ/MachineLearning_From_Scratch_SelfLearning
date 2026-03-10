import numpy as np

'''
    Precision | Recall | F1-Score | Accuracy
    
                                    Actual
                -----------------------------------------
                |               True        False       |
    Prediction  | True           TP          FP         |
                | False          FN          TN         |
                -----------------------------------------
                
    - Precision: Đo lường độ chính xác của dự đoán (Số lượng dự đoán đúng thực sự đúng):
        -> Precision = TP / (TP + FP) -> Tổng số dự đoán thực sự đúng / Tổng số dự đoán được dự đoán đúng
    - Recall: Đo lường độ bao phủ của dự đoán (Số lượng dự đoán thực sự đúng của dữ liệu thực sự đúng)
        -> Recall = TP / (TP + FN) -> Tổng số dự đoán thực sự đúng / Tổng số dữ liệu đúng
    -  F1-Score: Điều hòa giữa precision và recall:
        -> F1-score = 2 * Precision * Recall / (Precision + Recall) 
    - Accuracy: Độ chính xác tổng quát:
        -> Accuracy = (TP + TN) / (TP + TN + FP + FN) -> Tổng số dự đoán đúng / Toàn bộ dữ liêu
'''

eps = 1e-15 # Avoid division by 0 and log(0)

def precision(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return tp / (tp + fp + eps)

def recall(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp / (tp + fn + eps)

def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy

def f1_score(y_true, y_pred):
    pre = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * pre * rec / (pre + rec + eps)

def binary_cross_entropy_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, eps, 1 - eps) # np.clip(array, min_value, max_value) -> nếu nhỏ hơn min sẽ bằng min, lớn hơn max sẽ bằng max
    bce = - np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return bce

def multiclass_cross_entropy_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    ce = - np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    return ce