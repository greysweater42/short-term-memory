from collections import namedtuple

from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve,
)


def calculate_metrics(y_t, y_t_hat, y_v, y_v_hat, y_v_hat_proba):
    fpr, tpr, _ = roc_curve(y_v, y_v_hat_proba)
    cm = confusion_matrix(y_v, y_v_hat)
    Metrics = namedtuple(
        "Metrics", "acc_val acc_train precision recall specificity auc"
    )
    return Metrics(
        acc_val=accuracy_score(y_v, y_v_hat) * 100,
        acc_train=accuracy_score(y_t, y_t_hat) * 100,
        precision=precision_score(y_v, y_v_hat),
        recall=recall_score(y_v, y_v_hat),
        specificity=cm[0][0] / sum(cm[0]),
        auc=auc(fpr, tpr),
    )
