from src.eval.metrics import average_precision_score_binary, roc_auc_score_binary


def test_auc_roc_binary_basic():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]
    auc = roc_auc_score_binary(y_true, y_score)
    assert auc is not None
    assert 0.99 <= auc <= 1.0


def test_average_precision_binary_basic():
    y_true = [0, 1, 0, 1]
    y_score = [0.1, 0.9, 0.2, 0.8]
    ap = average_precision_score_binary(y_true, y_score)
    assert ap is not None
    assert 0.99 <= ap <= 1.0


def test_auc_ap_none_when_single_class():
    y_true = [0, 0, 0]
    y_score = [0.1, 0.2, 0.3]
    assert roc_auc_score_binary(y_true, y_score) is None
    assert average_precision_score_binary(y_true, y_score) is None