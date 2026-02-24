from quant.validation.robustness import compute_robustness_score


def test_robustness_score_range():
    train = {"cagr": 0.2, "sharpe": 1.0}
    test = {"cagr": 0.15, "sharpe": 0.8, "max_drawdown": -0.2, "stability_score": 0.7}
    score = compute_robustness_score(train, test)
    assert 0.0 <= score <= 1.0