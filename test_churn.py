from churn import generate,train

def test_reproducibility():
    assert generate(100).equals(generate(100))
    assert set(generate(100).churn)=={0,1}

def test_holdout():
    report=train(generate(400))
    assert report['training_rows']==300 and report['test_rows']==100
    assert sum(map(sum,report['holdout']['confusion_matrix']))==100
    assert 0<=report['holdout']['average_precision']<=1
