import sklearn.metrics

#TODO
def roc_curve(fp, tp):
    return sklearn.metrics.roc_curve()

def auc(fp, tp):
    return sklearn.metrics.auc(fp, tp)

def roc_auc_score():
    return sklearn.metrics.roc_auc_score()
