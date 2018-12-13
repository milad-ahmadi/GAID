import matplotlib.pyplot as plt
from sklearn.metrics import *
import numpy as np

def plot_roc_curve(fpr_ano, tpr_ano, roc_auc_ano):
    plt.figure(0,figsize=(8,8))
    plt.title('ROC Curve')
    plt.plot(fpr_ano, tpr_ano, linewidth=2, label='ROC curve (area = %0.4f)' % roc_auc_ano)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.axis([-0.005, 1, 0, 1.005])
    plt.xticks(np.arange(0,1, 0.05), rotation=90)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='best')
    plt.show()
    #plt.savefig("roc" + ".png", bbox_inches='tight')

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.figure(1,figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "r-", label="Recall")
    plt.axis([-0.005, 1, 0, 1.005])
    plt.xticks(np.arange(0, 1, 0.05), rotation=90)
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')
    plt.show()
    #plt.savefig("precision_recall"+".png", bbox_inches='tight')

def get_auc(y_true, y_score, plot=False):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    if plot:
        plot_roc_curve(fpr, tpr, roc_auc)
    return roc_auc

def get_precision_recall_curve(y_true, y_score):
    p, r, thresholds = precision_recall_curve(y_true, y_score)
    plot_precision_recall_vs_threshold(p, r, thresholds)