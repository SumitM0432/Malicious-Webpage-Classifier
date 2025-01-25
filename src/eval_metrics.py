import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [8, 6]

def metric_scores(y_true, y_pred):

    cm = metrics.confusion_matrix(y_true, y_pred)

    ax = sns.heatmap(cm, annot = True, fmt="d", cmap = 'YlGnBu')
    ax.set(title = "Confusion Matrix", xlabel = 'Predicted Labels', ylabel = 'True Labels')

    cls_report = metrics.classification_report(y_true, y_pred)
    
    print ("")
    print (f"Accuracy : {metrics.accuracy_score(y_true, y_pred)*100 : .3f} %") 
    print ("")
    print ("Classification Report : ")
    print (cls_report)

    plt.show()


