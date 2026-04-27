import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

t_actuals= np.array([1,0,0,1,1,0,1,1,0,0]) # targets
Y_predicted =np.array([0.85,0.25,0.3,0.5,0.25,0.1,0.9,0.6,0.5,0.2]) # predicted probabilities
fpr, tpr, threasholds= roc_curve(t_actuals,Y_predicted) # convert predicted into threasholds and sorting the predicted from the highest to lowest
roc_auc =auc(fpr,tpr) #Calculate the area under ROC Curve

#plot
plt.figure(figsize=(12,6))
plt.plot(fpr,tpr, marker="o", label=f"ROC Curve (AUC= {roc_auc:.2f})")
plt.plot([0,1],[0,1], linestyle="--") # a diagonal line( Random classifier) hence them to be good must be above the line
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()


