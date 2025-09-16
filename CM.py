import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from Alogrithm import y_test, y_pred_test

cm3 = confusion_matrix(y_test, y_pred_test)
print(cm3)
# Generate confusion matrix
class_names = ['Real', 'Fake']  # Change as per your class labels

plt.figure(figsize=(8,6))
sns.heatmap(cm3, annot=True, fmt='g', cmap='Blues', linewidths=1, linecolor='black', 
            xticklabels=class_names, yticklabels=class_names, cbar=True)

plt.title('Confusion Matrix for Fake News Detection', fontsize=16)
plt.xlabel('Predicted Labels', fontsize=12)
plt.ylabel('True Labels', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

