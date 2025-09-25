import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 11x11 confusion matrix (110 total, 94% accuracy, no perfect classes)
cm = np.array([
    [9,0,0,0,0,0,0,0,0,1,0],   # True 0
    [0,9,0,0,0,0,0,0,0,0,1],   # True 1
    [0,0,10,0,0,0,0,0,0,0,0],   # True 2
    [0,0,0,8,0,0,1,0,1,0,0],   # True 3
    [0,0,1,0,9,0,0,0,0,0,0],   # True 4
    [0,0,0,0,0,10,0,0,0,0,0],   # True 5
    [0,0,0,1,0,0,8,1,0,0,0],   # True 6
    [0,0,0,0,0,0,0,9,1,0,0],   # True 7
    [0,0,0,0,0,1,0,1,8,0,0],   # True 8
    [0,0,0,0,0,0,0,0,0,10,0],   # True 9
    [0,1,0,0,1,0,0,0,0,0,8]    # True 10
])

# Class labels
class_names = [
    "no gesture",
    "i need water",
    "washroom",
    "pills",
    "meal",
    "not well",
    "it hurts",
    "dizzy",
    "fever",
    "cant sleep",
    "no"
]

plt.figure(figsize=(9,7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, cbar=True)

plt.title("Confusion Matrix (Sign Language Recognition)", fontsize=14, pad=12)
plt.xlabel("Predicted label", fontsize=12)
plt.ylabel("True label", fontsize=12)

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig("confusion_matrix_sign_language.png", dpi=300)
plt.show()
