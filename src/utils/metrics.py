import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report



def plot_confusion_matrix(labels, preds, class_names, save_path):
    # Visualizing confusion matrix with percentage
    cm = confusion_matrix(labels, preds)
    cm = cm.astype(float) / (cm.sum(axis= 1, keepdims=True) + 1e-8)

    #Plot
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (%)")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()





def plot_per_class_accuracy(labels, preds, class_names, save_path):
    cm = confusion_matrix(labels, preds,)
    per_class_acc = cm.diagonal() / (cm.sum(axis = 1) + 1e-8)

    #Plots
    plt.figure(figsize=(10,6))
    plt.bar(class_names, per_class_acc)

    plt.xticks(rotation=45)
    plt.xlabel("Classes")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy")

    #Show Values On Bar
    for i , v in enumerate(per_class_acc):
        plt.text(i , v , f'{v:.2f}', ha='center', va='bottom')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()




def plot_training_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

    ax1.plot(epochs, history['train_loss'], label='Train')
    ax1.plot(epochs, history['val_loss'], label='Val')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot(epochs, history['train_acc'], label='Train Acc')
    ax2.plot(epochs, history['val_acc'], label='Val Acc')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    ax3.plot(epochs, history['train_f1'],  label='Train Macro F1')
    ax3.plot(epochs, history['val_f1'],    label='Val Macro F1')
    ax3.plot(epochs, history['train_f1w'], label='Train Weighted F1', linestyle='--')
    ax3.plot(epochs, history['val_f1w'],   label='Val Weighted F1',   linestyle='--')
    ax3.set_title('F1 Score')
    ax3.set_xlabel('Epoch')
    ax3.legend()

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_classification_report(labels, preds, loss, save_path, acc):
    """Saving classification report into .txt file"""
    report = classification_report(labels, preds, digits=3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        f.write(f"Test Loss     : {loss:.4f}\n")
        f.write(f"Test Accuracy : {acc:.4f}\n\n")
        f.write(report)

    return report

