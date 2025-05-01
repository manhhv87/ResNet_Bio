import matplotlib.pyplot as plt


def plot_history(history):
    plt.plot(history['train'], label='Train Loss')
    plt.plot(history['val'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()