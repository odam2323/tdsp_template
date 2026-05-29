import matplotlib.pyplot as plt

def plot_class_distribution(df):
    df["clase"].value_counts().plot(kind="bar")
    plt.title("Distribución de clases")
    plt.xlabel("Clase")
    plt.ylabel("Cantidad")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_train_test_distribution(df):
    pivot = pd.crosstab(df["clase"], df["split"])
    pivot.plot(kind="bar", figsize=(10, 6))
    plt.title("Distribución Train/Test")
    plt.xlabel("Clase")
    plt.ylabel("Cantidad")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
