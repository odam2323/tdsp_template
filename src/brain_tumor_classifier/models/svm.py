from sklearn.svm import SVC

def train_svm(X, y):
    model = SVC(
        kernel='rbf',
        C=1.0,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X, y)
    return model