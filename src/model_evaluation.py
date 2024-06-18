from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision (weighted):", precision_score(y_test, y_pred, average='weighted', zero_division=0))
    print("Recall (weighted):", recall_score(y_test, y_pred, average='weighted', zero_division=0))
    print("F1 Score (weighted):", f1_score(y_test, y_pred, average='weighted'))
