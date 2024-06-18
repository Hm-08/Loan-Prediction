import pickle

def deploy_model(model, file_path="../models/loan_model.pkl"):
    with open(file_path, "wb") as f:
        pickle.dump(model, f)
    print(f'Model saved to {file_path}')
