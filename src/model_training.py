import lightgbm as lgb
from sklearn.model_selection import train_test_split

def train_model(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=3)

    params = {
        'boosting_type': 'gbdt',  # default
        'num_leaves': 31,  # default
        'max_depth':-1,
        'n_estimators': 50,  # default = 100
        'learning_rate': 0.05,  # default = 0.10
        'min_data_in_leaf': 5,  # default = 20
        'random_state': 30,
        'verbosity': -1
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    return model, X_test, y_test
