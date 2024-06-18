import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV

def tune_model(features, target):
    X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=3)

    params = {
        'boosting_type': ['gbdt'],  # default
        'num_leaves': [31, 50],
        'max_depth': [-1, 10],
        'n_estimators': [50, 100],  # default = 100
        'learning_rate': [0.05, 0.1],  # default = 0.10
        'min_data_in_leaf': [5, 20],  # default = 20
        'random_state': [30],
        'verbosity': [-1]
    }

    model = lgb.LGBMClassifier()

    # Perform Grid Search
    grid_search = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', cv=3, verbose=1)
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    return best_model
