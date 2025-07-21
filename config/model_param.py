from scipy.stats import randint, uniform

MODEL_PARAMS = {
    'n_estimators': [50,90, 100],
    'max_depth': [3,4,5,6,7],
    'learning_rate': [0.01, 0.05],
    'reg_alpha': [ 0.5 ,1, 2, 5, 10],
    'reg_lambda': [ 0.5,1 ,2, 5, 10, 15],
    'subsample': [0.6, 0.7, 0.8],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'min_child_weight': [  5,7, 10,15],
}

RANDOM_SEARCH_PARAMS = {
    'n_iter': 1000,
    'cv': 5,
    'scoring': 'accuracy',
    'verbose': 2,
    'n_jobs': 3,
    'random_state': 42,
}