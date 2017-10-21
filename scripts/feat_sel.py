from load_data import load_data_XY
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LassoCV, Lasso, ElasticNet
import pickle as pkl
import sys
if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Not enough arguments")
    print(sys.argv, flush = True)
    X, y, labels, names = load_data_XY("../data/PGT121_Neu_OccAA.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print("loaded data", flush = True)
    if sys.argv[1] == '0' or sys.argv[1] == '2':
        #finding optimal alpha for selection
        lasso = LassoCV(cv = 10, verbose = 1).fit(X, y)
        print(lasso.alpha_, flush = True)
        #testing lasso
        lasso_pred = Lasso(alpha = lasso.alpha_).fit(X_train, y_train).predict(X_test)
        print("lasso r^2 score: %0.3f" %r2_score(y_test, lasso_pred), flush = True)
    if sys.argv[1] == '1' or sys.argv[1] == '2':
        if sys.argv[1] =='2':
            alpha = lasso.alpha_
        else:
            alpha = 0.036573321313
        #finding optimal l1_ratio using alpha from lasso
        params = [{
            'alpha': [alpha],
            'l1_ratio':np.linspace(0.0, 1,101) #narrowed down to 0.9 to 1.0 (originally 0.0 to 1.0)
        }]
        print(np.linspace(0.0, 1, 101), flush = True)
        reg = GridSearchCV(ElasticNet(), params, cv=15, n_jobs = -1, verbose = 1)
        reg.fit(X_train, y_train)
        print("Finished search for l1", flush = True)
        best_params = reg.best_params_
        opt_l1 = best_params["l1_ratio"]
        print(opt_l1, flush= True)

        enet = Pipeline([
            ('feature_selection', SelectFromModel(ElasticNet(alpha = alpha, l1_ratio = opt_l1))),
            ('regression', ElasticNet(alpha = alpha, l1_ratio = opt_l1))
        ])
        enet.fit(X, y)
        features = enet.steps[0][1].get_support(indices = True)
        print(features, flush = True)
        print(labels[features], flush = True)
        with open("../features.pkl", 'wb') as f:
            pkl.dump(features, f)