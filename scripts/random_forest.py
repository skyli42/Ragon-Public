import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import pickle as pkl
import time
from load_data import load_data_XY
from sklearn.model_selection import train_test_split
#%matplotlib notebook #if running in jupyter notebook
print("Finished imports", flush = True)

with open("../features.pkl", 'rb') as f:
    inds = pkl.load(f)
    
X, Y, labels, strains = load_data_XY("../data/PGT121_Neu_OccAA.csv", inds)

print("Loaded data", flush = True)

X_train, X_test, y_train, y_test = train_test_split(X, Y)

print(labels, flush = True)
print(len(strains), flush = True)

r2s = []
mses = []
for i in range(10): #train 10 times to increase representativeness
    X_train, X_test, y_train, y_test = train_test_split(X, Y)

    #create and train model
    model = RFR(97,verbose = 1).fit(X_train, y_train)
    #predict and evaluate model
    pred = model.predict(X_test)
    print(r2_score(y_test, pred), flush = True)
    print(mean_squared_error(y_test, pred), flush = True)
    r2s.append(r2_score(y_test, pred))
    mses.append(mean_squared_error(y_test, pred))

    # put data into graphs
    # matplotlib
    fig, ax = plt.subplots()
    ax.scatter(pred, y_test)
    plt.title("Performance of Random Forest model")
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("True Values")
    plt.show()

    # #plotly
    # import plotly
    # import plotly.graph_objs as go
    # scatter = go.Scatter(
    #     x = pred,
    #     y = y_test,
    #     mode = 'markers',
    #     marker = dict(
    #         size = 25,
    #         colorscale = 'Viridis',
    #         opacity = 1
    #     )
    # )
    # line = go.Scatter( #draw y = x line for "perfect" prediction
    #     x = [-2.5, 2],
    #     y = [-2.5, 2],
    #     mode = 'lines',
    #     name = 'y = x'
    # )
    # data = [scatter, line]
    # #     print(data)
    # layout = go.Layout(
    #     margin=dict(
    #         l=0,
    #         r=0,
    #         b=0,
    #         t=0
    #     )
    # )
    # fig = go.Figure(data=data, layout=layout)
    # #saves results in html files
    # plotly.offline.plot(fig, filename='../output/rf_results_{}.html'.format(i))

print(r2s, flush = True)
print(mses, flush = True)
print(np.mean(np.array(r2s)))
print(np.mean(np.array(mses)))