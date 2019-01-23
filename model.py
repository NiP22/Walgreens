import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AR
from  statsmodels.tools.sm_exceptions import MissingDataError
import random

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 3)
plt.rcParams['font.family'] = 'sans-serif'
'''
pd.set_option('display.expand_frame_repr', False)
df = pd.read_csv('Good_item.csv', sep='|')
df = df.drop("Unnamed: 0", 1)
raw_data = df[['Revenue', 'Promo', 'Year', 'Date']]
test = pd.read_csv('Good_pred_item.csv', sep='|')
test.drop('Unnamed: 0', 1)
test.drop('PLN', 1)
raw_data.to_csv('test.csv', sep='|')
'''
def delete_less_then_one(arr):
    for i in range(arr.size):
        if arr[i] < 1:
            arr[i] = 1
    return arr


def fill_promo(data):
    if data[data['Promo'] == 0].empty:
        return data
    indexes = data[data['Promo'] == 1].index
    for i in indexes:
        i_left = i - 1
        i_right = i + 1
        while i_left in indexes:
            i_left -= 1
        if i_left == -1:
            left_val = data[data['Promo'] == 0].iloc[0, 0]
        else:
            left_val = data.iloc[i_left, 0]
        while i_right in indexes:
            i_right += 1
        if i_right == data.shape[0]:
            right_val = data[data['Promo'] == 0].iloc[data[data['Promo'] == 0].shape[0] - 2, 0]
        else:
            right_val = data.iloc[i_right, 0]
        data.iloc[i, 0] = (right_val + left_val) / 2
    return data


def model(y_train, pln, weeks_to_predict=52):
    ar_model = AR(y_train)
    try:
        model_fit = ar_model.fit(maxlag=30, maxiter=100)
        predictions = model_fit.predict(start=len(y_train), end=len(y_train) + weeks_to_predict, dynamic=True)
    except ValueError:
        f = open("spliced_data.txt", 'a')
        f.write(str(pln) + "\n")
        return np.array([])
    for actual in predictions:
        if actual < 0:
            try:
                print("it's log time")
                ar_model = AR(np.log(delete_less_then_one(y_train)))
                model_fit = ar_model.fit(maxlag=18, maxiter=100)
                predictions = model_fit.predict(start=len(y_train), end=len(y_train) + weeks_to_predict, dynamic=True)
                return np.exp(predictions)
            except MissingDataError:
                f = open("bad_model.txt", 'a')
                f.write(str(pln) + "\n")
                return predictions
    return predictions


def promo_coefficient(data, raw_data):
    sum_pred = 1
    sum_true = 1
    y_pred = np.array(data['Revenue'])
    y_true = np.array(raw_data['Revenue'])
    indexes = raw_data[raw_data['Promo'] == 1].index
    for i in indexes:
        sum_pred += y_pred[i]
        sum_true += y_true[i]
    return sum_true/sum_pred


def predict_52(raw_data, promo, weeks_to_predict=52):
    if raw_data[raw_data['Year'] == 15].empty:
        print("history starts not from 15")
        return np.array([])
    data = fill_promo(raw_data.copy())
    y_train = np.array(data['Revenue'])
    pred = model(y_train, raw_data['PLN'].iloc[0],  weeks_to_predict)
    coef = promo_coefficient(data, raw_data)
    if coef > 4:
        f = open("promo_dependent.txt", 'a')
        f.write(str(raw_data['PLN'].iloc[0]) + "\n")
    for i in range(promo.shape[0]):
        if promo[i]:
            pred[i] *= coef
    return pred


'''
pred = predict_52(raw_data, np.array(test['Promo']))
x = np.arange(test.index[0], test.index[0] + 53)
print(pred)
plt.plot(x, pred)
plt.show()
plt.plot(raw_data['Revenue'])
'''

'''
true_val = np.array(test['Revenue'])
n = true_val.shape[0]
pred = pred[0:n]
mse = (pred - true_val)**2
print(np.sum(mse)/n)
'''

'''
#cont_x = np.arange(0, 200)
#Test = np.array([np.ones(cont_x.shape[0]), cont_x]).T
#cont_prediction = np.dot(w, Test.T)
#plt.plot(cont_x, np.concatenate([Y_train, np.zeros(cont_x.shape[0] - Y_train.shape[0])]))
#plt.plot(cont_x, cont_prediction, label='prediction')
#plt.show()


costs = []
w = np.random.randn(4)/np.sqrt(4)
learn_rate = 10**-15
lamb = 5
for t in range(10000):
    Y_pred = X.dot(w)
    delta = Y_pred - Y_train
    w = w - learn_rate*(X.T.dot(delta) + lamb*np.sign(w))
    print(w)
    mse = delta.dot(delta)/n
    costs.append(mse)
print(costs)
Cont_x = np.arange(0, 130)
Test = np.array([np.ones(Cont_x.shape[0]), Cont_x, Cont_x**2, Cont_x**3]).T

prediction = np.dot(w, Test.T)

plt.plot(Cont_x, np.concatenate([Y_train, np.zeros(Cont_x.shape[0] - Y_train.shape[0])]))
plt.plot(Cont_x, prediction, label='prediction')
plt.show()
'''
