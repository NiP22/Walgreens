import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 3)
plt.rcParams['font.family'] = 'sans-serif'


pd.set_option('display.expand_frame_repr', False)
df = pd.read_csv('GoodItem.csv', sep='|')
df = df.drop("Unnamed: 0", 1)
raw_data = df[['Revenue', 'Promo', 'Year', 'Date']]
test = raw_data[raw_data['Year'] == 17]
raw_data = raw_data[(raw_data['Year'] == 16) | (raw_data['Year'] == 15)]


def fill_promo(data):
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
            right_val = data[data['Promo'] == 0].iloc[data[data['Promo'] == 0].shape[0] - 1, 0]
        else:
            right_val = data.iloc[i_right, 0]
        data.iloc[i, 0] = (right_val + left_val) / 2
    return data


def model(x_train, y_train):
    X = np.array([np.ones(x_train.shape[0]), x_train]).T
    w = np.dot(
            np.dot(
                np.linalg.inv(
                    np.dot(X.T, X)
                )
                , X.T
            )
            , y_train
    )
    return w


def promo_coefficient(y_pred, data):
    sum_pred = 0
    sum_true = 0
    y_true = data['Revenue']
    indexes = data[data['Promo'] == 1].index
    for i in indexes:
        sum_pred += y_pred[i]
        sum_true += y_true[i]
    return sum_true/sum_pred


def predict_52(raw_data, promo):
    data = fill_promo(raw_data.copy())
    y_train = np.array(data['Revenue'])
    x_train = np.array(data.index)
    w = model(x_train, y_train)
    train_pred = np.dot(w, np.array([np.ones(x_train.shape[0]), x_train]))
    #plt.plot(x_train, raw_data['Revenue'])
    #plt.plot(x_train, pred)
    coef = promo_coefficient(train_pred, raw_data)
    #plt.show()
    cont_x = np.arange(0, raw_data.shape[0] + promo.shape[0])
    ans_x = np.array([np.ones(cont_x.shape[0]), cont_x]).T
    pred = np.dot(w, ans_x.T)
    ans = pred[raw_data.shape[0]:]
    for i in range(promo.shape[0]):
        if promo[i]:
            ans[i] *= coef
    return ans


pred = predict_52(raw_data, np.array(test['Promo']))
plt.plot(test.index, pred)
plt.plot(test.index, test['Revenue'])
plt.show()
print(test.index)


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
