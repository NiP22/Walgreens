import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 3)
plt.rcParams['font.family'] = 'sans-serif'

pd.set_option('display.expand_frame_repr', False)
df = pd.read_csv('GoodItem.csv', sep='|')
df = df.drop("Unnamed: 0", 1)
data = df[['Revenue', 'Promo', 'Year', 'Date']]
data = data[(data['Year'] == 16) | (data['Year'] == 15)]
#data['Revenue'].plot()
#plt.show()

Y_train = np.array(data['Revenue'])
X_train = np.array(data.index)
X = np.array([np.ones(X_train.shape[0]), X_train, X_train**2]).T
w = np.dot(
            np.dot(
                    np.linalg.inv(
                                np.dot(X.T, X)
                    )
                    , X.T
            )
            , Y_train
)
Cont_x = np.arange(0, 130)
Test = np.array([np.ones(Cont_x.shape[0]), Cont_x, Cont_x**2]).T
prediction = np.dot(w, Test.T)
print(np.concatenate([Y_train, np.zeros(23)]))
plt.plot(Cont_x, np.concatenate([Y_train, np.zeros(Cont_x.shape[0] - Y_train.shape[0])]))
plt.plot(Cont_x, prediction, label='prediction')
plt.show()
test_pred = np.dot(w, Test.T)
print(test_pred)
print(prediction)
