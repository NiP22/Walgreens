import sys
from Walgreens.Preprocessing import preprocessing
from Walgreens.model import predict_52
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 3)
plt.rcParams['font.family'] = 'sans-serif'

pd.set_option('display.expand_frame_repr', False)
data = pd.read_csv('data.csv', sep='|')
future = pd.read_csv('test_future.csv', sep='|')
future = future.drop('Unnamed: 0', 1)
all_pln = data[['PLN']]
all_pln = all_pln.drop_duplicates()
count = 0
size = all_pln.shape[0]
print(size)


for pln in all_pln['PLN']:
    count += 1
    pred_item = future[future['PLN'] == pln]
    item = data[data['PLN'] == pln]
    #future = future[future['PLN'] != pln]
    print(pln)
    print(count/size)
    #future.to_csv('test_future.csv', sep='|')
    pred_item = preprocessing(1, pred_item)
    item = preprocessing(0, item)
    item = item[['Revenue', 'Promo', 'Year', 'Date']]
    test = pred_item.drop('PLN', 1)
    item.index = np.arange(0, item.shape[0])
    if not pred_item.empty:
        prediction = predict_52(item, np.array(pred_item['Promo']), pred_item.shape[0] - 1)
        if prediction.shape[0]:
            pred_item['ACTUAL'] = prediction
            pred_item.index = pred_item['WEEK']
            pred_item = pred_item.drop('WEEK', 1)
            with open('test.csv', 'a') as f:
                pred_item[['PLN', "ACTUAL"]].to_csv(f, sep='|', header=False)
            f.close()

#print(prediction)
'''
with open('good_future.csv', 'a') as f:
    df_head.to_csv(f, sep='|')
    df.to_csv(f, header=False, sep='|')
#df[df['PLN'] == 40000914559].to_csv('Item.csv', sep='|')
#df[df['PLN'] == 40000581631].to_csv('Item.csv', sep='|')
#df[df['WEEK'] == 1161218].sample(11860).to_csv('NewYear.csv', sep='|')
'''
