import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 3)
plt.rcParams['font.family'] = 'sans-serif'

pd.set_option('display.expand_frame_repr', False)
df = pd.read_csv('data.csv', sep='|')
#df.iloc[0:1000000].to_csv('first_part.csv', sep='|')


#print(df['WEEK'].iloc[df['WEEK'].idxmax()])
#print(df['WEEK'].iloc[df['WEEK'].idxmin()])
#print(df['WEEK'].idxmin())

#df[df['PLN'] == 40000914559].to_csv('Item.csv', sep='|')
df[df['PLN'] == 40000581631].to_csv('Item.csv', sep='|')
#df[df['WEEK'] == 1161218].sample(11860).to_csv('NewYear.csv', sep='|')

'''
plotish = df[['OPSTUDY_LABEL', 'ACTUAL']]
print(type(plotish.head()))
print('________________________________________')
plotish.groupby('OPSTUDY_LABEL').sum().head(20).plot(kind='bar')
plt.show()
print('________________________________________')

tmp = df[['WEEK', 'ACTUAL']]
print(tmp)
print(tmp.groupby('WEEK').sum())
'''
#print(df.shape)
#print(df.info())
#print(df['WEEK'].head())
#df['WEEK'] = df['WEEK'].astype('int32')
#df = df.sort_values(by='WEEK', ascending=False)
#print(df.head())14187933
