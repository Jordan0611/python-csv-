import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pydot
import matplotlib.pyplot as plt
import seaborn as sns

# 檔案名稱
file_vet = '獸醫院.csv'
file_library = '高雄市圖書館.csv'
file_clinic = '診所資料.csv'
file_real_estate = '不動產買賣.csv'

# 讀取CSV檔案
vet_data = pd.read_csv(file_vet)
library_data = pd.read_csv(file_library)
clinic_data = pd.read_csv(file_clinic)
real_estate_data = pd.read_csv(file_real_estate)

# 提取各數據集中的行政區信息並統計數量
vet_data['行政區'] = vet_data['機構地址'].apply(lambda x: x.split('市')[1].split('區')[0] + '區' if '市' in x and '區' in x.split('市')[1] else None)
vet_count = vet_data['行政區'].value_counts().reset_index()
vet_count.columns = ['行政區', '獸醫院數量']

library_data['行政區'] = library_data['地址'].apply(lambda x: x.split('區')[0] + '區' if '區' in x else None)
library_count = library_data['行政區'].value_counts().reset_index()
library_count.columns = ['行政區', '圖書館數量']

clinic_count = clinic_data['行政區'].value_counts().reset_index()
clinic_count.columns = ['行政區', '診所數量']

# 合併數據集到不動產買賣數據
real_estate_data = pd.merge(real_estate_data, vet_count, on='行政區', how='left')
real_estate_data = pd.merge(real_estate_data, library_count, on='行政區', how='left')
real_estate_data = pd.merge(real_estate_data, clinic_count, on='行政區', how='left')

# 填補缺失值
real_estate_data.fillna(0, inplace=True)

# 準備特徵(X)和目標變量(y)
X = real_estate_data[['獸醫院數量', '圖書館數量', '診所數量']]
y = real_estate_data['不動產買賣件數']

# 分割數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 訓練隨機森林回歸模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 評估模型
score = rf.score(X_test, y_test)
print(f"模型 R^2 分數: {score:.2f}")

# 特徵重要性
importances = rf.feature_importances_
feature_names = X.columns
forest_importances = pd.Series(importances, index=feature_names)

plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']

# 繪製特徵重要性圖
fig, ax = plt.subplots()
forest_importances.plot(kind='bar', ax=ax)
ax.set_title("隨機森林模型特徵重要性")
ax.set_ylabel("特徵重要性")
plt.show()

# 可視化決策樹
estimator = rf.estimators_[0]

# 將決策樹導出為DOT格式
export_graphviz(estimator, out_file='tree.dot', 
                feature_names=feature_names,
                rounded=True, proportion=False, 
                precision=2, filled=True)

# 使用pydot將DOT格式轉換為PNG格式
(graph,) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')

# 顯示決策樹圖片
from PIL import Image
img = Image.open('tree.png')
img.show()
