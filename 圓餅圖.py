import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# 檔案路徑
file_vet = '獸醫院.csv'
file_library = '高雄市圖書館.csv'
file_clinic = '診所資料.csv'

# 讀取CSV檔案
vet_data = pd.read_csv(file_vet)
library_data = pd.read_csv(file_library)
clinic_data = pd.read_csv(file_clinic)

# 提取獸醫院數據中的行政區信息
vet_data['行政區'] = vet_data['機構地址'].apply(lambda x: x.split('市')[1].split('區')[0] + '區')
vet_count = vet_data['行政區'].value_counts()

# 提取圖書館數據中的行政區信息
library_data['行政區'] = library_data['地址'].apply(lambda x: x.split('區')[0] + '區')
library_count = library_data['行政區'].value_counts()

# 提取診所數據中的行政區信息
clinic_count = clinic_data['行政區'].value_counts()

# 設置支持中文的字體
plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']

# 繪製圓餅圖的函數
def plot_pie_chart(data, title):
    plt.figure(figsize=(10, 10))
    plt.pie(data, labels=data.index, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    plt.axis('equal')  # 確保餅圖是圓形
    plt.show()

# 繪製獸醫院圓餅圖
plot_pie_chart(vet_count, '各行政區獸醫院數量分佈')

# 繪製圖書館圓餅圖
plot_pie_chart(library_count, '各行政區圖書館數量分佈')

# 繪製診所圓餅圖
plot_pie_chart(clinic_count, '各行政區診所數量分佈')
