import japanize_matplotlib                           # グラフに日本語を表示
import numpy as np                                   # 数値計算や配列操作
import pandas as pd                                  # 表形式のデータを扱う
import matplotlib.pyplot as plt                      # 基本的なグラフ描画
import seaborn as sns                                # きれいで便利な統計グラフ
import missingno as msno                             # 欠損値の可視化

from sklearn.preprocessing import LabelEncoder       # カテゴリ変数を数値に変換
from sklearn.ensemble import RandomForestClassifier  # ランダムフォレスト分類器
from sklearn.model_selection import train_test_split # データの分割
from sklearn.metrics import accuracy_score           # 正解率の計算

PATH = "C:\\Users\\piyop\\OneDrive\\デスクトップ\\GCI 2025 Winter\\data\\"
train = pd.read_csv(PATH + 'train.csv') 
test = pd.read_csv(PATH + 'test.csv')

print(train.info())
print(train)

train["Ticket_Prefix"] = train["Ticket"].str.extract(r', (\w+)\.', expand=False)

# まず文字列にしておく
train["Ticket"] = train["Ticket"].astype(str)
test["Ticket"]  = test["Ticket"].astype(str)

# Ticket Prefix（文字部分）を抜き出し
train["Ticket_Prefix"] = train["Ticket"].str.extract(r'([A-Za-z./]+)', expand=False)
test["Ticket_Prefix"]  = test["Ticket"].str.extract(r'([A-Za-z./]+)', expand=False)

# 完全に数字だけの人は NaN になるので、'NONE' などで埋める
train["Ticket_Prefix"] = train["Ticket_Prefix"].fillna("NONE")
test["Ticket_Prefix"]  = test["Ticket_Prefix"].fillna("NONE")

# レアな Prefix をまとめる
prefix_counts = train["Ticket_Prefix"].value_counts()
rare_prefixes = prefix_counts[prefix_counts < 9].index  # 9件未満をレア扱いとか

train.loc[train["Ticket_Prefix"].isin(rare_prefixes), "Ticket_Prefix"] = "RARE"
test.loc[test["Ticket_Prefix"].isin(rare_prefixes),  "Ticket_Prefix"] = "RARE"

train["Ticket_Number"] = train["Ticket"].str.extract(r'(\d+)', expand=False)
test["Ticket_Number"]  = test["Ticket"].str.extract(r'(\d+)', expand=False)

train["Ticket_Number"] = train["Ticket_Number"].astype(float)
test["Ticket_Number"]  = test["Ticket_Number"].astype(float)

mean_num = train["Ticket_Number"].mean()
train["Ticket_Number"].fillna(mean_num, inplace=True)
test["Ticket_Number"].fillna(mean_num, inplace=True)

print(train["Ticket_Prefix"].unique())
print(train["Ticket_Prefix"].value_counts())

# Ticket_Prefixごとの平均Fareを集計
Ticket_soukan = (
    train.groupby("Ticket_Prefix")["Fare"]
    .mean()
    .sort_values()
)

# 棒グラフで可視化
plt.figure(figsize=(10, 5))
Ticket_soukan.plot(kind='bar')

plt.ylabel('平均運賃（Fare）')
plt.title('チケット接頭辞ごとの平均運賃')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
