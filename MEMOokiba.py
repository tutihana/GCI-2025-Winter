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
#敬称ごとの死亡率のplot
train['Title'] = train['Name'].str.extract(r', (\w+)\.', expand=False)

# Titleごとの死亡率（mean）を計算
title_rate = train.groupby("Title")["Perished"].mean().sort_values(ascending=False)

# 降順の並び順をリストとして渡す
order = title_rate.index.tolist()

plt.figure(figsize=(10, 6))

sns.barplot(
    data=train,
    x="Title",
    y="Perished",
    estimator=np.mean,
    errorbar="se",
    capsize=0.2,
    order=order   # ← ここがポイント
)

plt.title("敬称ごとの死亡率")
plt.xlabel("敬称")
plt.ylabel("死亡率")
plt.ylim(0, 1)
plt.show()
