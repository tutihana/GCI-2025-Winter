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

#Pclass＋Sexの死亡率の可視化を行う。
perished_rate = train.groupby(['Pclass', 'Sex'])['Perished'].mean()
print(perished_rate)

train.groupby(['Pclass', 'Sex'])['Perished'].mean().unstack().plot(kind='bar')
plt.ylabel('Perished Rate')
plt.title('Perished Rate by Pclass and Sex')
plt.show()

#女性の方がどのクラスでも生存率が高いことがわかり、特に等級が1号の客席の女性に関しては
#非常に生存率が高いことがわかる。
#また、2つ目の等級の女性に関しても同様に死亡率が低く、１，２等級の女性たちは優遇されていたことが
#数値からよくわかる。

