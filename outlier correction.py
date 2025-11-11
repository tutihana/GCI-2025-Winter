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

print(train['Fare'].describe())#Fareに関する基本統計量を表示したところ、Maxの外れ値をみつけた。データの前処理の段階でこれを削除したい。削除することによってなにか生まれる気はしないが、平均値・中央値などに影響がありそうなので一応除去しておく。
print(test['Fare'].describe())

IQR = train['Fare'].quantile(0.75) - train['Fare'].quantile(0.25)
lower_bound = train['Fare'].quantile(0.25) - 3 * IQR
upper_bound = train['Fare'].quantile(0.75) + 3 * IQR
train= train[train['Fare'] < upper_bound] #Fareの外れ値を削除
#test= test[test['Fare'] < upper_bound]   #Fareの外れ値を削除 数値は、四分位範囲（IQR）3倍ルールに基づいて計算されたもの

train = train.drop(columns=['Name', 'Ticket', 'Cabin',"PassengerId"])
test = test.drop(columns=['Name', 'Ticket', 'Cabin',"PassengerId"])

# 最頻値で補完する対象の列
cols_to_fill = ["Age", "Embarked"]

# train の最頻値で train/test 両方を補完
for col in cols_to_fill:
    mode_value = train[col].mode()[0]
    train[col] = train[col].fillna(mode_value)
    test[col] = test[col].fillna(mode_value)

label_encoders = {}
for c in ["Sex", "Embarked"]:
    label_encoders[c] = LabelEncoder()
    label_encoders[c].fit(pd.concat([train[c], test[c]]).astype(str))
    train[c] = label_encoders[c].transform(train[c].astype(str))
    test[c] = label_encoders[c].transform(test[c].astype(str))

# 特徴量と目的変数に分ける
X = train.drop(columns=["Perished"])  # 予測に使う説明変数
y = train["Perished"]                 # 予測したい目的変数

# データを7:3に分割（訓練データ70%、検証データ30%）
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.3, random_state=46, stratify=y
)

# モデルの設定（ランダムフォレスト）
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=2025
)

# モデルを訓練データで学習
model.fit(X_train, y_train)

# 検証データで予測（クラスラベルを出力）
y_valid_pred = model.predict(X_valid)

# Accuracyスコアで性能を評価
accuracy = accuracy_score(y_valid, y_valid_pred)
print("検証データの正解率:", round(accuracy, 4))

pred = model.predict(test)
submission = pd.read_csv(PATH + 'gender_submission.csv')
submission['Perished'] = pred
submission.to_csv(PATH + 'submission.csv', index=False)
#所感
#すごく難しかった。しかし、あまり苦にならなかったので、ここから特徴量エンジニアリングを行って精度向上に努めたい。
#今日やったこととしてはFareの外れ値を削除し、よりばらつきのない形に成形した。また、basic版のモデル構築・モデル評価での手法を取ったので、明日以降は理解を深めつつadvanced版の方のやり方も取り入れたい。
#demoのadvanced版の方を提出したときは、Public score 0.775だったので、今回自作したモデルがどれくらいのスコアになるのかがすごく気になる。
#スコアにとらわれず、自分の思うままにコンペを行って良い結果や経験を積めたら非常に嬉しい。
#Githubリポジトリ作成時に少しミスしちゃったので、ちょっとごっちゃになってるが、記録用として挙げているだけなので今はこのままでいいやという感じ。
