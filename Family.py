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
print(train.columns)
print(train.info())
print(test.info())

#１Ageの欠損値が多いので、平均値で埋める。
train['Age'].fillna(train['Age'].mean(), inplace=True)
test['Age'].fillna(train['Age'].mean(), inplace=True)

#２Cabinの欠損値が多いので、削除する。
train.drop('Cabin', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)

#３Embarkedの欠損値を最頻値で埋める。
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

#４Fareの欠損値を平均値で埋める。
test['Fare'].fillna(test['Fare'].mean(), inplace=True)

#５SibSPとParchをFamilyという新しい特徴量にまとめる。
train['Family'] = train['SibSp'] + train['Parch']
test['Family'] = test['SibSp'] + test['Parch']
train.drop(['SibSp', 'Parch'], axis=1, inplace=True)
test.drop(['SibSp', 'Parch'], axis=1, inplace=True)

#６name, ticket, passengerIdは予測に不要なので削除する。
train.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
test.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

label_encoders = {}
for c in ["Sex", "Embarked"]:
    label_encoders[c] = LabelEncoder()
    label_encoders[c].fit(pd.concat([train[c], test[c]]).astype(str))
    train[c] = label_encoders[c].transform(train[c].astype(str))
    test[c] = label_encoders[c].transform(test[c].astype(str))

#特徴量と目的変数に分ける
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
#Ageの欠損値を平均値で埋めたが、正直まだ改良の余地はたくさんあると感じている。例えば、タイタニック問題では年齢が低いほど死亡率が低いため、
#名前から年齢を推測して補完する方法や、逆に名前から年齢を補完するやり方なども考えられると思った。名前はMr > Rare > Master > Miss > Mrsの順で死亡率が高く。これを用いて、年齢を補完するやり方もありかもしれない。
#Cabinについては、補完の方法が思いつかないので明日以降は名前から年齢を推測して補完する方法
#Familyの特徴量エンジニアリングについては、あまりぱっとしないなと感じた。なぜなら本質的な分析になっていないような気がするから。
#タイタニック問題の背景にあるのは階級格差による生存率の違いであると考えているため、名前から年齢・階級を予測する手法が本質的なのではないかと思っている。
#11/13