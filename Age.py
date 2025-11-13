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

#Ageの欠損値の処理の方法を変えてみる→具体的にはNameから敬称（Mr,Mrs）などを抽出し、敬称ごとの平均年齢、敬称ごとに年齢幅を決めてランダムに補完する方法
#まずは、Nameの敬称と死亡率の相関のグラフを作成してみる。→Mrs Mr Miss Master 
# 敬称抽出+Rare敬称をまとめた。

train['Title'] = train['Name'].str.extract(r', (\w+)\.', expand=False)
test['Title'] = test['Name'].str.extract(r', (\w+)\.', expand=False)
train["Rare_name"] = train["Title"].replace([
    'Dr', 'Rev', 'Col', 'Major', 'Capt', 'Sir',
    'Lady', 'Countess', 'Jonkheer', 'Don', 'Dona',
    'Mlle', 'Mme', 'Ms'], 
    'Rare' ,inplace=False)
test["Rare_name"] = test["Title"].replace([
    'Dr', 'Rev', 'Col', 'Major', 'Capt', 'Sir',
    'Lady', 'Countess', 'Jonkheer', 'Don', 'Dona',
    'Mlle', 'Mme', 'Ms'], 
    'Rare' ,inplace=False)

#標準偏差での降順　Rare→Master→Miss→Mrs→Mr　なので、Rare Master Missの名前が付いている人は半分が死に半分が活きているとみていい。逆にMrでは標準偏差が小さくなっていてばらつきが比較的ないので、Mr,Mrsは年齢補完に使いやすいと考えられる。
#年齢の階層を決める　Mr 18-50歳　Mrs 18-45歳　Miss 15-50歳　Master 0-14歳　Rare 0-80歳

# 不要な列を削除
train = train.drop(columns=['Ticket', 'Cabin',"PassengerId"])
test = test.drop(columns=['Ticket', 'Cabin',"PassengerId"])

# Ageの欠損値を敬称ごとにランダムに補完する関数
def fill_age(row):
    if pd.isnull(row['Age']):
        if row['Title'] == 'Mr':
            return np.random.randint(18, 51)
        elif row['Title'] == 'Mrs':
            return np.random.randint(18, 46)
        elif row['Title'] == 'Miss':
            return np.random.randint(15, 51)
        elif row['Title'] == 'Master':
            return np.random.randint(0, 15)
        else:  # Rare
            return np.random.randint(0, 81)
    else:
        return row['Age']

train['Age'] = train.apply(fill_age, axis=1)
test['Age'] = test.apply(fill_age, axis=1)


#Title列とName列は不要なので削除
train = train.drop(columns=['Name', 'Title'])
test = test.drop(columns=['Name', 'Title'])

#なぜか欠損値があったので、削除
train = train.dropna()

#Familyの特徴量も追加してみる
train['Family'] = train['SibSp'] + train['Parch']
test['Family'] = test['SibSp'] + test['Parch']
train.drop(['SibSp', 'Parch'], axis=1, inplace=True)
test.drop(['SibSp', 'Parch'], axis=1, inplace=True)

#Fareなしでやってみる
train = train.drop(columns=['Fare'])
test = test.drop(columns=['Fare'])

print(train.isnull().sum())
print(test.isnull().sum())
print(train)

label_encoders = {}
for c in ["Sex", "Embarked", "Rare_name", "Family"]:
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
X_valid_pred = model.predict(X_valid)

# Accuracyスコアで性能を評価
accuracy1 = accuracy_score(y_valid, y_valid_pred)

print("検証データの正解率:", round(accuracy1, 4))

#訓練データでの性能
train_pred = model.predict(X_train)
accuracy2 = accuracy_score(y_train, train_pred)

print("訓練データの正解率:", round(accuracy2, 4))

pred = model.predict(test)
submission = pd.read_csv(PATH + 'gender_submission.csv')
submission['Perished'] = pred
submission.to_csv(PATH + 'submission.csv', index=False)