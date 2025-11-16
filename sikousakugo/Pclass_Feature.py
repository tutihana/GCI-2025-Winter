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

#Pclass+Sexの新しい特徴量を追加する
train["Pclass_Sex_Perished_rate"] = train.groupby(["Pclass", "Sex"])["Perished"].transform("mean")
pclass_sex_rate = train.groupby(["Pclass", "Sex"])["Perished"].mean()
test["Pclass_Sex_Perished_rate"] = test.set_index(["Pclass", "Sex"]).index.map(pclass_sex_rate)

# 敬称抽出+Rare敬称をまとめる
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

# Titleごとの中央値で埋める例
train['Age'] = train['Age'].fillna(train.groupby('Title')['Age'].transform('median'))
test['Age']  = test['Age'].fillna(test.groupby('Title')['Age'].transform('median'))
print(train.isnull().sum())
print(test.isnull().sum())
print(train)
# 不要な列を削除+Title列とName列は不要なので削除
train = train.drop(columns=['Ticket', 'Cabin',"PassengerId","Name", 'Title',"Sex"])
test = test.drop(columns=['Ticket', 'Cabin',"PassengerId","Name", 'Title',"Sex"])

# Embarked を最頻値で埋める
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
test['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

# Fareの欠損値を中央値で補完
train['Fare'].fillna(train['Fare'].median(), inplace=True)
test['Fare'].fillna(train['Fare'].median(), inplace=True)

#Familyの特徴量も追加してみる
train['Family'] = train['SibSp'] + train['Parch']
test['Family'] = test['SibSp'] + test['Parch']
train.drop(['SibSp', 'Parch'], axis=1, inplace=True)
test.drop(['SibSp', 'Parch'], axis=1, inplace=True)

train['Rare_name'] = train['Rare_name'].fillna(train['Rare_name'].mode()[0])
test['Rare_name']  = test['Rare_name'].fillna(train['Rare_name'].mode()[0])

print(train)
print(train.isnull().sum())
label_encoders = {}
for c in ["Embarked", "Rare_name"]:
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