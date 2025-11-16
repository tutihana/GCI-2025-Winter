import numpy as np                                   # 数値計算や配列操作
import pandas as pd                                  # 表形式のデータを扱う

from sklearn.preprocessing import LabelEncoder       # カテゴリ変数を数値に変換
from sklearn.ensemble import RandomForestClassifier  # ランダムフォレスト分類器
from sklearn.model_selection import train_test_split # データの分割
from sklearn.metrics import accuracy_score           # 正解率の計算
from lightgbm import LGBMClassifier 
from catboost import CatBoostClassifier


PATH = "C:\\Users\\piyop\\OneDrive\\デスクトップ\\GCI 2025 Winter\\data\\"
train = pd.read_csv(PATH + 'train.csv') 
test = pd.read_csv(PATH + 'test.csv')

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
rare_prefixes = prefix_counts[prefix_counts < 9].index  # 9件未満をレア扱い

train.loc[train["Ticket_Prefix"].isin(rare_prefixes), "Ticket_Prefix"] = "RARE"
test.loc[test["Ticket_Prefix"].isin(rare_prefixes),  "Ticket_Prefix"] = "RARE"

# 3つに区切って、新たな特徴量を作成
def map_ticket_prefix(x):
    if x == "PC":
        return "HIGH"
    elif x in ["CA.", "W/C", "A/"]:
        return "MID"
    else:
        return "LOW"

train["Prefix_Class"] = train["Ticket_Prefix"].apply(map_ticket_prefix)
test["Prefix_Class"]  = test["Ticket_Prefix"].apply(map_ticket_prefix)

# 敬称抽出+Rare敬称をまとめる
train['Title'] = train['Name'].str.extract(r', (\w+)\.', expand=False)
test['Title'] = test['Name'].str.extract(r', (\w+)\.', expand=False)
train["Rare_name"] = train["Title"].replace([
    'Rev', 'Col', 'Major', 'Capt', 'Sir',
    'Lady', 'Countess', 'Jonkheer', 'Don', 'Dona',
    'Mlle', 'Mme', 'Ms'], 
    'Rare' ,inplace=False)
test["Rare_name"] = test["Title"].replace([
    'Dr', 'Rev', 'Col', 'Major', 'Capt', 'Sir',
    'Lady', 'Countess', 'Jonkheer', 'Don', 'Dona',
    'Mlle', 'Mme', 'Ms'], 
    'Rare' ,inplace=False)
train['Rare_name'] = train['Rare_name'].fillna(train['Rare_name'].mode()[0])
test['Rare_name']  = test['Rare_name'].fillna(train['Rare_name'].mode()[0])

# Titleごとの中央値で埋める
train['Age'] = train['Age'].fillna(train.groupby('Title')['Age'].transform('median'))
test['Age']  = test['Age'].fillna(test.groupby('Title')['Age'].transform('median'))

# 不要な列を削除+Title列とName列は不要なので削除
train = train.drop(columns=['Ticket', 'Cabin',"PassengerId","Name", 'Title',"Sex","Ticket_Prefix"])
test = test.drop(columns=['Ticket', 'Cabin',"PassengerId","Name", 'Title',"Sex","Ticket_Prefix"])

# Embarked を最頻値で埋める
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
test['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

# Fareの欠損値を中央値で補完
train['Fare'].fillna(train['Fare'].median(), inplace=True)
test['Fare'].fillna(train['Fare'].median(), inplace=True)

# 基本の Family（兄弟＋親子）
train['Family'] = train['SibSp'] + train['Parch']
test['Family'] = test['SibSp'] + test['Parch']

# 自分を含めた家族人数
train['FamilySize'] = train['Family'] + 1
test['FamilySize']  = test['Family'] + 1

# 一人きりかどうか
train['IsAlone'] = (train['FamilySize'] == 1).astype(int)
test['IsAlone']  = (test['FamilySize'] == 1).astype(int)

# 家族人数をカテゴリ化
def cat_family(size):
    if size == 1:
        return "Alone"
    elif size <= 4:
        return "Small"
    else:
        return "Large"

train['FamilyCategory'] = train['FamilySize'].apply(cat_family)
test['FamilyCategory']  = test['FamilySize'].apply(cat_family)

# 家族構成のパターン（兄弟だけ／親子だけ）
train['SiblingsOnly'] = ((train['SibSp'] > 0) & (train['Parch'] == 0)).astype(int)
test['SiblingsOnly']  = ((test['SibSp'] > 0) & (test['Parch'] == 0)).astype(int)

train['ParentsOnly'] = ((train['Parch'] > 0) & (train['SibSp'] == 0)).astype(int)
test['ParentsOnly']  = ((test['Parch'] > 0) & (test['SibSp'] == 0)).astype(int)

# もう SibSp / Parch は生で使わないので削除
train.drop(['SibSp', 'Parch'], axis=1, inplace=True)
test.drop(['SibSp', 'Parch'], axis=1, inplace=True)

# 特徴量と目的変数に分ける
X = train.drop(columns=["Perished"])  # 予測に使う説明変数
y = train["Perished"]                 # 予測したい目的変数

# データを7:3に分割（訓練データ70%、検証データ30%）
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.3, random_state=46, stratify=y
)

# ==============================
# モデルの設定（CatBoost：提出スコア重視）
# ==============================

# ここは「カテゴリとして扱ってほしいカラム」
cat_features = ['Embarked', 'Rare_name', 'Prefix_Class','FamilyCategory']

model = CatBoostClassifier(
    iterations=500,          # 上限回数をまず減らす（300～500で十分なこと多い）
    learning_rate=0.05,      # 少し上げて学習を早く進める
    depth=6,
    l2_leaf_reg=4.0,
    loss_function='Logloss',
    random_state=2025,
    verbose=False,
    # ここから高速化のキモ
    od_type='Iter',          # early stoppingの方法
    od_wait=50,              # 50イテレーション改善しなかったら打ち切る
    thread_count=-1          # CPU全部使う
)

# モデルを訓練データで学習
model.fit(
    X_train,
    y_train,
    cat_features=cat_features
)

# ==============================
# 検証データで予測（クラスラベルを出力）
# ==============================

y_valid_pred = model.predict(X_valid)

# Accuracyスコアで性能を評価
accuracy1 = accuracy_score(y_valid, y_valid_pred)
print("検証データの正解率:", round(accuracy1, 4))

# ==============================
# 訓練データでの性能（過学習チェック）
# ==============================

train_pred = model.predict(X_train)
accuracy2 = accuracy_score(y_train, train_pred)
print("訓練データの正解率:", round(accuracy2, 4))

# ==============================
# テストデータで予測して提出ファイル作成
# ==============================

pred = model.predict(test)

submission = pd.read_csv(PATH + 'gender_submission.csv')
submission['Perished'] = pred
submission.to_csv(PATH + 'submission.csv', index=False)
print("submission.csv を出力しました")
print(train["Rare_name"].value_counts())