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
from lightgbm import LGBMClassifier 
from catboost import CatBoostClassifier


PATH = "C:\\Users\\piyop\\OneDrive\\デスクトップ\\GCI 2025 Winter\\data\\"
train = pd.read_csv(PATH + 'train.csv') 
test = pd.read_csv(PATH + 'test.csv')

print(train.info())
print(train)

# ==========================
# Ticket 関連の特徴量
# ==========================

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

# ==========================
# IsWomanOrChild（女性 or 子ども）
# ==========================
train['IsWomanOrChild'] = ((train['Sex'] == 'female') | (train['Age'] < 14)).astype(int)
test['IsWomanOrChild']  = ((test['Sex'] == 'female') | (test['Age'] < 14)).astype(int)

# ==========================
# 敬称抽出 + Rare 敬称
# ==========================
train['Title'] = train['Name'].str.extract(r', (\w+)\.', expand=False)
test['Title']  = test['Name'].str.extract(r', (\w+)\.', expand=False)

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

train['Rare_name'] = train['Rare_name'].fillna(train['Rare_name'].mode()[0])
test['Rare_name']  = test['Rare_name'].fillna(train['Rare_name'].mode()[0])

# Titleごとの中央値で Age を補完
train['Age'] = train['Age'].fillna(train.groupby('Title')['Age'].transform('median'))
test['Age']  = test['Age'].fillna(test.groupby('Title')['Age'].transform('median'))

# 不要な列を削除（Title, Name, Ticketなど）
train = train.drop(columns=['Ticket', 'Cabin', 'PassengerId', 'Name', 'Title', 'Ticket_Prefix'])
test  = test.drop(columns=['Ticket', 'Cabin', 'PassengerId', 'Name', 'Title', 'Ticket_Prefix'])

# ==========================
# Embarked, Fare の欠損補完
# ==========================
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
test['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

train['Fare'].fillna(train['Fare'].median(), inplace=True)
test['Fare'].fillna(train['Fare'].median(), inplace=True)

# ==========================
# Family 関連の特徴量エンジニアリング（拡張）
# ==========================

# 基本の Family（兄弟＋親子）
train['Family'] = train['SibSp'] + train['Parch']
test['Family']  = test['SibSp'] + test['Parch']

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

print(train.head())
print(train.isnull().sum())

# ==========================
# ラベルエンコード（カテゴリ → 数値）
# すべてのモデル（RF, LGBM, CatBoost）で使えるようにする
# ==========================
label_encoders = {}
# 文字列のカテゴリ列すべて
for c in ["Sex", "Embarked", "Rare_name", "Prefix_Class", "FamilyCategory"]:
    label_encoders[c] = LabelEncoder()
    label_encoders[c].fit(pd.concat([train[c], test[c]]).astype(str))
    train[c] = label_encoders[c].transform(train[c].astype(str))
    test[c]  = label_encoders[c].transform(test[c].astype(str))

# ==========================
# 特徴量と目的変数に分ける
# ==========================
X = train.drop(columns=["Perished"])  # 説明変数
y = train["Perished"]                 # 目的変数

# データを7:3に分割（訓練データ70%、検証データ30%）
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.3, random_state=46, stratify=y
)

# CatBoost に渡すカテゴリ特徴量の「列インデックス」
cat_feature_names = ['Sex', 'Embarked', 'Rare_name', 'Prefix_Class', 'FamilyCategory']
cat_features = [X.columns.get_loc(c) for c in cat_feature_names]

# ==========================
# モデル定義（アンサンブル用）
# ==========================

# 1. RandomForest
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    min_samples_leaf=20,
    random_state=2025
)

# 2. LightGBM
lgbm_model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    num_leaves=8,
    min_data_in_leaf=25,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.1,
    random_state=2025
)

# 3. CatBoost
cb_model = CatBoostClassifier(
    iterations=400,
    learning_rate=0.05,
    depth=4,
    l2_leaf_reg=6.0,
    subsample=0.8,
    random_strength=1.5,
    loss_function='Logloss',
    random_state=2025,
    verbose=False
)

# ==========================
# 学習
# ==========================
rf_model.fit(X_train, y_train)
lgbm_model.fit(X_train, y_train)
cb_model.fit(X_train, y_train, cat_features=cat_features)

# ==========================
# 検証データで確率予測 → ブレンディング
# ==========================
rf_valid_proba   = rf_model.predict_proba(X_valid)[:, 1]
lgbm_valid_proba = lgbm_model.predict_proba(X_valid)[:, 1]
cb_valid_proba   = cb_model.predict_proba(X_valid)[:, 1]

# 単純平均ブレンディング
blend_valid_proba = (rf_valid_proba + lgbm_valid_proba + cb_valid_proba) / 3.0
y_valid_pred = (blend_valid_proba >= 0.5).astype(int)

accuracy_valid = accuracy_score(y_valid, y_valid_pred)
print("ブレンディング：検証データの正解率:", round(accuracy_valid, 4))

# ==========================
# 訓練データ側でも同じくブレンディング（過学習チェック）
# ==========================
rf_train_proba   = rf_model.predict_proba(X_train)[:, 1]
lgbm_train_proba = lgbm_model.predict_proba(X_train)[:, 1]
cb_train_proba   = cb_model.predict_proba(X_train)[:, 1]

blend_train_proba = (rf_train_proba + lgbm_train_proba + cb_train_proba) / 3.0
y_train_pred = (blend_train_proba >= 0.5).astype(int)

accuracy_train = accuracy_score(y_train, y_train_pred)
print("ブレンディング：訓練データの正解率:", round(accuracy_train, 4))

# ==========================
# テストデータで予測 → ブレンディングして提出ファイル作成
# ==========================
rf_test_proba   = rf_model.predict_proba(test)[:, 1]
lgbm_test_proba = lgbm_model.predict_proba(test)[:, 1]
cb_test_proba   = cb_model.predict_proba(test)[:, 1]

blend_test_proba = (rf_test_proba + lgbm_test_proba + cb_test_proba) / 3.0
blend_test_pred  = (blend_test_proba >= 0.5).astype(int)

submission = pd.read_csv(PATH + 'gender_submission.csv')
submission['Perished'] = blend_test_pred
submission.to_csv(PATH + 'submission.csv', index=False)
print("submission.csv を出力しました（ブレンディング版）")

print(train.head())
