import numpy as np                                   # 数値計算や配列操作
import pandas as pd                                  # 表形式のデータを扱う

from sklearn.metrics import accuracy_score           # 正解率の計算
from catboost import CatBoostClassifier

PATH = "C:\\Users\\piyop\\OneDrive\\デスクトップ\\GCI 2025 Winter\\data\\"
train = pd.read_csv(PATH + 'train.csv') 
test = pd.read_csv(PATH + 'test.csv')

# =====================================================
# Ticket 関連特徴量
# =====================================================

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

# Ticket 数値部分
train["Ticket_Num"] = train["Ticket"].str.extract(r'(\d+)', expand=False).astype(float)
test["Ticket_Num"]  = test["Ticket"].str.extract(r'(\d+)', expand=False).astype(float)

# NaN は -1 にしておく（log のため）
train["Ticket_Num"] = train["Ticket_Num"].fillna(-1)
test["Ticket_Num"]  = test["Ticket_Num"].fillna(-1)

train["Ticket_Num_log"] = np.log1p(train["Ticket_Num"])
test["Ticket_Num_log"]  = np.log1p(test["Ticket_Num"])

# =====================================================
# 敬称・年齢・家族・Cabin 関連
# =====================================================

# 敬称抽出+Rare敬称をまとめる
train['Title'] = train['Name'].str.extract(r', (\w+)\.', expand=False)
test['Title'] = test['Name'].str.extract(r', (\w+)\.', expand=False)

train["Rare_name"] = train["Title"].replace(
    ['Rev', 'Col', 'Major', 'Capt', 'Sir',
     'Lady', 'Countess', 'Jonkheer', 'Don', 'Dona',
     'Mlle', 'Mme', 'Ms'],
    'Rare',
    inplace=False
)
test["Rare_name"] = test["Title"].replace(
    ['Rev', 'Col', 'Major', 'Capt', 'Sir',
     'Lady', 'Countess', 'Jonkheer', 'Don', 'Dona',
     'Mlle', 'Mme', 'Ms'],
    'Rare',
    inplace=False
)

train['Rare_name'] = train['Rare_name'].fillna(train['Rare_name'].mode()[0])
test['Rare_name']  = test['Rare_name'].fillna(train['Rare_name'].mode()[0])

# Titleごとの中央値でAgeを埋める
train['Age'] = train['Age'].fillna(train.groupby('Title')['Age'].transform('median'))
test['Age']  = test['Age'].fillna(test.groupby('Title')['Age'].transform('median'))

# Embarked を最頻値で埋める（将来のために代入スタイルに変更）
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
test['Embarked']  = test['Embarked'].fillna(train['Embarked'].mode()[0])

# Fareの欠損値を中央値で補完
train['Fare'] = train['Fare'].fillna(train['Fare'].median())
test['Fare']  = test['Fare'].fillna(train['Fare'].median())

train['Fare_log'] = np.log1p(train['Fare'])
test['Fare_log']  = np.log1p(test['Fare'])

# 自分を含めた家族人数
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize']  = test['SibSp'] + test['Parch'] + 1

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

# Deck
train["Deck"] = train["Cabin"].astype(str).str[0]
test["Deck"]  = test["Cabin"].astype(str).str[0]

# NaN（= 'n' など）を Unknown に置き換える
train["Deck"] = train["Deck"].replace("n", "Unknown")
test["Deck"]  = test["Deck"].replace("n", "Unknown")

# RARE デッキをまとめる（A/B/C/D/E/F/G 以外）
valid_decks = ["A", "B", "C", "D", "E", "F", "G"]
train.loc[~train["Deck"].isin(valid_decks), "Deck"] = "Unknown"
test.loc[~test["Deck"].isin(valid_decks),  "Deck"] = "Unknown"

# Cabin から部屋番号（数字）を抽出
train["Cabin_Num"] = train["Cabin"].astype(str).str.extract(r'(\d+)', expand=False)
test["Cabin_Num"]  = test["Cabin"].astype(str).str.extract(r'(\d+)', expand=False)

train["Cabin_Num"] = train["Cabin_Num"].astype(float).fillna(-1)
test["Cabin_Num"]  = test["Cabin_Num"].astype(float).fillna(-1)

def cabin_floor_bin(x):
    if x == -1:
        return "Unknown"     # Cabin 情報なし（3等客に多い）
    elif x < 40:
        return "Upper"       # 0〜39 → 比較的上層階
    elif x < 120:
        return "Middle"      # 40~119 → 中層階
    else:
        return "Lower"       # 120以上 → 下層階（浸水しやすいと仮定）

train["Cabin_Floor"] = train["Cabin_Num"].apply(cabin_floor_bin)
test["Cabin_Floor"]  = test["Cabin_Num"].apply(cabin_floor_bin)

train["Has_Cabin"] = (train["Cabin_Num"] != -1).astype(int)
test["Has_Cabin"]  = (test["Cabin_Num"] != -1).astype(int)

def count_cabins(x):
    if pd.isna(x):
        return 0
    return len(str(x).split())

train["Cabin_Count"] = train["Cabin"].apply(count_cabins)
test["Cabin_Count"]  = test["Cabin"].apply(count_cabins)

# =====================================================
# ★ あなたが貼ってくれた AgeBin / Key / 分布に合わせた検証データ作成
# =====================================================

# df, df_test として扱う
df = train.copy()
df_test = test.copy()

# 年齢を 15 ビンに分割（train から bins を作り test にも適用）
out, bins = pd.cut(df['Age'], 15, retbins=True, labels=False)
df['AgeBin'] = out
df_test['AgeBin'] = pd.cut(df_test['Age'], bins=bins, include_lowest=True, labels=False)

# 念のため欠損があった場合に備えて埋めておく
df['AgeBin'] = df['AgeBin'].fillna(-1).astype(int)
df_test['AgeBin'] = df_test['AgeBin'].fillna(-1).astype(int)

# Pclass × Sex × AgeBin で Key を作成
df['Key'] = df['Pclass'].astype(str) + '_' + df['Sex'].astype(str) + '_' + df['AgeBin'].astype(str)
df_test['Key'] = df_test['Pclass'].astype(str) + '_' + df_test['Sex'].astype(str) + '_' + df_test['AgeBin'].astype(str)

# テストデータの Key の分布に近づける形で検証データサイズを決める
val_size = 0.2  # train の 20% を検証に回す
df_test_distribution = (df_test['Key'].value_counts(normalize=True) * len(df) * val_size).round().astype(int)

# 検証用データに使うインデックスのリスト作成
validation_indices = []
for key, n in df_test_distribution.items():
    values = df[df['Key'] == key]
    if len(values) > 0:
        validation_indices += values.sample(min(n, len(values)), random_state=42).index.tolist()
    elif n > 0:
        print(f'Key {key} ({n}) は学習用データに見つかりませんでした。')

# 検証用データに使うインデックス以外を学習データにする
df_train = df.drop(validation_indices).reset_index(drop=True)
df_val   = df.loc[validation_indices].reset_index(drop=True)

print(f'学習データ: {len(df_train)}、検証データ: {len(df_val)}（比率{(len(df_val)/len(df)):.3f}）')

# =====================================================
# Drop するカラムを df_train / df_val / df_test 全部に適用
# =====================================================

drop_cols = ['Ticket', 'Cabin', 'PassengerId', 'Name', 'Title',
             'Sex', 'Ticket_Prefix', 'SibSp', 'Parch']

df_train = df_train.drop(columns=drop_cols)
df_val   = df_val.drop(columns=drop_cols)
df_test  = df_test.drop(columns=drop_cols)

# =====================================================
# 特徴量と目的変数に分ける
# =====================================================

X_train = df_train.drop(columns=["Perished"])
y_train = df_train["Perished"]

X_valid = df_val.drop(columns=["Perished"])
y_valid = df_val["Perished"]

# テストデータ（提出用）
X_test = df_test.copy()

# CatBoost で使うカテゴリ列を指定（手動版）
cat_features = [
    'Embarked',
    'Rare_name',
    'Prefix_Class',
    'FamilyCategory',
    'Deck',
    'Cabin_Floor',
    'Key',          # ← ここもカテゴリとして扱う
]

# =====================================================
# モデルの設定（CatBoost）
# =====================================================

model = CatBoostClassifier(
    iterations=400,
    learning_rate=0.03,
    depth=4,
    l2_leaf_reg=4.0,
    loss_function='Logloss',
    random_state=2025,
    verbose=False,
    od_type='Iter',
    od_wait=50,
    thread_count=-1
)

# 学習
model.fit(
    X_train,
    y_train,
    eval_set=(X_valid, y_valid),
    cat_features=cat_features,
    use_best_model=True
)

# 検証データで予測
y_valid_pred = model.predict(X_valid)
accuracy1 = accuracy_score(y_valid, y_valid_pred)
print("検証データの正解率:", round(accuracy1, 4))

# 学習データに対するスコア（過学習チェック）
train_pred = model.predict(X_train)
accuracy2 = accuracy_score(y_train, train_pred)
print("訓練データの正解率:", round(accuracy2, 4))
print("訓練データの正解率と検証データの正解率の差異:", round(accuracy2 - accuracy1, 4))

# テストデータ予測 → 提出ファイル作成
pred = model.predict(X_test)

submission = pd.read_csv(PATH + 'gender_submission.csv')
submission['Perished'] = pred
submission.to_csv(PATH + 'submission.csv', index=False)

print("submission.csv を出力しました")
print("検証データの正解率(raw):", accuracy1)
print("訓練データの正解率(raw):", accuracy2)
print(df_train)