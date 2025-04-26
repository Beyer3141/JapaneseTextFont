# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
import itertools
from typing import List, Tuple, Dict, Any, Optional

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    多重共線性を検出するための分散拡大係数(VIF)を計算
    
    Args:
        X: 特徴量のデータフレーム
        
    Returns:
        特徴量とVIF値を含むデータフレーム
    """
    dfv = pd.DataFrame({'feature': X.columns})
    dfv['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return dfv

def find_best_regression(
    df: pd.DataFrame, 
    target_col: str = '応募数', 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[List[str], float, Any, pd.DataFrame, pd.Series, np.ndarray]:
    """
    最適な回帰モデルを探索する
    
    Args:
        df: 入力データフレーム
        target_col: 目的変数の列名
        test_size: テストデータの割合
        random_state: 乱数のシード値
        
    Returns:
        (最適な特徴量のリスト, R2スコア, OLSモデル, テスト特徴量, テスト目的変数, 予測値)のタプル
    """
    from utils.data_processing import preprocess_data
    
    # 前処理
    dfp = preprocess_data(df)
    
    # 数値型のみ抽出し、欠損値を削除
    num = dfp.select_dtypes(include=[np.number]).dropna()
    
    # 目的変数が存在するかチェック
    if target_col not in num.columns:
        raise KeyError(f"{target_col}がデータに含まれていません")
    
    # 除外する特徴量
    excl = ['アクション数']
    
    # 特徴量と目的変数を分離
    feats = [c for c in num.columns if c != target_col and c not in excl]
    y = num[target_col]
    X = num[feats]
    
    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 最適なモデルを探索
    best_r2, best_feats, best_pred = -np.inf, None, None
    
    # 特徴量の組み合わせを総当たりで試す
    for k in range(1, len(feats) + 1):
        for combo in itertools.combinations(feats, k):
            # モデルを作成・訓練
            model = LinearRegression().fit(X_train[list(combo)], y_train)
            # テストデータで予測
            pred = model.predict(X_test[list(combo)])
            # R2スコアを計算
            r = r2_score(y_test, pred)
            # より良いモデルが見つかったら更新
            if r > best_r2:
                best_r2, best_feats, best_pred = r, combo, pred
    
    # 最適な特徴量でOLSモデルを作成
    ols = sm.OLS(y, sm.add_constant(X[list(best_feats)])).fit()
    
    return best_feats, best_r2, ols, X_test[list(best_feats)], y_test, best_pred

def perform_kmeans_clustering(
    df: pd.DataFrame, 
    n_clusters: int = 3
) -> Tuple[pd.DataFrame, KMeans]:
    """
    CTRとARに対してKMeansクラスタリングを実行
    
    Args:
        df: 入力データフレーム（CTRとAR列が必要）
        n_clusters: クラスタ数
        
    Returns:
        (クラスタ情報を含むデータフレーム, KMeansモデル)のタプル
    """
    from utils.data_processing import convert_percentage_columns
    
    # パーセント表記を小数に変換
    df2 = convert_percentage_columns(df)
    
    # CTRとAR列がある場合のみ実行
    if 'CTR' in df2.columns and 'AR' in df2.columns:
        # 欠損値を含まない行のみ抽出
        df_cluster = df2[['CTR', 'AR', '求人タイトル']].dropna()
        
        if df_cluster.empty:
            return pd.DataFrame(), None
        
        # クラスタリングの入力データ
        X_cluster = df_cluster[['CTR', 'AR']].values
        
        # KMeansを実行
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_cluster['cluster'] = kmeans.fit_predict(X_cluster)
        
        return df_cluster, kmeans
        
    return pd.DataFrame(), None
