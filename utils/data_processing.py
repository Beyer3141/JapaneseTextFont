# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import io
import re
from typing import Optional, Tuple, List, Dict

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    前処理: 通貨列の数値化、カテゴリ変数のダミー化、数値変数の対数変換
    
    Args:
        df: 入力データフレーム
        
    Returns:
        前処理済みのデータフレーム
    """
    df = df.copy()
    
    # 通貨列の数値化 (￥や,が含まれる列)
    for col in df.columns:
        if df[col].dtype == object and df[col].astype(str).str.contains('￥|,').any():
            df[col] = pd.to_numeric(df[col].astype(str).replace('[￥,]', '', regex=True), errors='coerce')
    
    # カテゴリ変数（勤務地, キャンペーン名）のダミー化
    for c in ['勤務地', 'キャンペーン名']:
        if c in df.columns:
            df = pd.get_dummies(df, columns=[c], drop_first=True)
    
    # 数値変数の対数変換（クリック数, 費用）
    for c in ['クリック数', '費用']:
        if c in df.columns:
            df[c] = np.log1p(df[c])
            
    return df

def convert_percentage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    パーセント表記の列を小数に変換する
    
    Args:
        df: 入力データフレーム
        
    Returns:
        変換後のデータフレーム（CTR, AR列を追加）
    """
    df2 = df.copy()
    
    # パーセント表記を小数に変換
    for col, new_col in [('クリック率', 'CTR'), ('応募率', 'AR')]:
        if col in df2.columns:
            df2[new_col] = df2[col].astype(str).str.rstrip('%').astype(float) / 100
            
    return df2

def detect_problem_ads(df: pd.DataFrame) -> pd.DataFrame:
    """
    クリック率高 & 応募率低の原稿を検出
    
    Args:
        df: 入力データフレーム
        
    Returns:
        クリック率が75%分位以上 & 応募率が25%分位以下の原稿のデータフレーム
    """
    df2 = convert_percentage_columns(df)
    
    if 'CTR' in df2.columns and 'AR' in df2.columns:
        ctr_thr = df2['CTR'].quantile(0.75)
        ar_thr = df2['AR'].quantile(0.25)
        anomalies = df2[(df2['CTR'] > ctr_thr) & (df2['AR'] < ar_thr)]
        
        # 結果を返す
        if not anomalies.empty:
            return anomalies[['求人番号', '求人タイトル', 'CTR', 'AR']]
            
    return pd.DataFrame()

def parse_csv_from_upload(uploaded_file) -> pd.DataFrame:
    """
    アップロードされたCSVファイルをデータフレームに変換
    
    Args:
        uploaded_file: Streamlitでアップロードされたファイル
        
    Returns:
        読み込まれたデータフレーム
    """
    try:
        return pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        # UTF-8でエラーが出たら他のエンコーディングを試す
        try:
            return pd.read_csv(uploaded_file, encoding='shift-jis')
        except UnicodeDecodeError:
            try:
                return pd.read_csv(uploaded_file, encoding='cp932')
            except:
                raise ValueError("ファイルのエンコーディングが認識できません。UTF-8、SHIFT-JIS、CP932のいずれかでエンコードしてください。")
    except Exception as e:
        raise ValueError(f"ファイルの読み込みエラー: {str(e)}")
