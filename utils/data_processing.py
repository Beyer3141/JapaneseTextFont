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

     # ← ここから追加 ↓
    # 「クリック数」と「費用」を文字列から数値へ強制変換
    for c in ['クリック数', '費用']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # ここまで追加 ↑
    
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

def parse_csv_from_upload(uploaded_file, group_by_title=True) -> pd.DataFrame:
    """
    アップロードされたCSVファイルをデータフレームに変換し、
    オプションで同じ求人タイトルのデータをグループ化して平均化
    
    Args:
        uploaded_file: Streamlitでアップロードされたファイル
        group_by_title: 同じタイトルの原稿をグループ化して平均化するかどうか（デフォルト：True）
        
    Returns:
        読み込まれたデータフレーム
    """
    try:
        # まずファイルを読み込む
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            # UTF-8でエラーが出たら他のエンコーディングを試す
            try:
                df = pd.read_csv(uploaded_file, encoding='shift-jis')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(uploaded_file, encoding='cp932')
                except:
                    raise ValueError("ファイルのエンコーディングが認識できません。UTF-8、SHIFT-JIS、CP932のいずれかでエンコードしてください。")
        
        # パーセント表記を数値に変換
        df = convert_percentage_columns(df)
        
        # 同じタイトルの原稿をまとめる（オプション）
        if group_by_title and '求人タイトル' in df.columns:
            # グループ化する前に情報を表示するために元のデータフレームの行数を保存
            original_rows = len(df)
            
            # 数値列のリスト
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            # グループ化して数値列の平均を計算
            grouped_df = df.groupby('求人タイトル')[numeric_cols].mean().reset_index()
            
            # カテゴリカル列（文字列列）のリスト
            categorical_cols = [col for col in df.columns 
                              if col not in numeric_cols 
                              and col != '求人タイトル' 
                              and df[col].dtype == 'object']
            
            # 最頻値を取得してグループ化データフレームに追加
            for col in categorical_cols:
                # 各グループごとに最頻値を計算
                mode_values = {}
                for title, group in df.groupby('求人タイトル'):
                    mode_val = group[col].mode()
                    mode_values[title] = mode_val.iloc[0] if not mode_val.empty else None
                
                # 値を設定
                grouped_df[col] = grouped_df['求人タイトル'].map(mode_values)
            
            # 前処理を適用
            processed_df = preprocess_data(grouped_df)
            
            # 元の行数とグループ化後の行数の情報を属性として追加
            processed_df.attrs['original_rows'] = original_rows
            processed_df.attrs['grouped_rows'] = len(processed_df)
            
            return processed_df
        
        # グループ化しない場合は通常の前処理を適用
        return preprocess_data(df)
        
    except Exception as e:
        raise ValueError(f"ファイルの読み込みエラー: {str(e)}")
