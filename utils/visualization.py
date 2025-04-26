# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
from typing import List, Tuple, Dict, Any
import io
import base64
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st

def setup_japanese_fonts():
    """
    日本語フォントをグローバルに設定
    """
    # 日本語フォントをセットアップ（Streamlit環境用）
    plt.rcParams['font.family'] = ['IPAexGothic', 'IPAGothic', 'Yu Gothic', 'Meiryo', 'sans-serif']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.figsize'] = [10, 6]
    
    # Streamlit環境で日本語フォントを使用するための追加設定
    try:
        st.set_page_config(
            page_title="求人広告データ分析ツール",
            page_icon="📊",
            layout="wide",
        )
    except:
        # 既に設定されている場合は無視
        pass

def plot_ctr_ar_scatter(df_cluster, kmeans=None, with_title=True):
    """
    CTR vs AR の散布図を作成し、クラスタリング結果で色分け
    
    Args:
        df_cluster: クラスタリング情報を含むデータフレーム
        kmeans: KMeansモデル（Noneの場合はクラスタリングなし）
        with_title: 求人タイトルをプロットに表示するかどうか
    
    Returns:
        プロット画像をバイト列で返す
    """
    if df_cluster.empty:
        return None
    
    # プロット作成
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # カラーマップ定義
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    
    # クラスタ別にプロット
    if 'cluster' in df_cluster.columns and kmeans is not None:
        for i in range(kmeans.n_clusters):
            cluster_data = df_cluster[df_cluster['cluster'] == i]
            ax.scatter(
                cluster_data['CTR'], 
                cluster_data['AR'], 
                c=[cmap(i/kmeans.n_clusters)], 
                label=f'クラスタ {i+1}',
                s=80,
                alpha=0.7
            )
            
        # クラスタの中心をプロット
        if hasattr(kmeans, 'cluster_centers_'):
            ax.scatter(
                kmeans.cluster_centers_[:, 0],
                kmeans.cluster_centers_[:, 1],
                c='red',
                marker='X',
                s=200,
                label='クラスタ中心'
            )
    else:
        ax.scatter(df_cluster['CTR'], df_cluster['AR'], alpha=0.7, s=80)
    
    # 求人タイトルをプロット
    if with_title and '求人タイトル' in df_cluster.columns:
        for idx, row in df_cluster.iterrows():
            ax.annotate(
                row['求人タイトル'],
                (row['CTR'], row['AR']),
                fontsize=8,
                alpha=0.7,
                xytext=(5, 5),
                textcoords='offset points'
            )
    
    # グラフの設定
    ax.set_xlabel('クリック率（CTR）', fontsize=14)
    ax.set_ylabel('応募率（AR）', fontsize=14)
    ax.set_title('クリック率 vs 応募率', fontsize=16)
    ax.grid(True, alpha=0.3)
    
    if 'cluster' in df_cluster.columns and kmeans is not None:
        ax.legend(fontsize=12)
    
    plt.tight_layout()
    
    # プロットをStreamlitに表示するために返す
    return fig

def plot_diagnostics(ols_model, X_test, y_test, y_pred):
    """
    回帰診断プロットを作成（残差プロット、QQプロット）
    
    Args:
        ols_model: statsmodelsのOLSモデル
        X_test: テスト用特徴量
        y_test: テスト用目的変数
        y_pred: 予測値
        
    Returns:
        (残差プロット, QQプロット)のタプル
    """
    # 残差を計算
    residuals = y_test - y_pred
    
    # 残差 vs 予測値プロット
    fig_resid, ax_resid = plt.subplots(figsize=(10, 6))
    ax_resid.scatter(y_pred, residuals, alpha=0.7)
    ax_resid.axhline(0, linestyle='--', color='red')
    ax_resid.set_xlabel('予測値', fontsize=14)
    ax_resid.set_ylabel('残差', fontsize=14)
    ax_resid.set_title('残差 vs 予測値', fontsize=16)
    ax_resid.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # QQプロット
    fig_qq = plt.figure(figsize=(10, 6))
    ax_qq = fig_qq.add_subplot(111)
    qqplot(residuals, line='s', ax=ax_qq)
    ax_qq.set_title('QQプロット（正規性の確認）', fontsize=16)
    ax_qq.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig_resid, fig_qq

def show_regression_summary(ols_model, best_feats, best_r2):
    """
    回帰分析の結果をStreamlit用にフォーマット
    
    Args:
        ols_model: statsmodelsのOLSモデル
        best_feats: 最良モデルの特徴量
        best_r2: 最良モデルのR2スコア
        
    Returns:
        フォーマットされたサマリー文字列
    """
    # 回帰係数と統計情報を取得
    coef_df = pd.DataFrame({
        '特徴量': ['定数項'] + list(best_feats),
        '係数': ols_model.params.values,
        'P値': ols_model.pvalues.values,
        '有意性': ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '' for p in ols_model.pvalues]
    })
    
    # 統計情報
    stats_df = pd.DataFrame({
        '統計量': ['R2', '調整済みR2', 'F値', 'P値(F)', 'AIC', 'BIC'],
        '値': [
            f"{ols_model.rsquared:.4f}",
            f"{ols_model.rsquared_adj:.4f}",
            f"{ols_model.fvalue:.4f}",
            f"{ols_model.f_pvalue:.4f}",
            f"{ols_model.aic:.4f}",
            f"{ols_model.bic:.4f}"
        ]
    })
    
    return coef_df, stats_df
