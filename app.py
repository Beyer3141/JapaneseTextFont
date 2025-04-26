# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import sys
import io
from typing import Optional, Dict, List, Tuple, Any

# 自作モジュールをインポート
from utils.data_processing import preprocess_data, detect_problem_ads, parse_csv_from_upload
from utils.analysis import find_best_regression, compute_vif, perform_kmeans_clustering
from utils.visualization import setup_japanese_fonts, plot_ctr_ar_scatter, plot_diagnostics, show_regression_summary

# 日本語フォントの設定
setup_japanese_fonts()

def main():
    st.title("求人広告データ分析ツール")
    st.write("CSVファイルをアップロードして求人広告データを分析します。")
    
    # サイドバーの設定
    st.sidebar.header("分析設定")
    n_clusters = st.sidebar.slider("クラスタ数", 2, 5, 3)
    target_col = st.sidebar.selectbox(
        "目的変数", 
        ["応募数", "クリック数", "費用", "表示回数"], 
        index=0
    )
    test_size = st.sidebar.slider("テストデータの割合", 0.1, 0.5, 0.2, 0.05)
    
    # ファイルアップロード
    uploaded_file = st.file_uploader("CSVファイルをアップロード", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # CSVファイルを読み込み
            df = parse_csv_from_upload(uploaded_file)
            
            # データの概要を表示
            st.subheader("データの概要")
            st.write(f"行数: {df.shape[0]}, 列数: {df.shape[1]}")
            
            # データプレビュー
            st.write("データプレビュー:")
            st.dataframe(df.head())
            
            # 1. クリック率高 & 応募率低の原稿検出
            st.subheader("クリック率高 & 応募率低の原稿検出")
            problem_ads = detect_problem_ads(df)
            
            if not problem_ads.empty:
                st.write("以下の原稿はクリック率が高いにも関わらず応募率が低い傾向があります:")
                st.dataframe(problem_ads)
            else:
                st.info("クリック率高 & 応募率低の原稿は検出されませんでした。")
            
            # 2. CTR vs AR 散布図 & KMeansクラスタリング
            st.subheader("クリック率 vs 応募率分析")
            
            # クラスタリング実行
            df_cluster, kmeans = perform_kmeans_clustering(df, n_clusters=n_clusters)
            
            if not df_cluster.empty:
                # 散布図を表示
                show_title = st.checkbox("求人タイトルを表示", value=False)
                fig = plot_ctr_ar_scatter(df_cluster, kmeans, with_title=show_title)
                st.pyplot(fig)
                
                # クラスタごとの統計情報
                st.write("クラスタごとの統計情報:")
                cluster_stats = df_cluster.groupby('cluster')[['CTR', 'AR']].agg(['mean', 'std', 'min', 'max'])
                st.dataframe(cluster_stats)
            else:
                st.warning("クリック率またはクリック率のデータが不足しているため、クラスタリングを実行できません。")
            
            # 3. 回帰分析
            st.subheader(f"回帰分析 (目的変数: {target_col})")
            
            try:
                # 最適な回帰モデルを探索
                best_feats, best_r2, ols_model, X_test, y_test, y_pred = find_best_regression(
                    df, target_col=target_col, test_size=test_size
                )
                
                # 結果表示
                st.write(f"最適モデルの特徴量: {', '.join(best_feats)}")
                st.write(f"R2スコア: {best_r2:.4f}")
                
                # 回帰係数と統計情報
                coef_df, stats_df = show_regression_summary(ols_model, best_feats, best_r2)
                
                # 回帰係数を表示
                st.write("回帰係数:")
                st.dataframe(coef_df)
                
                # 統計情報を表示
                st.write("モデル統計:")
                st.dataframe(stats_df)
                
                # VIF (多重共線性)
                st.subheader("多重共線性の検証 (VIF)")
                vif_df = compute_vif(X_test)
                st.dataframe(vif_df)
                
                # 診断プロット
                st.subheader("診断プロット")
                fig_resid, fig_qq = plot_diagnostics(ols_model, X_test, y_test, y_pred)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(fig_resid)
                    st.write("残差プロット: 予測値と残差の関係を示します。パターンがなければ良いモデルです。")
                
                with col2:
                    st.pyplot(fig_qq)
                    st.write("QQプロット: 残差の正規性を確認します。直線に近いほど良いモデルです。")
                
            except Exception as e:
                st.error(f"回帰分析中にエラーが発生しました: {str(e)}")
                
        except Exception as e:
            st.error(f"データ処理中にエラーが発生しました: {str(e)}")
    else:
        st.info("CSVファイルをアップロードしてください。")
        
    # フッター
    st.markdown("---")
    st.write("求人広告データ分析ツール | 利用上の注意: データは適切にフォーマットされている必要があります")

if __name__ == "__main__":
    main()
