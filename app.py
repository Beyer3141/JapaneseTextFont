# -*- coding: utf-8 -*-
import os, streamlit as st
st.write("📂 Files in project root:", os.listdir("."))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import io
import sys
import io
import json
from typing import Optional, Dict, List, Tuple, Any

# 自作モジュールをインポート
from utils.data_processing import preprocess_data, detect_problem_ads, parse_csv_from_upload
from utils.analysis import find_best_regression, compute_vif, perform_kmeans_clustering
from utils.visualization import setup_japanese_fonts, plot_ctr_ar_scatter, plot_diagnostics, show_regression_summary

# 日本語フォントの設定
setup_japanese_fonts()

def generate_analysis_report(
    df: pd.DataFrame,
    problem_ads: pd.DataFrame,
    df_cluster: pd.DataFrame,
    cluster_stats: pd.DataFrame,
    best_feats: List[str],
    best_r2: float,
    coef_df: pd.DataFrame,
    vif_df: pd.DataFrame
) -> str:
    """
    分析結果をテキストレポートとして生成
    
    Args:
        df: 元のデータフレーム
        problem_ads: 問題のある広告のデータフレーム
        df_cluster: クラスタリング結果のデータフレーム
        cluster_stats: クラスタごとの統計情報
        best_feats: 最適モデルの特徴量
        best_r2: 決定係数R2
        coef_df: 回帰係数のデータフレーム
        vif_df: VIF値のデータフレーム
        
    Returns:
        分析レポートの文字列
    """
    report = []
    report.append("# 求人広告データ分析レポート")
    report.append(f"\n## データ概要")
    report.append(f"- 対象レコード数: {df.shape[0]}件")
    report.append(f"- 対象項目数: {df.shape[1]}項目")
    
    # 1. 改善が必要な原稿と優良広告の分析
    report.append("\n## 1. 広告パフォーマンス分析")
    
    if 'CTR' in df.columns and 'AR' in df.columns and '求人タイトル' in df.columns:
        ctr_mean = df['CTR'].mean()
        ar_mean = df['AR'].mean()
        
        # 4つの象限に分類
        high_ctr_high_ar = df[(df['CTR'] > ctr_mean) & (df['AR'] > ar_mean)]
        high_ctr_low_ar = df[(df['CTR'] > ctr_mean) & (df['AR'] <= ar_mean)]
        low_ctr_high_ar = df[(df['CTR'] <= ctr_mean) & (df['AR'] > ar_mean)]
        low_ctr_low_ar = df[(df['CTR'] <= ctr_mean) & (df['AR'] <= ar_mean)]
        
        # 優良広告（改善優先度低）
        report.append("\n### 1.1 優良広告（特徴なし・改善優先度低）")
        if not high_ctr_high_ar.empty:
            report.append(f"クリック率と応募率がともに平均以上の原稿が{len(high_ctr_high_ar)}件あります。")
            report.append("これらの原稿は優れたパフォーマンスを示しており、現時点での改善優先度は低いと判断されます。")
            report.append("特徴:")
            report.append("- 広告内容と求人内容の一致度が高い")
            report.append("- ターゲットユーザーに適切にアピールしている")
            report.append("- 応募プロセスが最適化されている")
            
            if '求人タイトル' in high_ctr_high_ar.columns:
                titles = high_ctr_high_ar['求人タイトル'].tolist()
                if len(titles) > 5:
                    titles = titles[:5] + ["...他"]
                report.append(f"優良広告の例: {', '.join(titles)}")
        else:
            report.append("クリック率と応募率がともに平均以上の原稿はありませんでした。すべての原稿に何らかの改善の余地があります。")
        
        # 問題のある広告（高CTR、低AR）
        report.append("\n### 1.2 内容改善が必要な広告（高CTR・低AR）")
        if not high_ctr_low_ar.empty:
            report.append(f"クリック率が高いにも関わらず応募率が低い原稿が{len(high_ctr_low_ar)}件検出されました。")
            report.append("これらの原稿は、ユーザーの興味を引くものの、実際に応募に至らないという問題があります。")
            report.append("原因として考えられるのは以下の点です：")
            report.append("- 広告内容と実際の求人内容の不一致")
            report.append("- 応募プロセスの複雑さや使いづらさ")
            report.append("- 給与・待遇などの条件が期待を下回っている")
            
            if '求人タイトル' in high_ctr_low_ar.columns:
                titles = high_ctr_low_ar['求人タイトル'].tolist()
                if len(titles) > 3:
                    titles = titles[:3] + ["...他"]
                report.append(f"内容改善が必要な原稿例: {', '.join(titles)}")
        else:
            report.append("クリック率高・応募率低の原稿は検出されませんでした。")
            
        # タイトル改善が必要な広告（低CTR、高AR）
        report.append("\n### 1.3 タイトル改善が必要な広告（低CTR・高AR）")
        if not low_ctr_high_ar.empty:
            report.append(f"クリック率が低いにも関わらず応募率が高い原稿が{len(low_ctr_high_ar)}件検出されました。")
            report.append("これらの原稿は、ユーザーが広告をクリックする確率は低いものの、クリックしたユーザーが応募する確率が高いという特徴があります。")
            report.append("改善のポイント:")
            report.append("- 広告タイトルをより魅力的にする")
            report.append("- ターゲットユーザーの関心を引くキーワードを活用する")
            report.append("- 広告の表示位置や表示回数を最適化する")
            
            if '求人タイトル' in low_ctr_high_ar.columns:
                titles = low_ctr_high_ar['求人タイトル'].tolist()
                if len(titles) > 3:
                    titles = titles[:3] + ["...他"]
                report.append(f"タイトル改善が必要な原稿例: {', '.join(titles)}")
        else:
            report.append("クリック率低・応募率高の原稿は検出されませんでした。")
    else:
        report.append("クリック率(CTR)または応募率(AR)のデータが不足しているため、詳細な分析ができません。")
    
    # 2. クラスタリング分析
    report.append("\n## 2. クラスタリング分析")
    if not df_cluster.empty and 'cluster' in df_cluster.columns:
        # クラスタの数を取得（MultiIndexの場合とフラットなIndexの場合の両方に対応）
        if hasattr(cluster_stats.index, 'levels'):
            n_clusters = len(cluster_stats.index.levels[0])
        else:
            n_clusters = len(df_cluster['cluster'].unique())
            
        report.append(f"クリック率(CTR)と応募率(AR)に基づいて、{n_clusters}つのクラスタに分類しました。")
        
        # 各クラスタの特徴を抽出
        for i in range(n_clusters):
            try:
                # MultiIndexの場合
                if hasattr(cluster_stats.index, 'levels'):
                    ctr_mean = cluster_stats.loc[i, ('CTR', 'mean')]
                    ar_mean = cluster_stats.loc[i, ('AR', 'mean')]
                # フラットなIndexの場合
                else:
                    cluster_data = df_cluster[df_cluster['cluster'] == i]
                    ctr_mean = cluster_data['CTR'].mean()
                    ar_mean = cluster_data['AR'].mean()
                
                # クラスタの特徴づけ
                ctr_level = "高い" if ctr_mean > df_cluster['CTR'].mean() else "低い"
                ar_level = "高い" if ar_mean > df_cluster['AR'].mean() else "低い"
                
                report.append(f"\n### クラスタ{i+1}の特徴:")
                report.append(f"- クリック率: {ctr_mean:.2%} ({ctr_level})")
                report.append(f"- 応募率: {ar_mean:.2%} ({ar_level})")
                
                # クラスタの評価
                if ctr_level == "高い" and ar_level == "高い":
                    report.append("評価: 非常に優れた広告グループです。高いクリック率と高い応募率を両立しており、広告内容と求人内容の一致度が高いと考えられます。")
                elif ctr_level == "高い" and ar_level == "低い":
                    report.append("評価: 改善が必要なグループです。クリック率は高いものの応募率が低く、広告と実際の求人内容に乖離がある可能性があります。応募プロセスの見直しも検討すべきです。")
                elif ctr_level == "低い" and ar_level == "高い":
                    report.append("評価: 効率的なグループです。クリック率は低いものの、クリックした人の応募率が高く、ターゲットが絞られていると言えます。広告のリーチを拡大することで効果が高まる可能性があります。")
                else:
                    report.append("評価: 全体的な見直しが必要なグループです。クリック率と応募率の両方が低く、広告の魅力と求人内容の両方を改善する必要があります。")
                    
                # クラスタに属する原稿タイトルの取得（グループ化したデータから）
                if '求人タイトル' in df.columns:
                    # グループ化データからマッチするタイトルを取得
                    grouped_titles = []
                    cluster_original_titles = df_cluster[df_cluster['cluster'] == i]['求人タイトル'].tolist()
                    
                    for title in set(cluster_original_titles):  # 重複を排除
                        if title in df['求人タイトル'].values:
                            grouped_titles.append(title)
                    
                    if grouped_titles:
                        report.append("\n**このクラスタの代表的な原稿タイトル:**")
                        for idx, title in enumerate(grouped_titles[:5]):  # 最初の5件のみ表示
                            report.append(f"{idx+1}. {title}")
                        if len(grouped_titles) > 5:
                            report.append(f"...他 {len(grouped_titles)-5}件")
            except:
                report.append(f"\nクラスタ{i+1}のデータが不足しています。")
    else:
        report.append("クラスタリングに必要なデータが不足しているため、分析を実行できませんでした。")
    
    # 3. 回帰分析
    report.append("\n## 3. 回帰分析結果")
    if best_feats and best_r2 > 0:
        report.append(f"目的変数に最も影響を与える要素を特定するために回帰分析を実施しました。")
        report.append(f"モデルの説明力: {best_r2:.2%}")
        
        if best_r2 < 0.3:
            report.append("モデルの説明力は低く、他の要因が大きく影響していると考えられます。")
        elif best_r2 < 0.7:
            report.append("モデルの説明力は中程度です。特定された要因は一定の影響力を持ちますが、他の要因も考慮する必要があります。")
        else:
            report.append("モデルの説明力は高く、特定された要因が大きな影響力を持っていると言えます。")
        
        # 重要な特徴量の抽出
        if not coef_df.empty:
            # P値でフィルタリングして有意な変数のみ抽出
            sig_vars = coef_df[coef_df['P値'] < 0.05]
            if not sig_vars.empty:
                report.append("\n### 統計的に有意な影響要因:")
                for _, row in sig_vars.iterrows():
                    if row['特徴量'] != '定数項':
                        impact = "正の" if row['係数'] > 0 else "負の"
                        significance = "強い" if row['P値'] < 0.01 else "ある程度の"
                        report.append(f"- {row['特徴量']}: {significance}{impact}影響 (係数: {row['係数']:.4f})")
            else:
                report.append("\n統計的に有意な変数は検出されませんでした。")
                
            # 多重共線性の確認
            if not vif_df.empty:
                high_vif = vif_df[vif_df['VIF'] > 10]
                if not high_vif.empty:
                    report.append("\n### 多重共線性の懸念がある変数:")
                    for _, row in high_vif.iterrows():
                        report.append(f"- {row['feature']}: VIF={row['VIF']:.2f}")
                    report.append("これらの変数間には強い相関関係があり、個別の影響力の解釈には注意が必要です。")
    else:
        report.append("回帰分析に必要なデータが不足しているため、分析を実行できませんでした。")
    
    # 4. 推奨アクション
    report.append("\n## 4. 推奨アクション")
    report.append("分析結果に基づく改善提案:")
    
    if not problem_ads.empty:
        report.append("1. クリック率が高く応募率が低い広告の内容を見直し、広告と実際の求人内容の一致度を高めてください。")
    
    if not df_cluster.empty and 'cluster' in df_cluster.columns:
        report.append("2. クラスタ分析に基づいて広告戦略を最適化してください:")
        # クラスタの数を取得
        n_clusters_action = len(df_cluster['cluster'].unique())
        
        for i in range(n_clusters_action):
            try:
                # クラスタデータの取得方法を調整
                if hasattr(cluster_stats.index, 'levels'):
                    ctr_mean = cluster_stats.loc[i, ('CTR', 'mean')]
                    ar_mean = cluster_stats.loc[i, ('AR', 'mean')]
                else:
                    cluster_data = df_cluster[df_cluster['cluster'] == i]
                    ctr_mean = cluster_data['CTR'].mean()
                    ar_mean = cluster_data['AR'].mean()
                    
                ctr_level = "高い" if ctr_mean > df_cluster['CTR'].mean() else "低い"
                ar_level = "高い" if ar_mean > df_cluster['AR'].mean() else "低い"
                
                if ctr_level == "高い" and ar_level == "高い":
                    report.append(f"   - クラスタ{i+1}: このグループの広告予算を増やし、成功パターンを他の広告にも適用してください。")
                elif ctr_level == "高い" and ar_level == "低い":
                    report.append(f"   - クラスタ{i+1}: 求人内容と応募プロセスを改善してください。広告に見合う内容になっているか確認してください。")
                elif ctr_level == "低い" and ar_level == "高い":
                    report.append(f"   - クラスタ{i+1}: 広告の表示回数を増やし、より多くのターゲットユーザーにリーチしてください。広告のデザインは変更せず、配信を最適化してください。")
                else:
                    report.append(f"   - クラスタ{i+1}: 広告内容と求人内容の両方を見直してください。ターゲット層のニーズに合っているか再検討してください。")
            except:
                pass
    
    if best_feats and best_r2 > 0 and not coef_df.empty:
        sig_vars = coef_df[coef_df['P値'] < 0.05]
        if not sig_vars.empty:
            report.append("3. 以下の要因に注目して広告戦略を最適化してください:")
            for _, row in sig_vars.iterrows():
                if row['特徴量'] != '定数項' and row['係数'] > 0:
                    report.append(f"   - {row['特徴量']}の値を増加させることで効果が期待できます。")
    
    report.append("\n本レポートは統計的分析に基づいていますが、実際のビジネス状況や市場環境も考慮して判断してください。")
    
    return "\n".join(report)

def main():
    st.title("求人ボックス広告分析")
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
            # ファイルを読み込む前にバッファーに保存して、複数回使えるようにする
            file_content = uploaded_file.read()
            
            # 元のデータ（タイトルの重複あり）を読み込み
            uploaded_file_copy1 = io.BytesIO(file_content)
            uploaded_file_copy1.name = uploaded_file.name
            df_original = parse_csv_from_upload(uploaded_file_copy1, group_by_title=False)
            
            # タイトルをグループ化したデータも読み込み（データ概要タブ用）
            uploaded_file_copy2 = io.BytesIO(file_content)
            uploaded_file_copy2.name = uploaded_file.name
            df_grouped = parse_csv_from_upload(uploaded_file_copy2, group_by_title=True)
            
            # グループ化情報を表示
            if hasattr(df_grouped, 'attrs') and 'original_rows' in df_grouped.attrs:
                original_rows = df_grouped.attrs['original_rows']
                grouped_rows = df_grouped.attrs['grouped_rows']
                reduction = original_rows - grouped_rows
                
                if reduction > 0:
                    st.success(f"{original_rows}件の原稿データを読み込みました。データ概要タブでは同じタイトルの原稿を統合して{grouped_rows}件で分析し、クラスタリングと回帰分析では元データ（{original_rows}件）をそのまま使用します。")
            
            # タブを作成
            tab1, tab2, tab3, tab4 = st.tabs(["データ概要", "クラスタリング分析", "回帰分析", "分析レポート"])
            
            with tab1:
                # データの概要を表示
                st.subheader("データの概要")
                st.write(f"行数: {df_grouped.shape[0]}, 列数: {df_grouped.shape[1]}")
                
                # データプレビュー
                st.write("データプレビュー（同一タイトル統合済み）:")
                st.dataframe(df_grouped.head())
                
                # 改善すべき原稿の総合的な分析
                st.subheader("改善すべき原稿の分析")
                
                try:
                    # 必要なデータを生成
                    problem_ads = detect_problem_ads(df_grouped)
                    
                    # クリック率と応募率の平均値を計算して4象限に分類
                    if 'CTR' in df_grouped.columns and 'AR' in df_grouped.columns and '求人タイトル' in df_grouped.columns:
                        ctr_mean = df_grouped['CTR'].mean()
                        ar_mean = df_grouped['AR'].mean()
                        
                        # 4つの象限に分類
                        high_ctr_high_ar = df_grouped[(df_grouped['CTR'] > ctr_mean) & (df_grouped['AR'] > ar_mean)]
                        high_ctr_low_ar = df_grouped[(df_grouped['CTR'] > ctr_mean) & (df_grouped['AR'] <= ar_mean)]
                        low_ctr_high_ar = df_grouped[(df_grouped['CTR'] <= ctr_mean) & (df_grouped['AR'] > ar_mean)]
                        low_ctr_low_ar = df_grouped[(df_grouped['CTR'] <= ctr_mean) & (df_grouped['AR'] <= ar_mean)]
                        
                        # 修正が必要な原稿（高CTR低AR、低CTR高AR、低CTR低AR）を統合
                        need_improvement = pd.concat([high_ctr_low_ar, low_ctr_high_ar, low_ctr_low_ar])
                        
                        # 重複を削除し、タイトルでソート
                        if not need_improvement.empty:
                            need_improvement = need_improvement.drop_duplicates(subset=['求人タイトル'])
                            need_improvement = need_improvement.sort_values('求人タイトル')
                            
                            # タイプ別にラベル付け
                            def get_type(row):
                                if row['CTR'] > ctr_mean and row['AR'] <= ar_mean:
                                    return "クリック率高・応募率低（内容改善が必要）"
                                elif row['CTR'] <= ctr_mean and row['AR'] > ar_mean:
                                    return "クリック率低・応募率高（タイトル改善が必要）"
                                else:
                                    return "クリック率低・応募率低（全面的な見直しが必要）"
                            
                            need_improvement['改善タイプ'] = need_improvement.apply(get_type, axis=1)
                            
                            # 表示
                            st.write(f"以下の{len(need_improvement)}件の原稿は改善が必要と分析されました:")
                            
                            # 改善タイプでグループ化して集計
                            improvement_counts = need_improvement['改善タイプ'].value_counts().reset_index()
                            improvement_counts.columns = ['改善タイプ', '件数']
                            
                            # グラフ化
                            fig, ax = plt.subplots(figsize=(10, 4))
                            bars = ax.bar(improvement_counts['改善タイプ'], improvement_counts['件数'], color=['orange', 'blue', 'red'])
                            ax.set_ylabel('原稿数')
                            ax.set_title('改善が必要な原稿タイプの内訳')
                            plt.xticks(rotation=15, ha='right')
                            
                            # 棒グラフの上に数値を表示
                            for bar in bars:
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                        f'{height:.0f}',
                                        ha='center', va='bottom')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # タイプ別に表示
                            st.write("**改善が必要な原稿のリスト:**")
                            for type_name, group in need_improvement.groupby('改善タイプ'):
                                with st.expander(f"{type_name} ({len(group)}件)", expanded=True):
                                    for idx, row in group.iterrows():
                                        st.write(f"- {row['求人タイトル']} (CTR: {row['CTR']:.2%}, AR: {row['AR']:.2%})")
                                    
                                    if type_name == "クリック率高・応募率低（内容改善が必要）":
                                        st.info("**改善提案**: 広告内容と実際の求人条件のギャップを埋め、応募プロセスを簡素化することを検討してください。")
                                    elif type_name == "クリック率低・応募率高（タイトル改善が必要）":
                                        st.info("**改善提案**: 広告タイトルをより魅力的にし、ターゲットユーザーの関心を引くキーワードを活用してください。")
                                    else:
                                        st.info("**改善提案**: 広告内容、ターゲティング、求人内容の全面的な見直しが必要です。競合他社の成功事例を参考にしてください。")
                        else:
                            st.success("分析の結果、特に改善が必要な原稿は検出されませんでした。")
                    
                except Exception as e:
                    st.error(f"分析中にエラーが発生しました: {str(e)}")
                
                # 特徴なし（改善の優先度は低い）の原稿を表示
                with st.expander("特徴なし（改善の優先度は低い）の原稿"):
                    if 'CTR' in df_grouped.columns and 'AR' in df_grouped.columns and '求人タイトル' in df_grouped.columns:
                        # 高CTR・高ARの原稿（特徴なし・改善優先度低）
                        normal_ads = high_ctr_high_ar
                        
                        if not normal_ads.empty:
                            st.write(f"以下の{len(normal_ads)}件の原稿は平均以上のパフォーマンスを示しており、現時点での改善優先度は低いと判断されます:")
                            for idx, row in normal_ads.iterrows():
                                st.write(f"- {row['求人タイトル']} (CTR: {row['CTR']:.2%}, AR: {row['AR']:.2%})")
                            st.success("これらの原稿は高いクリック率と高い応募率を両立しています。広告内容と求人内容の一致度が高く、ユーザーに適切にアピールできています。")
                        else:
                            st.warning("クリック率と応募率が両方とも平均以上の原稿はありませんでした。すべての原稿に何らかの改善の余地があります。")
                    else:
                        st.error("必要なデータ（CTR、AR、求人タイトル）が不足しています。")
            
            with tab2:    
                # 2. CTR vs AR 散布図 & KMeansクラスタリング
                st.subheader("クリック率 vs 応募率分析")
                
                # クラスタリング実行（元データを使用）
                df_cluster, kmeans = perform_kmeans_clustering(df_original, n_clusters=n_clusters)
                
                if not df_cluster.empty:
                    # 散布図を表示
                    show_title = st.checkbox("求人タイトルを表示", value=False)
                    fig = plot_ctr_ar_scatter(df_cluster, kmeans, with_title=show_title)
                    st.pyplot(fig)
                    
                    # クラスタごとの統計情報
                    st.subheader("クラスタごとの統計情報")
                    cluster_stats = df_cluster.groupby('cluster')[['CTR', 'AR']].agg(['mean', 'std', 'min', 'max'])
                    st.dataframe(cluster_stats)
                    
                    # クラスタの解釈
                    st.subheader("クラスタの解釈")
                    for i in range(kmeans.n_clusters):
                        try:
                            ctr_mean = cluster_stats.loc[i, ('CTR', 'mean')]
                            ar_mean = cluster_stats.loc[i, ('AR', 'mean')]
                            
                            ctr_level = "高い" if ctr_mean > df_cluster['CTR'].mean() else "低い"
                            ar_level = "高い" if ar_mean > df_cluster['AR'].mean() else "低い"
                            
                            if ctr_level == "高い" and ar_level == "高い":
                                evaluation = "非常に優れた広告グループです。広告内容と求人内容の一致度が高いと考えられます。"
                                advice = "このグループの広告戦略を他のグループにも応用してください。予算配分を増やすことも検討してください。"
                            elif ctr_level == "高い" and ar_level == "低い":
                                evaluation = "改善が必要なグループです。広告と実際の求人内容に乖離がある可能性があります。"
                                advice = "応募プロセスや求人内容を改善してください。広告の内容と実際の求人条件の一致度を高めることが重要です。"
                            elif ctr_level == "低い" and ar_level == "高い":
                                evaluation = "効率的なグループです。ターゲットが絞られていると言えます。"
                                advice = "広告タイトルの改善によりクリック率を向上させることで、より多くの応募が期待できます。"
                            else:
                                evaluation = "全体的な見直しが必要なグループです。"
                                advice = "広告内容、ターゲティング、求人内容の全体的な見直しが必要です。"
                                
                            st.info(f"**クラスタ{i+1}**: クリック率{ctr_mean:.2%}({ctr_level})、応募率{ar_mean:.2%}({ar_level}) - {evaluation}")
                            st.markdown(f"**改善提案**: {advice}")
                            
                            # クラスタに属する原稿のタイトル（元データとグループ化したデータの両方）を表示
                            if '求人タイトル' in df_cluster.columns:
                                # 元データのクラスタに属するタイトル
                                cluster_titles = df_cluster[df_cluster['cluster'] == i]['求人タイトル'].tolist()
                                
                                # グループ化データからマッチするタイトルを取得（重複なし）
                                grouped_titles = []
                                if '求人タイトル' in df_grouped.columns:
                                    for title in set(cluster_titles):  # 重複を排除
                                        if title in df_grouped['求人タイトル'].values:
                                            grouped_titles.append(title)
                                
                                if cluster_titles:
                                    with st.expander(f"クラスタ{i+1}に属する原稿 ({len(cluster_titles)}件)", expanded=True):
                                        # まずグループ化したデータから取得したタイトルを表示
                                        if grouped_titles:
                                            st.write("**このクラスタのグループ化された代表的な原稿タイトル:**")
                                            for idx, title in enumerate(grouped_titles):
                                                st.write(f"{idx+1}. {title}")
                                            
                                            # 区切り線
                                            st.markdown("---")
                                            
                                        # 次に元データのすべてのタイトルを表示（折りたたみ可能）
                                        with st.expander("このクラスタに属するすべての原稿タイトル（重複あり）"):
                                            for idx, title in enumerate(cluster_titles):
                                                st.write(f"{idx+1}. {title}")
                        except Exception as e:
                            st.warning(f"クラスタ{i+1}のデータ処理中にエラーが発生しました: {str(e)}")
                    
                    # クリック率と応募率の4象限分析
                    st.subheader("クリック率・応募率による4象限分析")
                    
                    if '求人タイトル' in df_cluster.columns:
                        # 平均値を基準に4つのグループに分類
                        ctr_mean = df_cluster['CTR'].mean()
                        ar_mean = df_cluster['AR'].mean()
                        
                        # 4つの象限に分類
                        high_ctr_high_ar = df_cluster[(df_cluster['CTR'] > ctr_mean) & (df_cluster['AR'] > ar_mean)]
                        high_ctr_low_ar = df_cluster[(df_cluster['CTR'] > ctr_mean) & (df_cluster['AR'] <= ar_mean)]
                        low_ctr_high_ar = df_cluster[(df_cluster['CTR'] <= ctr_mean) & (df_cluster['AR'] > ar_mean)]
                        low_ctr_low_ar = df_cluster[(df_cluster['CTR'] <= ctr_mean) & (df_cluster['AR'] <= ar_mean)]
                        
                        # グラフ表示
                        fig, ax = plt.subplots(figsize=(10, 8))
                        
                        # 散布図のプロット
                        ax.scatter(high_ctr_high_ar['CTR'], high_ctr_high_ar['AR'], 
                                   c='green', alpha=0.7, label='高CTR・高AR (優秀広告)')
                        ax.scatter(high_ctr_low_ar['CTR'], high_ctr_low_ar['AR'], 
                                   c='orange', alpha=0.7, label='高CTR・低AR (内容改善)')
                        ax.scatter(low_ctr_high_ar['CTR'], low_ctr_high_ar['AR'], 
                                   c='blue', alpha=0.7, label='低CTR・高AR (タイトル改善)')
                        ax.scatter(low_ctr_low_ar['CTR'], low_ctr_low_ar['AR'], 
                                   c='red', alpha=0.7, label='低CTR・低AR (全面見直し)')
                        
                        # 平均値の線
                        ax.axvline(x=ctr_mean, color='gray', linestyle='--', alpha=0.5)
                        ax.axhline(y=ar_mean, color='gray', linestyle='--', alpha=0.5)
                        
                        # グラフの設定
                        ax.set_xlabel('クリック率 (CTR)', fontsize=12)
                        ax.set_ylabel('応募率 (AR)', fontsize=12)
                        ax.set_title('クリック率と応募率の4象限分析', fontsize=14)
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        
                        # テキスト表示
                        ax.text(df_cluster['CTR'].max() * 0.9, df_cluster['AR'].min() * 1.1, 
                                f'平均CTR: {ctr_mean:.2%}\n平均AR: {ar_mean:.2%}', 
                                bbox=dict(facecolor='white', alpha=0.7))
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # 象限ごとの件数表示
                        counts = [
                            len(high_ctr_high_ar),
                            len(high_ctr_low_ar),
                            len(low_ctr_high_ar),
                            len(low_ctr_low_ar)
                        ]
                        labels = [
                            '高CTR・高AR\n(優秀広告)', 
                            '高CTR・低AR\n(内容改善)', 
                            '低CTR・高AR\n(タイトル改善)', 
                            '低CTR・低AR\n(全面見直し)'
                        ]
                        colors = ['green', 'orange', 'blue', 'red']
                        
                        fig2, ax2 = plt.subplots(figsize=(10, 4))
                        bars = ax2.bar(labels, counts, color=colors)
                        ax2.set_ylabel('原稿数')
                        ax2.set_title('4象限別の原稿数')
                        
                        # 棒グラフの上に数値を表示
                        for bar in bars:
                            height = bar.get_height()
                            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                    f'{height:.0f}',
                                    ha='center', va='bottom')
                        
                        plt.tight_layout()
                        st.pyplot(fig2)
                        
                        # 各象限の原稿を表示
                        st.write("### 4象限別の原稿リスト")
                        cols = st.columns(2)
                        
                        with cols[0]:
                            with st.expander(f"高CTR・高AR ({len(high_ctr_high_ar)}件) - 優秀広告", expanded=False):
                                st.markdown("**評価**: 最も効果的な広告グループです。広告内容と求人内容の一致度が高く、ターゲットユーザーに適切にアピールできています。")
                                st.markdown("**対策**: このグループの広告戦略を他の広告にも応用し、予算配分を増やすことを検討してください。")
                                if len(high_ctr_high_ar) > 0:
                                    st.write("所属する原稿:")
                                    for idx, row in high_ctr_high_ar.iterrows():
                                        st.write(f"- {row['求人タイトル']} (CTR: {row['CTR']:.2%}, AR: {row['AR']:.2%})")
                            
                            with st.expander(f"低CTR・高AR ({len(low_ctr_high_ar)}件) - タイトル改善が必要", expanded=True):
                                st.markdown("**評価**: 求人内容自体は魅力的ですが、広告タイトルや表現が魅力的でない可能性があります。")
                                st.markdown("**対策**: 広告タイトルを改善し、より魅力的なキーワードや表現を使用してクリック率を向上させることが重要です。")
                                if len(low_ctr_high_ar) > 0:
                                    st.write("所属する原稿:")
                                    for idx, row in low_ctr_high_ar.iterrows():
                                        st.write(f"- {row['求人タイトル']} (CTR: {row['CTR']:.2%}, AR: {row['AR']:.2%})")
                        
                        with cols[1]:
                            with st.expander(f"高CTR・低AR ({len(high_ctr_low_ar)}件) - 内容改善が必要", expanded=True):
                                st.markdown("**評価**: 広告は注目を集めていますが、実際の求人内容やプロセスに問題がある可能性があります。")
                                st.markdown("**対策**: 応募プロセスを簡素化し、広告内容と実際の求人条件の一致度を高めることが重要です。")
                                if len(high_ctr_low_ar) > 0:
                                    st.write("所属する原稿:")
                                    for idx, row in high_ctr_low_ar.iterrows():
                                        st.write(f"- {row['求人タイトル']} (CTR: {row['CTR']:.2%}, AR: {row['AR']:.2%})")
                            
                            with st.expander(f"低CTR・低AR ({len(low_ctr_low_ar)}件) - 全面的な見直しが必要", expanded=True):
                                st.markdown("**評価**: 広告の魅力も求人内容も十分ではない可能性があります。")
                                st.markdown("**対策**: 広告内容、ターゲティング、求人内容の全体的な見直しが必要です。競合他社の成功事例を参考にしてください。")
                                if len(low_ctr_low_ar) > 0:
                                    st.write("所属する原稿:")
                                    for idx, row in low_ctr_low_ar.iterrows():
                                        st.write(f"- {row['求人タイトル']} (CTR: {row['CTR']:.2%}, AR: {row['AR']:.2%})")
                    else:
                        st.warning("求人タイトルのデータがないため、詳細な分析ができません。")
                else:
                    st.warning("クリック率またはクリック率のデータが不足しているため、クラスタリングを実行できません。")
            
            with tab3:
                # 3. 回帰分析
                st.subheader(f"回帰分析 (目的変数: {target_col})")
                
                try:
                    # 最適な回帰モデルを探索（元データを使用）
                    best_feats, best_r2, ols_model, X_test, y_test, y_pred = find_best_regression(
                        df_original, target_col=target_col, test_size=test_size
                    )
                    
                    # 結果表示
                    st.write(f"最適モデルの特徴量: {', '.join(best_feats)}")
                    st.metric("モデルの説明力 (R2スコア)", f"{best_r2:.2%}")
                    
                    # モデルの説明力の解釈
                    if best_r2 < 0.3:
                        st.warning("モデルの説明力は低く、他の要因が大きく影響していると考えられます。")
                    elif best_r2 < 0.7:
                        st.info("モデルの説明力は中程度です。特定された要因は一定の影響力を持ちますが、他の要因も考慮する必要があります。")
                    else:
                        st.success("モデルの説明力は高く、特定された要因が大きな影響力を持っていると言えます。")
                    
                    # 回帰係数と統計情報
                    coef_df, stats_df = show_regression_summary(ols_model, best_feats, best_r2)
                    
                    # 回帰係数を表示
                    st.subheader("回帰係数（影響度）")
                    st.dataframe(coef_df.style.background_gradient(subset=['係数'], cmap='coolwarm'))
                    
                    # 統計的に有意な変数の抽出
                    sig_vars = coef_df[coef_df['P値'] < 0.05]
                    if not sig_vars.empty:
                        st.subheader("統計的に有意な影響要因")
                        for _, row in sig_vars.iterrows():
                            if row['特徴量'] != '定数項':
                                impact = "正の" if row['係数'] > 0 else "負の"
                                significance = "強い" if row['P値'] < 0.01 else "ある程度の"
                                st.write(f"- **{row['特徴量']}**: {significance}{impact}影響があります (係数: {row['係数']:.4f})")
                    
                    # 統計情報を表示
                    st.subheader("モデル統計")
                    st.dataframe(stats_df)
                    
                    # VIF (多重共線性)
                    st.subheader("多重共線性の検証 (VIF)")
                    vif_df = compute_vif(X_test)
                    st.dataframe(vif_df.style.background_gradient(subset=['VIF'], cmap='YlOrRd'))
                    
                    # 多重共線性の警告
                    high_vif = vif_df[vif_df['VIF'] > 10]
                    if not high_vif.empty:
                        st.warning(f"以下の変数に多重共線性の懸念があります: {', '.join(high_vif['feature'].tolist())}")
                        st.write("これらの変数間には強い相関関係があり、個別の影響力の解釈には注意が必要です。")
                    
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
            
            with tab4:
                # 4. 分析レポート
                st.subheader("分析レポート")
                
                # 変数の準備
                try:
                    problem_ads = detect_problem_ads(df_grouped)
                except:
                    problem_ads = pd.DataFrame()
                    
                try:
                    df_cluster, kmeans = perform_kmeans_clustering(df_original, n_clusters=n_clusters)
                    cluster_stats = df_cluster.groupby('cluster')[['CTR', 'AR']].agg(['mean', 'std', 'min', 'max'])
                except:
                    df_cluster = pd.DataFrame()
                    cluster_stats = pd.DataFrame()
                
                try:
                    best_feats, best_r2, ols_model, X_test, y_test, y_pred = find_best_regression(
                        df_original, target_col=target_col, test_size=test_size
                    )
                    coef_df, stats_df = show_regression_summary(ols_model, best_feats, best_r2)
                    vif_df = compute_vif(X_test)
                except:
                    best_feats = []
                    best_r2 = 0
                    coef_df = pd.DataFrame()
                    vif_df = pd.DataFrame()
                
                # レポート生成
                report = generate_analysis_report(
                    df_grouped, problem_ads, df_cluster, cluster_stats,
                    best_feats, best_r2, coef_df, vif_df
                )
                
                # レポート表示
                st.markdown(report)
                
                # レポートダウンロード
                st.download_button(
                    label="レポートをダウンロード",
                    data=report,
                    file_name="求人広告分析レポート.md",
                    mime="text/markdown"
                )
                
        except Exception as e:
            st.error(f"データ処理中にエラーが発生しました: {str(e)}")
    else:
        st.info("CSVファイルをアップロードしてください。")
        
    # フッター
    st.markdown("---")
    st.write("求人広告データ分析ツール | 利用上の注意: データは適切にフォーマットされている必要があります")

if __name__ == "__main__":
    main()
