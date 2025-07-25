# 制限酵素サイト導入ツール - Streamlit Webアプリ
# streamlit run app.py で実行

import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
import re
from itertools import product
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ページ設定
st.set_page_config(
    page_title="制限酵素サイト導入ツール",
    page_icon="🧬",
    layout="wide"
)

class RestrictionEnzymeFinder:
    def __init__(self):
        # コドンテーブル
        self.codon_table = {
            'A': ['GCA', 'GCC', 'GCG', 'GCT'],  # Ala
            'R': ['AGA', 'AGG', 'CGA', 'CGC', 'CGG', 'CGT'],  # Arg
            'N': ['AAC', 'AAT'],  # Asn
            'D': ['GAC', 'GAT'],  # Asp
            'C': ['TGC', 'TGT'],  # Cys
            'Q': ['CAA', 'CAG'],  # Gln
            'E': ['GAA', 'GAG'],  # Glu
            'G': ['GGA', 'GGC', 'GGG', 'GGT'],  # Gly
            'H': ['CAC', 'CAT'],  # His
            'I': ['ATA', 'ATC', 'ATT'],  # Ile
            'L': ['CTA', 'CTC', 'CTG', 'CTT', 'TTA', 'TTG'],  # Leu
            'K': ['AAA', 'AAG'],  # Lys
            'M': ['ATG'],  # Met
            'F': ['TTC', 'TTT'],  # Phe
            'P': ['CCA', 'CCC', 'CCG', 'CCT'],  # Pro
            'S': ['AGC', 'AGT', 'TCA', 'TCC', 'TCG', 'TCT'],  # Ser
            'T': ['ACA', 'ACC', 'ACG', 'ACT'],  # Thr
            'W': ['TGG'],  # Trp
            'Y': ['TAC', 'TAT'],  # Tyr
            'V': ['GTA', 'GTC', 'GTG', 'GTT'],  # Val
        }
        
        # 制限酵素データベース（拡充版）
        self.restriction_enzymes = {
            # AokiLab Common - 最もよく使用される制限酵素
            'EcoRI': {'site': 'GAATTC', 'cut': 'G/AATTC', 'length': 6, 'category': 'AokiLab Common'},
            'XhoI': {'site': 'CTCGAG', 'cut': 'C/TCGAG', 'length': 6, 'category': 'AokiLab Common'},
            'NotI': {'site': 'GCGGCCGC', 'cut': 'GC/GGCCGC', 'length': 8, 'category': 'AokiLab Common'},
            'XbaI': {'site': 'TCTAGA', 'cut': 'T/CTAGA', 'length': 6, 'category': 'AokiLab Common'},
            'SalI': {'site': 'GTCGAC', 'cut': 'G/TCGAC', 'length': 6, 'category': 'AokiLab Common'},
            'BamHI': {'site': 'GGATCC', 'cut': 'G/GATCC', 'length': 6, 'category': 'AokiLab Common'},
            
            # AokiLab Standard - よく使用される制限酵素
            'HindIII': {'site': 'AAGCTT', 'cut': 'A/AGCTT', 'length': 6, 'category': 'AokiLab Standard'},
            'KpnI': {'site': 'GGTACC', 'cut': 'GGTAC/C', 'length': 6, 'category': 'AokiLab Standard'},
            'PstI': {'site': 'CTGCAG', 'cut': 'CTGCA/G', 'length': 6, 'category': 'AokiLab Standard'},
            'NcoI': {'site': 'CCATGG', 'cut': 'C/CATGG', 'length': 6, 'category': 'AokiLab Standard'},
            'NdeI': {'site': 'CATATG', 'cut': 'CA/TATG', 'length': 6, 'category': 'AokiLab Standard'},
            'NheI': {'site': 'GCTAGC', 'cut': 'G/CTAGC', 'length': 6, 'category': 'AokiLab Standard'},
            'SpeI': {'site': 'ACTAGT', 'cut': 'A/CTAGT', 'length': 6, 'category': 'AokiLab Standard'},
            'ApaI': {'site': 'GGGCCC', 'cut': 'GGGCC/C', 'length': 6, 'category': 'AokiLab Standard'},
            'BglII': {'site': 'AGATCT', 'cut': 'A/GATCT', 'length': 6, 'category': 'AokiLab Standard'},
            'EcoRV': {'site': 'GATATC', 'cut': 'GAT/ATC', 'length': 6, 'category': 'AokiLab Standard'},
            'MluI': {'site': 'ACGCGT', 'cut': 'A/CGCGT', 'length': 6, 'category': 'AokiLab Standard'},
            'SacI': {'site': 'GAGCTC', 'cut': 'GAGCT/C', 'length': 6, 'category': 'AokiLab Standard'},
            'SmaI': {'site': 'CCCGGG', 'cut': 'CCC/GGG', 'length': 6, 'category': 'AokiLab Standard'},
            'AscI': {'site': 'GGCGCGCC', 'cut': 'GG/CGCGCC', 'length': 8, 'category': 'AokiLab Standard'},
            'PacI': {'site': 'TTAATTAA', 'cut': 'TTAAT/TAA', 'length': 8, 'category': 'AokiLab Standard'},
            'SwaI': {'site': 'ATTTAAAT', 'cut': 'ATTT/AAAT', 'length': 8, 'category': 'AokiLab Standard'},
            'ClaI': {'site': 'ATCGAT', 'cut': 'AT/CGAT', 'length': 6, 'category': 'AokiLab Standard'},
            'EagI': {'site': 'CGGCCG', 'cut': 'C/GGCCG', 'length': 6, 'category': 'AokiLab Standard'},
            'SbfI': {'site': 'CCTGCAGG', 'cut': 'CCTGCA/GG', 'length': 8, 'category': 'AokiLab Standard'},
            'SfiI': {'site': 'GGCCNNNNGGCC', 'cut': 'GGCCN/NNNGGCC', 'length': 12, 'category': 'AokiLab Standard'},
            
            # Others - 頻繁ではないがNEBで利用可能な制限酵素
            'AatII': {'site': 'GACGTC', 'cut': 'GACGT/C', 'length': 6, 'category': 'Others'},
            'AccI': {'site': 'GTMKAC', 'cut': 'GT/MKAC', 'length': 6, 'category': 'Others'},
            'Acc65I': {'site': 'GGTACC', 'cut': 'G/GTACC', 'length': 6, 'category': 'Others'},
            'AflII': {'site': 'CTTAAG', 'cut': 'C/TTAAG', 'length': 6, 'category': 'Others'},
            'AgeI': {'site': 'ACCGGT', 'cut': 'A/CCGGT', 'length': 6, 'category': 'Others'},
            'ApaLI': {'site': 'GTGCAC', 'cut': 'G/TGCAC', 'length': 6, 'category': 'Others'},
            'AseI': {'site': 'ATTAAT', 'cut': 'AT/TAAT', 'length': 6, 'category': 'Others'},
            'AvrII': {'site': 'CCTAGG', 'cut': 'C/CTAGG', 'length': 6, 'category': 'Others'},
            'BclI': {'site': 'TGATCA', 'cut': 'T/GATCA', 'length': 6, 'category': 'Others'},
            'BsiWI': {'site': 'CGTACG', 'cut': 'C/GTACG', 'length': 6, 'category': 'Others'},
            'BspDI': {'site': 'ATCGAT', 'cut': 'AT/CGAT', 'length': 6, 'category': 'Others'},
            'BspEI': {'site': 'TCCGGA', 'cut': 'T/CCGGA', 'length': 6, 'category': 'Others'},
            'BspHI': {'site': 'TCATGA', 'cut': 'T/CATGA', 'length': 6, 'category': 'Others'},
            'BsrBI': {'site': 'GAGCGG', 'cut': 'GAG/CGG', 'length': 6, 'category': 'Others'},
            'BssHII': {'site': 'GCGCGC', 'cut': 'G/CGCGC', 'length': 6, 'category': 'Others'},
            'BstBI': {'site': 'TTCGAA', 'cut': 'TT/CGAA', 'length': 6, 'category': 'Others'},
            'BstZ17I': {'site': 'GTATAC', 'cut': 'GTA/TAC', 'length': 6, 'category': 'Others'},
            'DraI': {'site': 'TTTAAA', 'cut': 'TTT/AAA', 'length': 6, 'category': 'Others'},
            'EcoO109I': {'site': 'RGGNCCY', 'cut': 'RG/GNCCY', 'length': 7, 'category': 'Others'},
            'FspI': {'site': 'TGCGCA', 'cut': 'TGC/GCA', 'length': 6, 'category': 'Others'},
            'HaeII': {'site': 'RGCGCY', 'cut': 'RGCGC/Y', 'length': 6, 'category': 'Others'},
            'HincII': {'site': 'GTYRAC', 'cut': 'GTY/RAC', 'length': 6, 'category': 'Others'},
            'HpaI': {'site': 'GTTAAC', 'cut': 'GTT/AAC', 'length': 6, 'category': 'Others'},
            'KasI': {'site': 'GGCGCC', 'cut': 'G/GCGCC', 'length': 6, 'category': 'Others'},
            'MscI': {'site': 'TGGCCA', 'cut': 'TGG/CCA', 'length': 6, 'category': 'Others'},
            'NaeI': {'site': 'GCCGGC', 'cut': 'GCC/GGC', 'length': 6, 'category': 'Others'},
            'NarI': {'site': 'GGCGCC', 'cut': 'GG/CGCC', 'length': 6, 'category': 'Others'},
            'NgoMIV': {'site': 'GCCGGC', 'cut': 'G/CCGGC', 'length': 6, 'category': 'Others'},
            'NruI': {'site': 'TCGCGA', 'cut': 'TCG/CGA', 'length': 6, 'category': 'Others'},
            'NsiI': {'site': 'ATGCAT', 'cut': 'ATGCA/T', 'length': 6, 'category': 'Others'},
            'PaeR7I': {'site': 'CTCGAG', 'cut': 'C/TCGAG', 'length': 6, 'category': 'Others'},
            'PmeI': {'site': 'GTTTAAAC', 'cut': 'GTTT/AAAC', 'length': 8, 'category': 'Others'},
            'PmlI': {'site': 'CACGTG', 'cut': 'CAC/GTG', 'length': 6, 'category': 'Others'},
            'Psp1406I': {'site': 'AACGTT', 'cut': 'AA/CGTT', 'length': 6, 'category': 'Others'},
            'PvuI': {'site': 'CGATCG', 'cut': 'CGAT/CG', 'length': 6, 'category': 'Others'},
            'PvuII': {'site': 'CAGCTG', 'cut': 'CAG/CTG', 'length': 6, 'category': 'Others'},
            'SacII': {'site': 'CCGCGG', 'cut': 'CCGC/GG', 'length': 6, 'category': 'Others'},
            'ScaI': {'site': 'AGTACT', 'cut': 'AGT/ACT', 'length': 6, 'category': 'Others'},
            'SnaBI': {'site': 'TACGTA', 'cut': 'TAC/GTA', 'length': 6, 'category': 'Others'},
            'SphI': {'site': 'GCATGC', 'cut': 'GCATG/C', 'length': 6, 'category': 'Others'},
            'SspI': {'site': 'AATATT', 'cut': 'AAT/ATT', 'length': 6, 'category': 'Others'},
            'StuI': {'site': 'AGGCCT', 'cut': 'AGG/CCT', 'length': 6, 'category': 'Others'},
            'XmaI': {'site': 'CCCGGG', 'cut': 'C/CCGGG', 'length': 6, 'category': 'Others'},
            'ZraI': {'site': 'GACGTC', 'cut': 'GAC/GTC', 'length': 6, 'category': 'Others'},
            'AhdI': {'site': 'GACNNNNNGTC', 'cut': 'GACNNN/NNGTC', 'length': 11, 'category': 'Others'},
            'AlwNI': {'site': 'CAGNNNCTG', 'cut': 'CAGNNN/CTG', 'length': 9, 'category': 'Others'},
            'BaeI': {'site': 'ACNNNNGTAYC', 'cut': 'ACNNNN/GTAYC', 'length': 11, 'category': 'Others'},
            'BsaI': {'site': 'GGTCTC', 'cut': 'GGTCTC/NNNN', 'length': 6, 'category': 'Others'},
            'BsmBI': {'site': 'CGTCTC', 'cut': 'CGTCTC/N', 'length': 6, 'category': 'Others'},
            'BspMI': {'site': 'ACCTGC', 'cut': 'ACCTGC/NNNN', 'length': 6, 'category': 'Others'},
            'Esp3I': {'site': 'CGTCTC', 'cut': 'CGTCTC/N', 'length': 6, 'category': 'Others'},
            'FokI': {'site': 'GGATG', 'cut': 'GGATG/NNNNNNNNN', 'length': 5, 'category': 'Others'},
            'HphI': {'site': 'GGTGA', 'cut': 'GGTGA/NNNNNNNN', 'length': 5, 'category': 'Others'},
            'MboI': {'site': 'GATC', 'cut': '/GATC', 'length': 4, 'category': 'Others'},
            'Sau3AI': {'site': 'GATC', 'cut': '/GATC', 'length': 4, 'category': 'Others'},
            'TaqI': {'site': 'TCGA', 'cut': 'T/CGA', 'length': 4, 'category': 'Others'},
            'HaeIII': {'site': 'GGCC', 'cut': 'GG/CC', 'length': 4, 'category': 'Others'},
            'MspI': {'site': 'CCGG', 'cut': 'C/CGG', 'length': 4, 'category': 'Others'},
            'HpaII': {'site': 'CCGG', 'cut': 'C/CGG', 'length': 4, 'category': 'Others'},
            'RsaI': {'site': 'GTAC', 'cut': 'GT/AC', 'length': 4, 'category': 'Others'},
            'AluI': {'site': 'AGCT', 'cut': 'AG/CT', 'length': 4, 'category': 'Others'},
        }
        
        # 優先度設定（カテゴリに基づく）
        self.enzyme_priority = {
            # AokiLab Common - 最高優先度
            'EcoRI': 1, 'XhoI': 1, 'NotI': 1, 'XbaI': 1, 'SalI': 1, 'BamHI': 1,
            
            # AokiLab Standard - 中優先度
            'HindIII': 2, 'KpnI': 2, 'PstI': 2, 'NcoI': 2, 'NdeI': 2, 'NheI': 2,
            'SpeI': 2, 'ApaI': 2, 'BglII': 2, 'EcoRV': 2, 'MluI': 2, 'SacI': 2,
            'SmaI': 2, 'AscI': 2, 'PacI': 2, 'SwaI': 2, 'ClaI': 2, 'EagI': 2,
            'SbfI': 2, 'SfiI': 2,
            
            # Others - 低優先度（デフォルトで3）
        }
    
    def find_restriction_sites_in_sequence(self, aa_sequence, selected_enzymes, max_aa_span=4):
        """選択された制限酵素でアミノ酸配列を検索"""
        results = []
        
        # 選択された制限酵素のみを検索
        target_enzymes = {name: data for name, data in self.restriction_enzymes.items() 
                         if name in selected_enzymes}
        
        for span in range(2, min(max_aa_span + 1, len(aa_sequence) + 1)):
            for start_pos in range(len(aa_sequence) - span + 1):
                aa_segment = aa_sequence[start_pos:start_pos + span]
                
                enzyme_matches = self.check_restriction_sites_in_segment(aa_segment, target_enzymes)
                
                for match in enzyme_matches:
                    results.append({
                        'enzyme': match['enzyme'],
                        'site': match['site'],
                        'cut_site': self.restriction_enzymes[match['enzyme']]['cut'],
                        'position': start_pos + 1,
                        'span': span,
                        'aa_pattern': aa_segment,
                        'priority': self.enzyme_priority.get(match['enzyme'], 3),  # Othersは優先度3
                        'codon_options': match['codon_options'][:5]  # 上位5個まで
                    })
        
        # 重複除去とソート
        unique_results = []
        seen = set()
        for result in results:
            key = (result['enzyme'], result['position'], result['span'])
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
        
        unique_results.sort(key=lambda x: (x['priority'], x['enzyme'], x['position']))
        return unique_results
    
    def check_restriction_sites_in_segment(self, aa_segment, target_enzymes):
        """アミノ酸セグメントで制限酵素サイトをチェック"""
        matches = []
        
        for enzyme_name, enzyme_data in target_enzymes.items():
            target_site = enzyme_data['site']
            
            codon_combinations = self.generate_all_codon_combinations(aa_segment)
            
            valid_options = []
            for combination in codon_combinations:
                dna_sequence = ''.join(combination['codons'])
                
                if target_site in dna_sequence:
                    site_start = dna_sequence.find(target_site)
                    valid_options.append({
                        'codons': combination['codons'],
                        'dna': dna_sequence,
                        'site_position': site_start,
                        'codon_text': ' '.join(combination['codons'])
                    })
            
            if valid_options:
                valid_options.sort(key=lambda x: abs(x['site_position'] - len(x['dna'])//2))
                matches.append({
                    'enzyme': enzyme_name,
                    'site': target_site,
                    'codon_options': valid_options[:10]
                })
        
        return matches
    
    def generate_all_codon_combinations(self, aa_segment):
        """アミノ酸セグメントのコドン組み合わせを生成"""
        codon_lists = []
        for aa in aa_segment:
            if aa in self.codon_table:
                codon_lists.append(self.codon_table[aa])
            else:
                codon_lists.append(['XXX'])
        
        combinations = []
        for combo in product(*codon_lists):
            combinations.append({'codons': combo})
        
        return combinations

def main():
    # タイトルとヘッダー
    st.title("🧬 制限酵素サイト導入ツール")
    st.markdown("**コドンの冗長性を利用した効率的な制限酵素サイト検索**")
    st.markdown(f"📊 **データベース**: {len(finder.restriction_enzymes)}種類の制限酵素を収録")
    
    # 使用方法の説明
    with st.expander("💡 使用方法"):
        st.markdown("""
        1. **アミノ酸配列を入力**: サイドバーにタンパク質の1文字コード配列を入力
        2. **制限酵素を選択**: 
           - 🌟 **AokiLab Common**: 最頻用の6酵素（EcoRI, XhoI, NotI, XbaI, SalI, BamHI）
           - ⭐ **AokiLab Standard**: よく使用される酵素（20種類）
           - 📋 **Others**: その他のNEB制限酵素（50種類以上）
        3. **最大スパンを調整**: 検索範囲のアミノ酸数を設定
        4. **結果を確認**: 4つのタブで詳細な解析結果を表示
        5. **データをエクスポート**: CSV/JSON形式でダウンロード可能
        """)
    
    st.markdown("---")
    
    # サイドバー設定
    st.sidebar.header("⚙️ 設定")
    
    # アミノ酸配列入力
    st.sidebar.subheader("📝 配列入力")
    aa_sequence = st.sidebar.text_area(
        "アミノ酸配列（1文字コード）:",
        value="MVKFNGLLKEGAQSVRQSL",
        height=100,
        help="例: MVKFNGLLKEGAQSVRQSL"
    )
    
    # 最大スパン設定
    max_span = st.sidebar.slider(
        "最大アミノ酸スパン:",
        min_value=2,
        max_value=6,
        value=4,
        help="制限酵素サイト検索で考慮する最大アミノ酸数"
    )
    
    # 制限酵素選択
    st.sidebar.subheader("🔬 制限酵素選択")
    
    finder = RestrictionEnzymeFinder()
    
    # カテゴリー別に制限酵素を整理
    categories = {}
    for enzyme, data in finder.restriction_enzymes.items():
        category = data['category']
        if category not in categories:
            categories[category] = []
        categories[category].append(enzyme)
    
    selected_enzymes = []
    
    # 全選択/全解除ボタン
    col1, col2 = st.sidebar.columns(2)
    if col1.button("全選択"):
        st.session_state.select_all = True
    if col2.button("全解除"):
        st.session_state.select_all = False
    
    # カテゴリー別チェックボックス（優先順位順に表示）
    category_order = ['AokiLab Common', 'AokiLab Standard', 'Others']
    for category in category_order:
        if category in categories:
            enzymes = sorted(categories[category])  # アルファベット順にソート
            st.sidebar.markdown(f"**{category}** ({len(enzymes)}個)")
            
            # デフォルト選択の設定
            if hasattr(st.session_state, 'select_all'):
                default_value = st.session_state.select_all
            else:
                default_value = (category == 'AokiLab Common')  # AokiLab Commonのみデフォルト選択
            
            for enzyme in enzymes:
                enzyme_data = finder.restriction_enzymes[enzyme]
                if st.sidebar.checkbox(
                    f"{enzyme} ({enzyme_data['site']})",
                    value=default_value,
                    key=f"enzyme_{enzyme}"
                ):
                    selected_enzymes.append(enzyme)
            
            st.sidebar.markdown("---")  # カテゴリ間の区切り線
    # メインコンテンツ
    if not aa_sequence.strip():
        st.warning("アミノ酸配列を入力してください。")
        return
    
    if not selected_enzymes:
        st.warning("制限酵素を選択してください。")
        return
    
    # 解析実行
    aa_sequence = aa_sequence.upper().strip()
    st.subheader(f"📊 解析結果: {aa_sequence}")
    
    # 選択された制限酵素の詳細情報
    selected_by_category = {}
    for enzyme in selected_enzymes:
        category = finder.restriction_enzymes[enzyme]['category']
        if category not in selected_by_category:
            selected_by_category[category] = 0
        selected_by_category[category] += 1
    
    category_summary = " | ".join([f"{cat}: {count}個" for cat, count in selected_by_category.items()])
    
    st.markdown(f"**配列長:** {len(aa_sequence)} アミノ酸 | **選択制限酵素:** {len(selected_enzymes)}個 ({category_summary})")
    
    with st.spinner("解析中..."):
        results = finder.find_restriction_sites_in_sequence(aa_sequence, selected_enzymes, max_span)
    
    if not results:
        st.error("選択した制限酵素サイトが見つかりませんでした。")
        st.info("💡 ヒント: 最大スパンを増やすか、異なる制限酵素を選択してみてください。")
        return
    
    # 結果表示
    st.success(f"🎉 {len(results)}個の制限酵素サイトが見つかりました！")
    
    # タブで結果を整理
    tab1, tab2, tab3, tab4 = st.tabs(["📋 結果一覧", "📊 可視化", "🧪 詳細情報", "💾 エクスポート"])
    
    with tab1:
        # 結果テーブル
        df_data = []
        for i, result in enumerate(results):
            df_data.append({
                'No.': i + 1,
                '制限酵素': result['enzyme'],
                '位置': f"{result['position']}-{result['position'] + result['span'] - 1}",
                'アミノ酸パターン': result['aa_pattern'],
                '認識配列': result['site'],
                '切断位置': result['cut_site'],
                'スパン': result['span'],
                '優先度': result['priority'],
                'オプション数': len(result['codon_options'])
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
        
        # 上位結果の詳細
        st.subheader("🔝 推奨制限酵素サイト（上位5個）")
        for i, result in enumerate(results[:5]):
            with st.expander(f"{i+1}. {result['enzyme']} (位置: {result['position']})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**認識配列:** `{result['site']}`")
                    st.markdown(f"**切断位置:** `{result['cut_site']}`")
                    st.markdown(f"**アミノ酸パターン:** `{result['aa_pattern']}`")
                
                with col2:
                    st.markdown(f"**位置:** {result['position']}-{result['position'] + result['span'] - 1}")
                    st.markdown(f"**スパン:** {result['span']} アミノ酸")
                    st.markdown(f"**優先度:** {result['priority']}")
                
                st.markdown("**推奨コドン組み合わせ:**")
                for j, option in enumerate(result['codon_options'][:3]):
                    st.code(f"{j+1}. {option['codon_text']} → {option['dna']}")
    
    with tab2:
        # 可視化
        if len(results) > 0:
            # データフレーム準備
            viz_data = []
            for result in results:
                viz_data.append({
                    'Enzyme': result['enzyme'],
                    'Position': result['position'],
                    'Span': result['span'],
                    'Priority': result['priority'],
                    'Options': len(result['codon_options'])
                })
            
            viz_df = pd.DataFrame(viz_data)
            
            # グラフ作成
            col1, col2 = st.columns(2)
            
            with col1:
                # 制限酵素の頻度
                enzyme_counts = viz_df['Enzyme'].value_counts()
                fig1 = px.bar(
                    x=enzyme_counts.index,
                    y=enzyme_counts.values,
                    title="制限酵素の出現頻度",
                    labels={'x': '制限酵素', 'y': '出現回数'}
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # 位置とスパンの関係
                fig2 = px.scatter(
                    viz_df,
                    x='Position',
                    y='Span',
                    color='Priority',
                    size='Options',
                    hover_data=['Enzyme'],
                    title="制限酵素サイトの位置とスパン",
                    labels={'Position': 'アミノ酸位置', 'Span': 'アミノ酸スパン'}
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # 優先度分布
            priority_counts = viz_df['Priority'].value_counts().sort_index()
            fig3 = px.pie(
                values=priority_counts.values,
                names=[f'Priority {i}' for i in priority_counts.index],
                title="制限酵素の優先度分布"
            )
            st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        # 詳細情報
        st.subheader("🔍 詳細解析情報")
        
        # 統計情報
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("総制限酵素サイト", len(results))
        with col2:
            unique_enzymes = len(set(r['enzyme'] for r in results))
            st.metric("ユニーク制限酵素", unique_enzymes)
        with col3:
            avg_span = np.mean([r['span'] for r in results])
            st.metric("平均スパン", f"{avg_span:.1f}")
        with col4:
            high_priority = len([r for r in results if r['priority'] <= 1])
            st.metric("AokiLab Common", high_priority)
        
        # カテゴリ別統計
        st.subheader("📈 カテゴリ別統計")
        category_stats = {}
        for result in results:
            enzyme = result['enzyme']
            category = finder.restriction_enzymes[enzyme]['category']
            if category not in category_stats:
                category_stats[category] = 0
            category_stats[category] += 1
        
        if category_stats:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("AokiLab Common", category_stats.get('AokiLab Common', 0))
            with col2:
                st.metric("AokiLab Standard", category_stats.get('AokiLab Standard', 0))
            with col3:
                st.metric("Others", category_stats.get('Others', 0))
        
        # 制限酵素データベース情報
        st.subheader("📚 制限酵素データベース情報")
        
        # データベース統計
        total_enzymes = len(finder.restriction_enzymes)
        common_count = len([e for e, d in finder.restriction_enzymes.items() if d['category'] == 'AokiLab Common'])
        standard_count = len([e for e, d in finder.restriction_enzymes.items() if d['category'] == 'AokiLab Standard'])
        others_count = len([e for e, d in finder.restriction_enzymes.items() if d['category'] == 'Others'])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("総制限酵素数", total_enzymes)
        with col2:
            st.metric("AokiLab Common", common_count)
        with col3:
            st.metric("AokiLab Standard", standard_count)
        with col4:
            st.metric("Others", others_count)
        
        # 選択された制限酵素の詳細テーブル
        st.subheader("🔬 選択された制限酵素の詳細")
        enzyme_info = []
        for enzyme in selected_enzymes:
            data = finder.restriction_enzymes[enzyme]
            enzyme_info.append({
                '制限酵素': enzyme,
                '認識配列': data['site'],
                '切断パターン': data['cut'],
                '長さ': data['length'],
                'カテゴリー': data['category'],
                '優先度': finder.enzyme_priority.get(enzyme, 3)
            })
        
        if enzyme_info:
            enzyme_df = pd.DataFrame(enzyme_info)
            # カテゴリー別に色分け
            st.dataframe(
                enzyme_df.style.apply(
                    lambda x: ['background-color: #e8f5e8' if v == 'AokiLab Common' 
                              else 'background-color: #e8f0ff' if v == 'AokiLab Standard'
                              else 'background-color: #fff5e8' if v == 'Others'
                              else '' for v in x], 
                    subset=['カテゴリー']
                ),
                use_container_width=True
            )
        else:
            st.info("制限酵素を選択してください。")
    
    with tab4:
        # エクスポート
        st.subheader("💾 結果のエクスポート")
        
        # 詳細結果データフレーム
        export_data = []
        for result in results:
            for i, option in enumerate(result['codon_options']):
                export_data.append({
                    'Enzyme': result['enzyme'],
                    'Position': result['position'],
                    'End_Position': result['position'] + result['span'] - 1,
                    'AA_Pattern': result['aa_pattern'],
                    'Recognition_Site': result['site'],
                    'Cut_Site': result['cut_site'],
                    'Span': result['span'],
                    'Priority': result['priority'],
                    'Option_Rank': i + 1,
                    'Codons': option['codon_text'],
                    'DNA_Sequence': option['dna'],
                    'Site_Position_in_DNA': option['site_position']
                })
        
        export_df = pd.DataFrame(export_data)
        
        # CSV ダウンロード
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📄 CSV形式でダウンロード",
            data=csv,
            file_name=f"restriction_enzyme_analysis_{aa_sequence[:10]}.csv",
            mime="text/csv"
        )
        
        # JSON ダウンロード
        import json
        json_data = json.dumps(results, indent=2, ensure_ascii=False)
        st.download_button(
            label="📋 JSON形式でダウンロード",
            data=json_data,
            file_name=f"restriction_enzyme_analysis_{aa_sequence[:10]}.json",
            mime="application/json"
        )
        
        # データプレビュー
        st.subheader("👀 エクスポートデータプレビュー")
        st.dataframe(export_df.head(10), use_container_width=True)

if __name__ == "__main__":
    main()