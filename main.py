import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Mobile Reviews Cluster Dashboard", layout="wide")

# ── Global Style ───────────────────────────────────────────────────────────────
st.markdown("""
    <style>
        [data-testid="stSidebar"] { background-color: #1a1a2e; }
        [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
        .metric-card {
            background-color: #f4f6f9;
            border-left: 4px solid #1a73e8;
            padding: 14px 18px;
            border-radius: 6px;
            margin-bottom: 8px;
        }
        .metric-card h4 { margin: 0; font-size: 13px; color: #555; font-weight: 500; }
        .metric-card h2 { margin: 4px 0 0 0; font-size: 26px; color: #1a1a2e; font-weight: 700; }
        .metric-card span { font-size: 12px; color: #777; }
        .section-title {
            font-size: 15px;
            font-weight: 600;
            color: #1a1a2e;
            border-bottom: 2px solid #1a73e8;
            padding-bottom: 5px;
            margin-bottom: 14px;
        }
        .insight-box {
            padding: 14px 18px;
            border-radius: 6px;
            font-size: 14px;
            line-height: 1.7;
            margin-top: 8px;
        }
        .insight-happy   { background: #e8f5e9; border-left: 4px solid #2e7d32; color: #1b5e20; }
        .insight-avg     { background: #fff8e1; border-left: 4px solid #f9a825; color: #5d4037; }
        .insight-unhappy { background: #fce4ec; border-left: 4px solid #c62828; color: #7f0000; }
    </style>
""", unsafe_allow_html=True)

# ── Load & Cluster Data ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df_clean = pd.read_csv('Mobile Reviews Sentiment Cleaned.csv')
    dfe      = pd.read_csv('Mobile Reviews Sentiment Encoded.csv')

    scaler      = StandardScaler()
    scaled_data = scaler.fit_transform(dfe)

    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    dfe['cluster']            = km.fit_predict(scaled_data)
    df_clean['cluster']       = dfe['cluster']

    cluster_labels = {0: 'Happy Users', 1: 'Unhappy Users', 2: 'Average Users'}
    df_clean['cluster_label'] = df_clean['cluster'].map(cluster_labels)

    return df_clean

df = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## Mobile Reviews")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["Overview", "Cluster Analysis", "Recommendation"])
st.sidebar.markdown("---")

# ── Helpers ───────────────────────────────────────────────────────────────────
def metric_card(label, value, sub=""):
    st.markdown(f"""
        <div class="metric-card">
            <h4>{label}</h4>
            <h2>{value}</h2>
            <span>{sub}</span>
        </div>""", unsafe_allow_html=True)

CLUSTER_ORDER  = ['Happy Users', 'Average Users', 'Unhappy Users']
CLUSTER_COLORS = {
    'Happy Users':   '#2e7d32',
    'Average Users': '#f9a825',
    'Unhappy Users': '#c62828'
}

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Mobile Reviews — Cluster Overview")
    st.markdown(
        "K-Means clustering (K = 3) applied on mobile phone reviews. "
        "Clusters are derived from rating, price, battery, camera, and performance features."
    )
    st.markdown("---")

    # Top KPIs
    counts = df['cluster_label'].value_counts()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Total Reviews", f"{df.shape[0]:,}", "Across all clusters")
    with col2:
        n = counts.get('Happy Users', 0)
        metric_card("Happy Users", f"{n:,}", f"{n / df.shape[0] * 100:.1f}% of data")
    with col3:
        n = counts.get('Average Users', 0)
        metric_card("Average Users", f"{n:,}", f"{n / df.shape[0] * 100:.1f}% of data")
    with col4:
        n = counts.get('Unhappy Users', 0)
        metric_card("Unhappy Users", f"{n:,}", f"{n / df.shape[0] * 100:.1f}% of data")

    st.markdown("---")

    # Cluster Summary Table
    st.markdown('<p class="section-title">Cluster-wise Average Summary</p>', unsafe_allow_html=True)
    summary = df.groupby('cluster_label').agg(
        Total_Reviews   = ('rating',              'count'),
        Avg_Rating      = ('rating',              'mean'),
        Avg_Price_USD   = ('price_usd',           'mean'),
        Avg_Battery     = ('battery_life_rating', 'mean'),
        Avg_Camera      = ('camera_rating',       'mean'),
        Avg_Performance = ('performance_rating',  'mean'),
    ).round(2).reindex(CLUSTER_ORDER)
    st.dataframe(summary, use_container_width=True)

    st.markdown("---")

    col_left, col_right = st.columns(2)

    # Cluster size bar chart
    with col_left:
        st.markdown('<p class="section-title">Cluster Size Comparison</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3))
        bars   = [counts.get(l, 0) for l in CLUSTER_ORDER]
        colors = [CLUSTER_COLORS[l] for l in CLUSTER_ORDER]
        ax.bar(CLUSTER_ORDER, bars, color=colors, edgecolor='white', width=0.5)
        ax.set_ylabel("Number of Reviews", fontsize=10)
        ax.set_title("Reviews per Cluster", fontsize=11, fontweight='bold')
        for i, v in enumerate(bars):
            ax.text(i, v + 200, f"{v:,}", ha='center', fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    # Feature comparison grouped bar
    with col_right:
        st.markdown('<p class="section-title">Average Metrics by Cluster</p>', unsafe_allow_html=True)
        metrics = ['Avg_Rating', 'Avg_Battery', 'Avg_Camera', 'Avg_Performance']
        labels  = ['Rating', 'Battery', 'Camera', 'Performance']
        x       = np.arange(len(labels))
        width   = 0.25
        fig, ax = plt.subplots(figsize=(5, 3))
        for i, label in enumerate(CLUSTER_ORDER):
            vals = summary.loc[label, metrics].values if label in summary.index else [0] * 4
            ax.bar(x + i * width, vals, width, label=label,
                   color=CLUSTER_COLORS[label], edgecolor='white')
        ax.set_xticks(x + width)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Average Score", fontsize=10)
        ax.set_title("Feature Comparison Across Clusters", fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")

    # Sentiment distribution per cluster
    st.markdown('<p class="section-title">Sentiment Distribution per Cluster</p>', unsafe_allow_html=True)
    sent_dist = (
        df.groupby(['cluster_label', 'sentiment'])
        .size()
        .unstack(fill_value=0)
        .reindex(CLUSTER_ORDER)
    )
    fig, ax = plt.subplots(figsize=(9, 3.5))
    sent_dist.plot(kind='bar', ax=ax, colormap='Set2', edgecolor='white', width=0.6)
    ax.set_xlabel("")
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Sentiment Breakdown per Cluster", fontsize=11, fontweight='bold')
    ax.legend(title="Sentiment", fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    plt.xticks(rotation=0, fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # Price distribution histogram
    st.markdown('<p class="section-title">Price Distribution by Cluster</p>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for label in CLUSTER_ORDER:
        subset = df[df['cluster_label'] == label]['price_usd'].dropna()
        ax.hist(subset, bins=40, alpha=0.6, label=label, color=CLUSTER_COLORS[label])
    ax.set_xlabel("Price (USD)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Price Distribution Across Clusters", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CLUSTER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Cluster Analysis":
    st.title("Cluster Deep-Dive Analysis")
    st.markdown(
        "Select a cluster to explore its detailed characteristics, brand breakdown, "
        "and behavioral pattern."
    )
    st.markdown("---")

    selected = st.selectbox("Select Cluster", CLUSTER_ORDER)
    c        = df[df['cluster_label'] == selected]
    color    = CLUSTER_COLORS[selected]

    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: metric_card("Total Reviews",  f"{c.shape[0]:,}")
    with col2: metric_card("Avg Rating",     f"{c['rating'].mean():.2f} / 5")
    with col3: metric_card("Avg Price",      f"${c['price_usd'].mean():.0f}")
    with col4: metric_card("Avg Battery",    f"{c['battery_life_rating'].mean():.2f}")
    with col5: metric_card("Avg Camera",     f"{c['camera_rating'].mean():.2f}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    # Top brands horizontal bar
    with col_left:
        st.markdown('<p class="section-title">Top 10 Brands by Review Count</p>', unsafe_allow_html=True)
        top_brands = c['brand'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(5, 4))
        top_brands[::-1].plot(kind='barh', ax=ax, color=color, edgecolor='white')
        ax.set_xlabel("Review Count", fontsize=10)
        ax.set_title(f"Top Brands — {selected}", fontsize=11, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    # Sentiment pie
    with col_right:
        st.markdown('<p class="section-title">Sentiment Split</p>', unsafe_allow_html=True)
        sent_counts = c['sentiment'].value_counts()
        pie_colors  = ['#2e7d32', '#f9a825', '#c62828']
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(
            sent_counts.values,
            labels=sent_counts.index,
            autopct='%1.1f%%',
            colors=pie_colors[:len(sent_counts)],
            startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        ax.set_title(f"Sentiment — {selected}", fontsize=11, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")

    col_left, col_right = st.columns(2)

    # Rating distribution
    with col_left:
        st.markdown('<p class="section-title">Rating Distribution</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        c['rating'].value_counts().sort_index().plot(
            kind='bar', ax=ax, color=color, edgecolor='white', width=0.6
        )
        ax.set_xlabel("Rating", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title("Rating Breakdown", fontsize=11, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

    # Price range metric cards
    with col_right:
        st.markdown('<p class="section-title">Price Range Summary</p>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            metric_card("Min Price",    f"${c['price_usd'].min():.0f}")
            metric_card("Avg Price",    f"${c['price_usd'].mean():.0f}")
        with col_b:
            metric_card("Median Price", f"${c['price_usd'].median():.0f}")
            metric_card("Max Price",    f"${c['price_usd'].max():.0f}")

    st.markdown("---")

    # Sub-rating vs overall comparison
    st.markdown(
        '<p class="section-title">Sub-Rating Averages vs Overall Dataset</p>',
        unsafe_allow_html=True
    )
    sub_cols    = ['battery_life_rating', 'camera_rating', 'performance_rating']
    sub_labels  = ['Battery', 'Camera', 'Performance']
    cluster_avg = [c[col].mean() for col in sub_cols]
    overall_avg = [df[col].mean() for col in sub_cols]

    x     = np.arange(len(sub_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(x - width / 2, cluster_avg, width, label=selected,    color=color,     edgecolor='white')
    ax.bar(x + width / 2, overall_avg, width, label='All Users',  color='#90a4ae', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(sub_labels, fontsize=10)
    ax.set_ylabel("Average Score", fontsize=10)
    ax.set_title(f"{selected} vs Overall Average", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    for i, (cv, ov) in enumerate(zip(cluster_avg, overall_avg)):
        ax.text(i - width / 2, cv + 0.02, f"{cv:.2f}", ha='center', fontsize=8)
        ax.text(i + width / 2, ov + 0.02, f"{ov:.2f}", ha='center', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    col_left, col_right = st.columns(2)

    # Top sources table
    with col_left:
        st.markdown('<p class="section-title">Top 5 Review Sources</p>', unsafe_allow_html=True)
        top_sources           = c['source'].value_counts().head(5).reset_index()
        top_sources.columns   = ['Source', 'Review Count']
        top_sources['Share %'] = (top_sources['Review Count'] / c.shape[0] * 100).round(1)
        st.dataframe(top_sources, use_container_width=True)

    # Sentiment count table
    with col_right:
        st.markdown('<p class="section-title">Sentiment Counts</p>', unsafe_allow_html=True)
        sent_tbl             = c['sentiment'].value_counts().reset_index()
        sent_tbl.columns     = ['Sentiment', 'Count']
        sent_tbl['Share %']  = (sent_tbl['Count'] / c.shape[0] * 100).round(1)
        st.dataframe(sent_tbl, use_container_width=True)

    st.markdown("---")

    # Cluster Interpretation Box
    st.markdown('<p class="section-title">Cluster Interpretation</p>', unsafe_allow_html=True)
    avg_r     = c['rating'].mean()
    avg_p     = c['price_usd'].mean()
    avg_bat   = c['battery_life_rating'].mean()
    avg_cam   = c['camera_rating'].mean()
    avg_perf  = c['performance_rating'].mean()
    top_sent  = c['sentiment'].value_counts().index[0]
    top_brand = c['brand'].value_counts().index[0]

    if selected == 'Happy Users':
        css  = "insight-happy"
        text = (
            f"This cluster represents <b>satisfied buyers</b> who gave high ratings (avg {avg_r:.2f}/5). "
            f"Users purchased phones averaging <b>${avg_p:.0f}</b> and scored battery life ({avg_bat:.2f}), "
            f"camera ({avg_cam:.2f}), and performance ({avg_perf:.2f}) above the overall average. "
            f"Dominant sentiment is <b>{top_sent}</b>, with <b>{top_brand}</b> being the most reviewed brand. "
            f"These users feel the phone delivered value for its price."
        )
    elif selected == 'Unhappy Users':
        css  = "insight-unhappy"
        text = (
            f"This cluster represents <b>dissatisfied buyers</b> who gave low ratings (avg {avg_r:.2f}/5). "
            f"Despite spending around <b>${avg_p:.0f}</b> on average, battery ({avg_bat:.2f}), "
            f"camera ({avg_cam:.2f}), and performance ({avg_perf:.2f}) were rated below the overall average. "
            f"Dominant sentiment is <b>{top_sent}</b>, with <b>{top_brand}</b> leading in review count. "
            f"These users felt the phone did not meet expectations for its price."
        )
    else:
        css  = "insight-avg"
        text = (
            f"This cluster represents <b>neutral buyers</b> with moderate ratings (avg {avg_r:.2f}/5). "
            f"Phones average <b>${avg_p:.0f}</b> with decent but unremarkable battery ({avg_bat:.2f}), "
            f"camera ({avg_cam:.2f}), and performance ({avg_perf:.2f}) scores. "
            f"Dominant sentiment is <b>{top_sent}</b>, with <b>{top_brand}</b> leading in review count. "
            f"These users had an average experience — neither very satisfied nor dissatisfied."
        )

    st.markdown(f'<div class="insight-box {css}">{text}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Sample records
    if 'review_text' in c.columns:
        st.markdown('<p class="section-title">Sample Reviews</p>', unsafe_allow_html=True)
        sample_cols = ['brand', 'model', 'rating', 'sentiment', 'source', 'price_usd', 'review_text']
    else:
        st.markdown('<p class="section-title">Sample Records</p>', unsafe_allow_html=True)
        sample_cols = ['brand', 'model', 'rating', 'sentiment', 'source', 'price_usd',
                       'battery_life_rating', 'camera_rating', 'performance_rating']

    sample_cols = [col for col in sample_cols if col in c.columns]
    st.dataframe(c[sample_cols].sample(8, random_state=42), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Recommendation":
    from sklearn.metrics.pairwise import cosine_similarity

    st.title("Phone Recommendation System")
    st.markdown(
        "Enter your preferred brand, budget price, and expected rating. "
        "The system uses Cosine Similarity on price and rating to find the most similar phones."
    )
    st.markdown("---")

    # ── Build Phone Profiles ───────────────────────────────────────────────────
    @st.cache_data
    def build_phone_profiles(dataframe):
        profiles = dataframe.groupby(['brand', 'model']).agg(
            price_usd=('price_usd', 'mean'),
            rating=('rating', 'mean')
        ).reset_index()
        profiles['price_usd'] = profiles['price_usd'].round(2)
        profiles['rating']    = profiles['rating'].round(2)
        return profiles

    phone_profiles = build_phone_profiles(df)

    # ── Scale price + rating ───────────────────────────────────────────────────
    rec_scaler     = StandardScaler()
    scaled_features = rec_scaler.fit_transform(phone_profiles[['price_usd', 'rating']])

    # ── User Inputs ────────────────────────────────────────────────────────────
    brand_options = sorted(phone_profiles['brand'].unique().tolist())

    st.markdown('<p class="section-title">Enter Your Preferences</p>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        user_brand  = st.selectbox("Brand", [""] + brand_options)
    with col_b:
        user_price  = st.number_input("Budget Price (USD)", min_value=100, max_value=2000,
                                      value=700, step=50)
    with col_c:
        user_rating = st.number_input("Expected Rating (1.0 - 5.0)", min_value=1.0, max_value=5.0,
                                      value=3.1, step=0.1, format="%.1f")

    top_n = st.slider("Number of Recommendations", min_value=3, max_value=10, value=5)

    st.markdown("")
    run_btn = st.button("Get Recommendations")

    if run_btn:
        st.markdown("---")

        # ── Cosine Similarity ──────────────────────────────────────────────────
        user_input  = pd.DataFrame([[user_price, user_rating]], columns=['price_usd', 'rating'])
        user_scaled = rec_scaler.transform(user_input)

        similarity_scores = cosine_similarity(user_scaled, scaled_features)[0]

        results = phone_profiles.copy()
        results['similarity_score'] = similarity_scores.round(4)

        # ── Brand Prioritisation ───────────────────────────────────────────────
        if user_brand:
            brand_match  = results['brand'].str.lower() == user_brand.lower()
            same_brand   = results[brand_match].sort_values('similarity_score', ascending=False)
            other_brands = results[~brand_match].sort_values('similarity_score', ascending=False)
            final_result = pd.concat([same_brand, other_brands]).head(top_n).reset_index(drop=True)
        else:
            final_result = results.sort_values('similarity_score', ascending=False).head(top_n).reset_index(drop=True)

        final_result.index      = final_result.index + 1
        final_result.index.name = 'rank'

        # ── Results Table ──────────────────────────────────────────────────────
        st.markdown('<p class="section-title">Recommended Phones</p>', unsafe_allow_html=True)
        st.dataframe(
            final_result[['brand', 'model', 'price_usd', 'rating', 'similarity_score']],
            use_container_width=True
        )

        st.markdown("---")

        # ── Visualizations ─────────────────────────────────────────────────────
        st.markdown('<p class="section-title">Recommendation Analysis</p>', unsafe_allow_html=True)

        viz_col1, viz_col2 = st.columns(2)

        # Chart 1: Similarity Score Bar Chart
        with viz_col1:
            st.markdown('<p class="section-title">Similarity Score by Phone</p>', unsafe_allow_html=True)
            phone_labels = (final_result['brand'] + " " + final_result['model']).tolist()
            sim_scores   = final_result['similarity_score'].tolist()

            bar_colors = []
            for b in final_result['brand']:
                if user_brand and b.lower() == user_brand.lower():
                    bar_colors.append('#1a73e8')
                else:
                    bar_colors.append('#90a4ae')

            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.barh(phone_labels[::-1], sim_scores[::-1], color=bar_colors[::-1],
                           edgecolor='white', height=0.55)
            ax.set_xlabel("Similarity Score", fontsize=10)
            ax.set_title("Cosine Similarity Score", fontsize=11, fontweight='bold')
            ax.set_xlim(min(sim_scores) - 0.05 if sim_scores else 0, 1.05)
            for bar, score in zip(bars, sim_scores[::-1]):
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{score:.4f}", va='center', fontsize=8)
            ax.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)

        # Chart 2: Price vs Rating Scatter
        with viz_col2:
            st.markdown('<p class="section-title">Price vs Rating — All Phones</p>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4))

            # All phones in grey
            ax.scatter(
                phone_profiles['price_usd'], phone_profiles['rating'],
                color='#cfd8dc', edgecolor='white', s=60, zorder=2, label='All Phones'
            )

            # Recommended phones highlighted
            rec_prices  = final_result['price_usd'].tolist()
            rec_ratings = final_result['rating'].tolist()
            rec_colors  = [
                '#1a73e8' if (user_brand and b.lower() == user_brand.lower()) else '#e53935'
                for b in final_result['brand']
            ]
            ax.scatter(rec_prices, rec_ratings, color=rec_colors,
                       edgecolor='white', s=100, zorder=3, label='Recommended')

            # User input point
            ax.scatter(user_price, user_rating, color='#f9a825', marker='*',
                       s=220, zorder=4, label='Your Input')

            # Labels for recommended phones
            for _, row in final_result.iterrows():
                ax.annotate(
                    row['model'],
                    (row['price_usd'], row['rating']),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=7, color='#1a1a2e'
                )

            ax.set_xlabel("Price (USD)", fontsize=10)
            ax.set_ylabel("Rating", fontsize=10)
            ax.set_title("Price vs Rating", fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("---")

        # Chart 3: Price Comparison Bar
        st.markdown('<p class="section-title">Price Comparison of Recommended Phones</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(9, 3.5))
        bar_labels  = (final_result['brand'] + "\n" + final_result['model']).tolist()
        bar_prices  = final_result['price_usd'].tolist()
        bar_colors2 = [
            '#1a73e8' if (user_brand and b.lower() == user_brand.lower()) else '#90a4ae'
            for b in final_result['brand']
        ]

        ax.bar(bar_labels, bar_prices, color=bar_colors2, edgecolor='white', width=0.5)
        ax.axhline(y=user_price, color='#f9a825', linestyle='--', linewidth=1.5,
                   label=f'Your Budget: ${user_price}')
        for i, v in enumerate(bar_prices):
            ax.text(i, v + 5, f"${v:.0f}", ha='center', fontsize=9)
        ax.set_ylabel("Price (USD)", fontsize=10)
        ax.set_title("Price of Recommended Phones vs Your Budget", fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)