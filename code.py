import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, mean_absolute_error

HEAVY_TAIL_FEATURES = ["Area", "ConvexArea", "Perimeter", "MajorAxisLength", "MinorAxisLength", "EquivDiameter"]

# ==========================================
# CẤU HÌNH TRANG VÀ GIAO DIỆN
# ==========================================
st.set_page_config(page_title="Dry Bean Predictor", page_icon="🫘", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f7fcf6 0%, #eef7ea 100%);
    }
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #1b5e20;
        text-align: center;
        letter-spacing: 0.5px;
    }
    .main-subtitle {
        font-size: 16px;
        color: #4f6f52;
        text-align: center;
        margin-top: 8px;
        margin-bottom: 16px;
    }
    .sub-title {
        font-size: 24px;
        font-weight: 700;
        color: #1565c0;
        margin-top: 18px;
        margin-bottom: 10px;
    }
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f5fbf3 100%);
        padding: 16px;
        border-radius: 14px;
        border: 1px solid #dcecdc;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    }
    .stButton > button {
        border-radius: 10px;
        padding: 0.6rem 1rem;
        font-weight: 600;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #173c24 0%, #265e39 100%);
    }
    div[data-testid="stSidebar"] .st-emotion-cache-16txtl3 {
        color: #f5fff5;
    }
    </style>
""", unsafe_allow_html=True)

# Header section
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("<div style='font-size: 42px; text-align: center;'>🌱</div>", unsafe_allow_html=True)
with col_title:
    st.markdown('<div class="main-title">DRY BEAN CLASSIFICATION & ANALYTICS PLATFORM</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Predict bean variety and compare machine learning models through a polished dashboard.</div>', unsafe_allow_html=True)

# ==========================================
# LOAD DỮ LIỆU VÀ HUẤN LUYỆN MÔ HÌNH (CACHE)
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_excel('Dry_Bean_Dataset.xlsx')
    return df

@st.cache_resource
def prepare_and_train_models(df):
    df_model = df.copy()
    for col in HEAVY_TAIL_FEATURES:
        if col in df_model.columns:
            df_model[col] = np.log1p(df_model[col])

    X = df_model.drop(columns=['Class'])
    y = df_model['Class']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # --- Feature Selection (Loại bỏ các feature có correlation > 0.9) ---
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    X_selected = X.drop(columns=to_drop)
    selected_features_list = X_selected.columns.tolist()
    
    # Định nghĩa hàm train để tái sử dụng
    def train_evaluate(X_data):
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_encoded, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        
        results = {}
        trained_models = {}
        for name, model in models.items():
            # Đo thời gian Train
            start_train = time.time()
            model.fit(X_train_sc, y_train)
            train_time = (time.time() - start_train) * 1000 # ms
            
            # Đo thời gian Test
            start_test = time.time()
            y_pred = model.predict(X_test_sc)
            test_time = (time.time() - start_test) * 1000 # ms
            
            # Tính metrics
            acc = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
            
            results[name] = {
                'Accuracy': acc, 'Precision': precision, 'Recall': recall, 
                'Weighted F1': f1, 'Train Time (ms)': train_time, 'Test Time (ms)': test_time
            }
            trained_models[name] = model
            
        return trained_models, scaler, results

    # Train cho 2 chế độ
    models_full, scaler_full, res_full = train_evaluate(X)
    models_sel, scaler_sel, res_sel = train_evaluate(X_selected)
    
    return X.columns.tolist(), selected_features_list, le, models_full, scaler_full, res_full, models_sel, scaler_sel, res_sel

df = load_data()
full_features, selected_features, label_encoder, models_full, scaler_full, res_full, models_sel, scaler_sel, res_sel = prepare_and_train_models(df)

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("Navigation")
st.sidebar.markdown("Choose a module to explore the app.")
menu = st.sidebar.radio("Select a section:", ["🏠 Home", "📊 About Dataset", "🔍 Feature Selection"])

# ==========================================
# TRANG 1: HOME (DỰ ĐOÁN & SO SÁNH)
# ==========================================
if menu == "🏠 Home":
    st.markdown('<div class="sub-title">Prediction Configuration</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        mode = st.radio("Select feature mode:", ["Full Feature", "After Feature Selection (Correlation)"])
    with col2:
        model_choice = st.selectbox("Choose a prediction model:", ["Logistic Regression", "Decision Tree", "KNN"])
        
    # Xác định các biến dựa trên chế độ
    current_features = full_features if mode == "Full Feature" else selected_features
    current_models = models_full if mode == "Full Feature" else models_sel
    current_scaler = scaler_full if mode == "Full Feature" else scaler_sel
    current_results = res_full if mode == "Full Feature" else res_sel
    
    st.markdown("---")
    st.markdown("### 📝 Enter Bean Feature Values")
    
    # Form nhập liệu tự động tạo các ô input dựa trên số lượng features
    input_data = {}
    cols = st.columns(4)
    for i, feature in enumerate(current_features):
        mean_val = float(df[feature].mean())
        # Tạo number_input với giá trị mặc định là giá trị trung bình của tập dữ liệu
        input_data[feature] = cols[i % 4].number_input(f"{feature}", value=mean_val, format="%.4f")
        
    if st.button("🚀 Predict Result", use_container_width=True, type="primary"):
        # 1. Thực hiện dự đoán
        input_df = pd.DataFrame([input_data])
        for col in HEAVY_TAIL_FEATURES:
            if col in input_df.columns:
                input_df[col] = np.log1p(input_df[col])
        input_scaled = current_scaler.transform(input_df)
        pred_encoded = current_models[model_choice].predict(input_scaled)
        pred_class = label_encoder.inverse_transform(pred_encoded)[0]
        
        st.success(f"### 🎉 The predicted bean class is: **{pred_class}**")
        st.markdown("---")
        
        # 2. Hiển thị đồ thị so sánh các độ đo
        st.markdown('<div class="sub-title">Performance Comparison: {} vs Other Models ({})</div>'.format(model_choice, mode), unsafe_allow_html=True)
        
        # Biến đổi dict kết quả thành DataFrame để vẽ biểu đồ
        res_df = pd.DataFrame(current_results).T.reset_index().rename(columns={'index': 'Model'})
        
        # Đánh dấu màu cho mô hình được chọn
        colors = ['#EF4444' if m == model_choice else '#9CA3AF' for m in res_df['Model']]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'Weighted F1', 'Train Time (ms)', 'Test Time (ms)']
        
        for i, metric in enumerate(metrics_to_plot):
            sns.barplot(data=res_df, x='Model', y=metric, ax=axes[i], palette=colors)
            axes[i].set_title(f"Comparison of {metric}", fontweight='bold')
            axes[i].set_ylabel(metric)
            axes[i].set_xlabel("")
            # Thêm text giá trị lên đầu cột
            for p in axes[i].patches:
                axes[i].annotate(f"{p.get_height():.4f}" if "Time" not in metric else f"{p.get_height():.2f}", 
                                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                                 ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points')
                
        plt.tight_layout()
        st.pyplot(fig)

# ==========================================
# TRANG 2: FEATURE SELECTION
# ==========================================
elif menu == "🔍 Feature Selection":
    st.markdown('<div class="sub-title">Feature Selection</div>', unsafe_allow_html=True)
    st.markdown("""
    Feature selection is the process of removing redundant or weak features so the model becomes simpler,
    less affected by multicollinearity, and often faster to train. In this section, we use a correlation-based
    method to identify highly correlated features and compare model performance before and after selection.
    """)

    st.markdown('<div class="sub-title">1. Correlation Analysis</div>', unsafe_allow_html=True)
    corr_matrix = df.drop(columns=['Class']).corr().abs()
    fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=ax_corr)
    ax_corr.set_title('Correlation Matrix of Features')
    st.pyplot(fig_corr)

    threshold = 0.85
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    selected_features_fs = [col for col in df.columns if col != 'Class' and col not in to_drop]

    st.markdown(f"**Correlation threshold:** {threshold}")
    st.markdown(f"**Number of original features:** {len(df.columns) - 1}")
    st.markdown(f"**Number of features kept after selection:** {len(selected_features_fs)}")
    if to_drop:
        st.markdown(f"**Removed features:** {', '.join(to_drop)}")
    else:
        st.markdown("**Removed features:** None")

    st.markdown('<div class="sub-title">2. Top Highly Correlated Pairs</div>', unsafe_allow_html=True)
    pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            value = corr_matrix.iloc[i, j]
            if value > threshold:
                pairs.append((corr_matrix.index[i], corr_matrix.columns[j], value))

    if pairs:
        pairs_df = pd.DataFrame(pairs, columns=['Feature 1', 'Feature 2', 'Correlation'])
        pairs_df = pairs_df.sort_values('Correlation', ascending=False)
        top_pairs = pairs_df.head(10)
        fig_pairs, ax_pairs = plt.subplots(figsize=(10, 5))
        sns.barplot(data=top_pairs, x='Correlation', y=top_pairs['Feature 1'] + ' - ' + top_pairs['Feature 2'], ax=ax_pairs, color='teal')
        ax_pairs.set_title('Top Highly Correlated Feature Pairs')
        ax_pairs.set_xlabel('Correlation')
        ax_pairs.set_ylabel('Feature Pair')
        st.pyplot(fig_pairs)
    else:
        st.info('No highly correlated pairs were found above the selected threshold.')

    st.markdown('<div class="sub-title">3. Regression Experiment with Linear Regression (MAE)</div>', unsafe_allow_html=True)
    st.markdown("""
    Because the document asks for a comparison using Linear Regression and MAE, the most suitable approach is to switch to a regression task
    with a continuous target such as Area or Perimeter. Different feature subsets from the feature-selection step are then tested and compared by MAE.
    """)

    df_reg = df.copy()
    for col in HEAVY_TAIL_FEATURES:
        if col in df_reg.columns:
            df_reg[col] = np.log1p(df_reg[col])

    target_col = st.selectbox("Select a continuous target for regression:", ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "EquivDiameter"])

    corr_thresholds = [0.85, 0.90]
    corr_sets = {}
    for threshold in corr_thresholds:
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        drop_cols = [col for col in upper.columns if any(upper[col] > threshold)]
        corr_sets[f"Correlation > {threshold}"] = [c for c in df_reg.columns if c not in ['Class', target_col] and c not in drop_cols]

    feature_sets = {
        'Full Features': [c for c in df_reg.columns if c not in ['Class', target_col]],
        **corr_sets,
    }

    def evaluate_regression(feature_cols, label):
        X = df_reg[feature_cols]
        y = df_reg[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)

        return pd.DataFrame([{
            'Scenario': label,
            'Target': target_col,
            'Num Features': len(feature_cols),
            'MAE': mae,
        }])

    regression_results = []
    for label, feature_cols in feature_sets.items():
        regression_results.append(evaluate_regression(feature_cols, label))

    regression_df = pd.concat(regression_results, ignore_index=True).sort_values('MAE')
    st.dataframe(regression_df, use_container_width=True)

    fig_mae, ax_mae = plt.subplots(figsize=(10, 5))
    sns.barplot(data=regression_df, x='Scenario', y='MAE', palette='viridis', ax=ax_mae)
    ax_mae.set_title(f'MAE Comparison for Different Feature Sets (Target: {target_col})')
    ax_mae.set_xlabel('Feature Set')
    ax_mae.set_ylabel('MAE')
    ax_mae.tick_params(axis='x', rotation=20)
    st.pyplot(fig_mae)

    st.markdown('<div class="sub-title">4. Model Comparison Before vs After Feature Selection</div>', unsafe_allow_html=True)

    def evaluate_feature_set(feature_cols, label):
        X = df[feature_cols]
        y = df['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        models = {
            'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=7),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
        }

        rows = []
        for model_name, model in models.items():
            start = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = (time.perf_counter() - start) * 1000

            start_test = time.perf_counter()
            y_pred = model.predict(X_test)
            test_time = (time.perf_counter() - start_test) * 1000

            report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report_dict).transpose()

            rows.append({
                'Scenario': label,
                'Model': model_name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Weighted Precision': report_df.loc['weighted avg', 'precision'],
                'Weighted Recall': report_df.loc['weighted avg', 'recall'],
                'Weighted F1': report_df.loc['weighted avg', 'f1-score'],
                'Train Time (ms)': train_time,
                'Test Time (ms)': test_time,
                'Num Features': len(feature_cols),
            })

        return pd.DataFrame(rows)

    all_features = [col for col in df.columns if col != 'Class']
    full_metrics = evaluate_feature_set(all_features, 'Full Features')
    selected_metrics = evaluate_feature_set(selected_features_fs, 'Selected Features')
    comparison = pd.concat([full_metrics, selected_metrics], ignore_index=True)

    comparison_display = comparison[['Scenario', 'Model', 'Accuracy', 'Weighted F1', 'Train Time (ms)', 'Test Time (ms)', 'Num Features']]
    st.dataframe(comparison_display, use_container_width=True)

    plot_df = comparison[['Scenario', 'Model', 'Accuracy', 'Weighted F1', 'Train Time (ms)', 'Test Time (ms)']].copy()
    plot_df = plot_df.melt(id_vars=['Scenario', 'Model'], var_name='Metric', value_name='Value')

    g = sns.catplot(
        data=plot_df,
        x='Model',
        y='Value',
        hue='Scenario',
        col='Metric',
        kind='bar',
        col_wrap=2,
        height=4,
        aspect=1.2,
        sharey=False,
    )
    g.set_titles('{col_name}')
    g.set_xticklabels(rotation=15)
    plt.tight_layout()
    st.pyplot(g.fig)

    st.markdown('<div class="sub-title">4. Interpretation</div>', unsafe_allow_html=True)
    st.markdown("""
    If the selected feature set keeps similar accuracy or F1-score while reducing training time,
    then feature selection is beneficial. If performance drops significantly, the removed features may still be informative.
    """)

# ==========================================
# TRANG 3: ABOUT DATASET
# ==========================================
elif menu == "📊 About Dataset":
    st.markdown('<div class="sub-title">1. Data Schema & Missing Values</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.write("Data Types:")
        st.dataframe(df.dtypes.astype(str).rename("Data Type"), use_container_width=True)
    with c2:
        st.write("Missing Values:")
        st.dataframe(df.isnull().sum().rename("Null Count"), use_container_width=True)
        
    st.markdown('<div class="sub-title">2. Descriptive Statistics</div>', unsafe_allow_html=True)
    st.write(f"Dataset size: **{df.shape[0]} rows, {df.shape[1]} columns**")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.markdown('<div class="sub-title">3. Correlation Matrix</div>', unsafe_allow_html=True)
    fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
    numeric_df = df.drop(columns=['Class'])
    sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)
    
    st.markdown('<div class="sub-title">4. Numeric Explorer</div>', unsafe_allow_html=True)
    selected_num_col = st.selectbox("Select a feature to inspect its distribution:", numeric_df.columns)
    
    fig_exp, (ax_box, ax_hist) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Biểu đồ Boxplot theo Class
    sns.boxplot(data=df, x='Class', y=selected_num_col, ax=ax_box, palette='Set2')
    ax_box.set_title(f"Boxplot of {selected_num_col} by Class")
    ax_box.tick_params(axis='x', rotation=45)
    
    # Biểu đồ Histogram
    sns.histplot(data=df, x=selected_num_col, hue='Class', kde=True, ax=ax_hist, palette='Set2')
    ax_hist.set_title(f"Distribution of {selected_num_col}")
    
    st.pyplot(fig_exp)