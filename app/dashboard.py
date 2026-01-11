import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    f1_score,
    roc_auc_score,
    roc_curve,
    accuracy_score
)
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import prepare_dataset
from src.predict import load_trained_model
from src.utils import load_config, get_device, set_seed

st.set_page_config(page_title=" Tennis Momentum Predictor", layout="wide")
st.title(" 网球比赛动量预测系统 — 多模型性能对比")

# 固定随机种子确保可复现
set_seed(42)

def build_features_from_points(points, window_size=10, horizon=1):
    """简化版特征构建，用于单场比赛"""
    if len(points) <= window_size + horizon - 1:
        return None, None, None
    
    labels = [1 if p['point_char'] in ['S','A'] else 0 for p in points]
    serves = [p['server_id'] for p in points]
    is_ace_df = [1 if p['point_char'] in ['A','D'] else 0 for p in points]
    
    seq_X, scalar_X = [], []
    for i in range(window_size, len(labels) - horizon + 1):
        past_labels = labels[i-window_size:i]
        past_serves = serves[i-window_size:i]
        past_adf = is_ace_df[i-window_size:i]
        seq_feat = np.column_stack([past_labels, past_serves, past_adf])
        
        recent_wins = sum(labels[max(0, i-3):i])
        momentum = recent_wins / min(3, i)
        scalar_feat = np.array([momentum])
        
        seq_X.append(seq_feat)
        scalar_X.append(scalar_feat)
    
    return np.array(seq_X), np.array(scalar_X), None

def compute_sliding_momentum(points, window=5):
    labels = [1 if p['point_char'] in ['S','A'] else 0 for p in points]
    momentum = []
    for i in range(len(labels)):
        start = max(0, i - window + 1)
        rate = sum(labels[start:i+1]) / (i - start + 1)
        momentum.append(rate)
    return momentum

def compute_momentum_heatmap(points, window=5):
    """计算每分的动量，并映射到局和盘维度"""
    labels = [1 if p['point_char'] in ['S','A'] else 0 for p in points]
    momentum = []
    for i in range(len(labels)):
        start = max(0, i - window + 1)
        rate = sum(labels[start:i+1]) / (i - start + 1)
        momentum.append(rate)
    
    # 构建 DataFrame 带 set/game 信息
    data = []
    for i, p in enumerate(points):
        data.append({
            'set': p['set_idx'],
            'game': p['game_idx'],
            'point_in_game': p['score_in_game'],
            'momentum': momentum[i],
            'server_won': labels[i]
        })
    df = pd.DataFrame(data)
    
    # 聚合到 game level: 平均动量
    game_mom = df.groupby(['set', 'game'])['momentum'].mean().reset_index()
    pivot = game_mom.pivot(index='set', columns='game', values='momentum')
    return pivot

@st.cache_resource
def load_all_models(_config):
    """加载所有模型到内存中"""
    MODEL_LIST = ["RNN", "LSTM", "GRU", "Transformer"]
    model_dict = {}
    
    for model_name in MODEL_LIST:
        try:
            model = load_trained_model(model_name, _config)
            device = get_device(_config['train']['device'])
            model.to(device)
            model.eval()  # 设置为评估模式
            model_dict[model_name] = model
            st.sidebar.success(f"✅ {model_name} 模型加载成功")
        except Exception as e:
            st.sidebar.warning(f"⚠️ {model_name} 模型加载失败: {str(e)}")
    
    return model_dict

def predict_with_model(model, seq_X, scalar_X, device):
    """使用指定模型进行预测"""
    with torch.no_grad():
        seq_tensor = torch.tensor(seq_X, dtype=torch.float32).to(device)
        scal_tensor = torch.tensor(scalar_X, dtype=torch.float32).to(device)
        if len(seq_X.shape) == 2:
            seq_tensor = seq_tensor.unsqueeze(0)
        if len(scalar_X.shape) == 1:
            scal_tensor = scal_tensor.unsqueeze(0)
        preds = model(seq_tensor, scal_tensor).cpu().numpy().flatten()
    return preds

def evaluate_single_model(model_name, seq_X, scalar_X, y_true, config):
    try:
        model = load_trained_model(model_name, config)
        device = get_device(config['train']['device'])
        model.to(device)
        
        with torch.no_grad():
            seq_tensor = torch.tensor(seq_X, dtype=torch.float32).to(device)
            scal_tensor = torch.tensor(scalar_X, dtype=torch.float32).to(device)
            preds = model(seq_tensor, scal_tensor).cpu().numpy().flatten()
        
        y_flat = y_true.flatten() if y_true.ndim > 1 else y_true
        
        # 计算指标
        mae = mean_absolute_error(y_flat, preds)
        rmse = np.sqrt(mean_squared_error(y_flat, preds))
        acc = accuracy_score(y_flat, preds > 0.5)
        f1 = f1_score(y_flat, preds > 0.5)
        auc = roc_auc_score(y_flat, preds)
        
        return {
            'model': model_name,
            'preds': preds,
            'y_true': y_flat,
            'mae': mae,
            'rmse': rmse,
            'acc': acc,
            'f1': f1,
            'auc': auc
        }
    except Exception as e:
        st.warning(f"加载 {model_name} 失败: {str(e)}")
        return None

def plot_roc_curves(results):
    fig = go.Figure()
    for res in results:
        fpr, tpr, _ = roc_curve(res['y_true'], res['preds'])
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f"{res['model']} (AUC={res['auc']:.3f})",
            line=dict(width=2)
        ))
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig.update_layout(
        title="ROC 曲线对比",
        xaxis_title="假正率 (FPR)",
        yaxis_title="真正率 (TPR)",
        height=500
    )
    return fig

# ==================== 主界面 ====================
st.sidebar.header("配置")

# 模型选择
st.sidebar.subheader("模型选择")
selected_models = st.sidebar.multiselect(
    "选择要评估的模型",
    ["RNN", "LSTM", "GRU", "Transformer"],
    default=["LSTM", "GRU", "Transformer"]
)

sample_matches = st.sidebar.slider("用于评估的比赛数量", 10, 100, 50)
window_size = st.sidebar.slider("特征窗口大小", 5, 20, 10)
momentum_window = st.sidebar.slider("动量滑动窗口", 3, 10, 5)

# 更新配置
config = load_config()
config['data']['min_points'] = 30
config['feature']['window_size'] = window_size
config['dashboard']['momentum_window'] = momentum_window

# 加载所有选择的模型
with st.sidebar.expander("模型加载状态", expanded=False):
    if st.sidebar.button("重新加载模型"):
        st.cache_resource.clear()
    model_dict = load_all_models(config)

# 获取评估数据
df_all = load_and_clean_data(config)
eval_df = df_all.head(sample_matches).reset_index(drop=True)
seq_X, scalar_X, y_true = prepare_dataset(eval_df, config)

# 执行多模型评估（仅针对选择的模型）
st.subheader("模型性能对比")
if not selected_models:
    st.warning("请至少选择一个模型进行评估")
else:
    with st.spinner(f"正在评估 {len(selected_models)} 个模型..."):
        results = []
        for model_name in selected_models:
            if model_name in model_dict:
                res = evaluate_single_model(model_name, seq_X, scalar_X, y_true, config)
                if res:
                    results.append(res)
            else:
                st.warning(f"模型 {model_name} 未成功加载，跳过评估")

    if not results:
        st.error("未成功加载任何模型，请先运行训练脚本。")
    else:
        # === 性能指标表格 ===
        metrics_df = pd.DataFrame([
            {
                'Model': r['model'],
                'MAE ↓': f"{r['mae']:.4f}",
                'RMSE ↓': f"{r['rmse']:.4f}",
                'Accuracy ↑': f"{r['acc']:.4f}",
                'F1-Score ↑': f"{r['f1']:.4f}",
                'AUC ↑': f"{r['auc']:.4f}"
            }
            for r in results
        ])
        st.dataframe(metrics_df, use_container_width=True)

        # === ROC 曲线 ===
        st.plotly_chart(plot_roc_curves(results), use_container_width=True)

        # === 选择比赛进行详细可视化 ===
        st.subheader("🔍 单场比赛动量分析")
        
        # 选择比赛
        long_matches = df_all[df_all['n_points'] > 80].head(20).reset_index(drop=True)
        match_options = [
            f"{row['server1']} vs {row['server2']} (ID:{idx}, {row['n_points']}分)"
            for idx, row in long_matches.iterrows()
        ]
        selected_match = st.selectbox("选择比赛", match_options)
        
        if selected_match:
            idx = int(selected_match.split("ID:")[-1].split(",")[0].rstrip(')'))
            match_row = long_matches.iloc[idx]
            points = match_row['parsed_points']
            
            # 创建两列布局
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # 动量热力图
                heatmap_pivot = compute_momentum_heatmap(points, window=momentum_window)
                
                fig_heat = px.imshow(
                    heatmap_pivot.values,
                    labels=dict(x="局 (Game)", y="盘 (Set)", color="平均动量"),
                    x=[f"G{i}" for i in heatmap_pivot.columns],
                    y=[f"S{i}" for i in heatmap_pivot.index],
                    color_continuous_scale="RdYlBu_r",
                    aspect="auto"
                )
                fig_heat.update_layout(title=f"动量变化热力图：{match_row['server1']} vs {match_row['server2']}")
                st.plotly_chart(fig_heat, use_container_width=True)
            
            with col2:
                st.markdown("### 比赛信息")
                st.info(f"""
                **比赛ID:** {idx}
                **总分数:** {len(points)}
                **发球员1:** {match_row['server1']}
                **发球员2:** {match_row['server2']}
                **最大盘数:** {max(p['set_idx'] for p in points) + 1}
                **最大局数:** {max(p['game_idx'] for p in points) + 1}
                """)
            
            # 模型预测对比区域
            st.subheader("📊 模型预测对比")
            
            # 模型选择器
            available_models = [m for m in selected_models if m in model_dict]
            if not available_models:
                st.warning("没有可用的模型进行预测")
            else:
                # 创建多列布局用于模型选择
                cols = st.columns(len(available_models))
                selected_model_for_plot = None
                
                # 添加默认选择
                default_model = available_models[0]
                
                # 模型选择按钮
                for i, model_name in enumerate(available_models):
                    with cols[i]:
                        if st.button(f"📈 {model_name}", 
                                   use_container_width=True,
                                   type="primary" if model_name == default_model else "secondary"):
                            selected_model_for_plot = model_name
                
                # 如果还没有选择，使用默认
                if selected_model_for_plot is None:
                    selected_model_for_plot = default_model
                
                # 或者使用下拉选择
                selected_model_for_plot = st.selectbox(
                    "或使用下拉菜单选择模型",
                    available_models,
                    index=available_models.index(selected_model_for_plot)
                )
                
                # 构建该场比赛的特征
                X_seq, X_scal, _ = build_features_from_points(points, window_size, horizon=1)
                
                if X_seq is not None and selected_model_for_plot in model_dict:
                    # 获取设备
                    device = get_device(config['train']['device'])
                    
                    # 计算所有可用模型的预测（用于对比）
                    all_predictions = {}
                    for model_name in available_models:
                        if model_name in model_dict:
                            preds = predict_with_model(model_dict[model_name], X_seq, X_scal, device)
                            all_predictions[model_name] = preds
                    
                    # 计算实际动量
                    actual_momentum = compute_sliding_momentum(points, window=momentum_window)
                    
                    # 创建预测对比图
                    fig_pred = go.Figure()
                    
                    # 添加实际动量
                    fig_pred.add_trace(go.Scatter(
                        y=actual_momentum[window_size:],
                        mode='lines',
                        name='实际滑动胜率（动量）',
                        line=dict(color='black', width=3, dash='solid')
                    ))
                    
                    # 添加每个模型的预测
                    colors = ['blue', 'green', 'red', 'purple', 'orange']
                    for i, (model_name, preds) in enumerate(all_predictions.items()):
                        line_style = dict(width=3) if model_name == selected_model_for_plot else dict(width=2, dash='dash')
                        color = colors[i % len(colors)]
                        
                        fig_pred.add_trace(go.Scatter(
                            y=preds,
                            mode='lines',
                            name=f"{model_name} 预测",
                            line=dict(color=color, **line_style),
                            opacity=1.0 if model_name == selected_model_for_plot else 0.6
                        ))
                    
                    # 添加中线
                    fig_pred.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
                    
                    # 更新布局
                    fig_pred.update_layout(
                        title=f"模型预测对比 - 当前选择: {selected_model_for_plot}",
                        xaxis_title="逐分序号",
                        yaxis_title="概率 / 动量",
                        height=500,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # 添加模型预测统计信息
                    st.subheader("📈 模型预测统计")
                    
                    # 创建统计表格
                    stats_data = []
                    for model_name, preds in all_predictions.items():
                        stats_data.append({
                            'Model': model_name,
                            'Mean Prediction': f"{np.mean(preds):.3f}",
                            'Std Prediction': f"{np.std(preds):.3f}",
                            'Max Prediction': f"{np.max(preds):.3f}",
                            'Min Prediction': f"{np.min(preds):.3f}",
                            '>0.5 Ratio': f"{np.mean(preds > 0.5):.3f}"
                        })
                    
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df, use_container_width=True)
                    
                    # 添加预测差异分析
                    st.subheader("🔍 模型间预测差异")
                    
                    if len(all_predictions) > 1:
                        # 计算模型间的相关系数矩阵
                        model_names = list(all_predictions.keys())
                        corr_matrix = np.zeros((len(model_names), len(model_names)))
                        
                        for i, model_i in enumerate(model_names):
                            for j, model_j in enumerate(model_names):
                                corr_matrix[i, j] = np.corrcoef(
                                    all_predictions[model_i],
                                    all_predictions[model_j]
                                )[0, 1]
                        
                        # 绘制相关系数热力图
                        fig_corr = px.imshow(
                            corr_matrix,
                            labels=dict(color="相关系数"),
                            x=model_names,
                            y=model_names,
                            color_continuous_scale="RdBu",
                            zmin=-1,
                            zmax=1,
                            text_auto=True
                        )
                        fig_corr.update_layout(
                            title="模型预测相关系数矩阵",
                            height=400
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.warning("无法为该场比赛构建特征或模型不可用")