import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import glob

# --- 1. 页面配置 ---
st.set_page_config(
    page_title="中老年人视力障碍风险预测系统",
    page_icon="👓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS 样式
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .stSelectbox label, .stNumberInput label {
        font-weight: bold;
        color: #1f1f1f;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 资源加载 ---
@st.cache_resource
def load_assets():
    try:
        model_files = glob.glob('model_assets/best_model*.pkl')
        if not model_files:
            return None, None, None, "未找到模型文件 (.pkl)"
        
        model = joblib.load(model_files[0])
        scaler = joblib.load('model_assets/scaler.pkl')
        encoders = joblib.load('model_assets/label_encoders.pkl')
        
        with open('model_assets/feature_list.txt', 'r', encoding='utf-8') as f:
            features = [line.strip().split('. ')[1] for line in f.readlines() if '. ' in line]
        return model, scaler, encoders, features
    except Exception as e:
        return None, None, None, str(e)

model, scaler, encoders, feature_list = load_assets()

# --- 3. 映射字典定义 ---
EDU_MAP = {"1": "高中及以上", "2": "中学", "3": "小学", "4": "文盲/半文盲"}
RURAL_MAP = {"1": "城市", "2": "农村"}
BINARY_MAP = {"0": "否", "1": "是"}
HAS_MAP = {"0": "无", "1": "有"}
HEAR_MAP = {"0": "正常", "1": "听力障碍"}

# 根据 SHAP 图确定的核心指标（包含用户要求的变量）
TOP_15_FEATURES = [
    'hear', 'province', 'age', 'edu', 'total_cognition', 
    'rural', 'fcamt', 'executive', 'memeory', 'pension', 
    'mweight', 'social_total', 'da042s_total', 'income_total', 'mheight'
]

# --- 4. 页面主体 ---
st.title("👓 中老年人视力障碍风险预测系统")

if model is None:
    st.error(f"❌ 资源加载失败。错误: {feature_list}")
    st.stop()

# --- 5. 模式选择 ---
st.subheader("第一步：选择筛查模式")
mode = st.selectbox(
    "请选择适合您的筛查版本：",
    options=["请选择...", "精简版 (15个核心指标)", "完整版 (全量指标)"],
    index=0
)

if mode == "请选择...":
    st.warning("👈 请在上方下拉框中选择一个版本以开始。")
    st.stop()

st.markdown("---")
st.subheader("第二步：录入受试者数据")

user_inputs = {}
is_simplified = "精简版" in mode

# 选项卡布局
tab1, tab2, tab3 = st.tabs(["基本人口学", "生理与感官", "认知、社会与支持"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        user_inputs['age'] = st.number_input("年龄 (age)", 45, 120, 65)
        user_inputs['province'] = st.number_input("出生地区代码 (province)", 0, 100, 1)
        user_inputs['rural'] = st.selectbox("居住环境 (rural)", options=["1", "2"], format_func=lambda x: RURAL_MAP[x])
    with col2:
        user_inputs['edu'] = st.selectbox("教育情况 (edu)", options=["1", "2", "3", "4"], format_func=lambda x: EDU_MAP[x])
        user_inputs['pension'] = st.selectbox("退休/养老金状况 (pension)", options=["0", "1"], format_func=lambda x: HAS_MAP[x])

with tab2:
    col3, col4 = st.columns(2)
    with col3:
        user_inputs['hear'] = st.selectbox("听力障碍 (hear)", options=["0", "1"], format_func=lambda x: HEAR_MAP[x])
        user_inputs['mweight'] = st.number_input("体重 (kg)", 30.0, 150.0, 65.0)
        user_inputs['mheight'] = st.number_input("身高 (cm)", 100.0, 220.0, 165.0)
    with col4:
        # 修改：da042s_total 改为 疼痛评分（部位），范围 0-15
        user_inputs['da042s_total'] = st.slider("疼痛评分 (部位) (da042s_total)", 0, 15, 0)

with tab3:
    col5, col6 = st.columns(2)
    with col5:
        # 修改：认知能力 (0-21)，执行力 (0-11)，记忆力 (0-9.5)
        user_inputs['total_cognition'] = st.slider("认知能力总分 (0-21)", 0, 21, 15)
        user_inputs['executive'] = st.slider("心智执行力 (0-11)", 0, 11, 5)
        user_inputs['memeory'] = st.slider("记忆能力 (0-9.5)", 0.0, 9.5, 5.0, step=0.5)
    with col6:
        # 修改：fcamt 和 tcamt (此处用 income_total 占位或若模型包含 tcamt 请替换) 变成 1/0
        user_inputs['fcamt'] = st.selectbox("是否有子女经济支持 (fcamt)", options=["0", "1"], format_func=lambda x: HAS_MAP[x])
        
        # 针对您提到的 tcamt，如果在 feature_list 中则录入，否则录入模型需要的 income_total
        if 'tcamt' in feature_list:
            user_inputs['tcamt'] = st.selectbox("是否有转移收入 (tcamt)", options=["0", "1"], format_func=lambda x: HAS_MAP[x])
        else:
            user_inputs['income_total'] = st.number_input("家庭年总收入 (元)", 0, 500000, 20000)
            
        # 修改：社交评分 (0-9)
        user_inputs['social_total'] = st.slider("社交评分 (0-9)", 0, 9, 5)

# 如果是完整版，展示其余变量
if not is_simplified:
    with st.expander("更多详细指标 (完整版选填)"):
        remaining_features = [f for f in feature_list if f not in user_inputs]
        cols = st.columns(3)
        for idx, feat in enumerate(remaining_features):
            user_inputs[feat] = cols[idx % 3].number_input(f"{feat}", value=0.0)

# --- 6. 侧边栏配置 ---
with st.sidebar:
    st.header("⚙️ 系统配置")
    optimal_threshold = st.number_input("风险判断阈值", 0.1, 0.9, 0.45, 0.01)
    st.divider()
    st.markdown("### 更新说明")
    st.caption("1. 认知/执行/记忆量表评分范围已更新。")
    st.caption("2. 疼痛评分更名为'部位评分'，范围 0-15。")
    st.caption("3. 经济支持类指标已转为二元(有/无)输入。")

# --- 7. 预测执行 ---
st.markdown("---")
if st.button("🚀 开始风险评估"):
    with st.status("正在分析数据...", expanded=True) as status:
        # 数据对齐
        final_data = {feat: user_inputs.get(feat, 0) for feat in feature_list}
        input_df = pd.DataFrame([final_data])[feature_list]
        
        # 编码转换
        for col, le in encoders.items():
            if col in input_df.columns:
                val = str(input_df[col].values[0])
                input_df[col] = le.transform([val])[0] if val in le.classes_ else 0
        
        # 缩放与预测
        input_scaled = scaler.transform(input_df)
        prob = model.predict_proba(input_scaled)[:, 1][0]
        is_high_risk = prob >= optimal_threshold
        status.update(label="评估完成！", state="complete", expanded=False)

    # --- 8. 结果展示 ---
    st.subheader("🔮 预测评估报告")
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.metric(label="视力障碍患病概率", value=f"{prob:.2%}")
        if is_high_risk:
            st.error("结论：高风险人群")
        else:
            st.success("结论：低风险人群")

    with res_col2:
        st.write("#### 风险程度")
        st.progress(prob)
        st.caption(f"当前阈值设定为: {optimal_threshold}")

st.markdown("---")
st.caption("© 2025 牡丹江医科大学护理学院 | 仅供科研参考")
