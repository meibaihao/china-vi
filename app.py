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
HEAR_MAP = {"0": "正常", "1": "听力障碍"}

# --- 重点：根据 SHAP 图更新的前 15 指标 ---
TOP_15_FEATURES = [
    'hear', 'province', 'age', 'edu', 'total_cognition', 
    'rural', 'fcamt', 'executive', 'memeory', 'pension', 
    'mweight', 'social_total', 'da042s_total', 'income_total', 'mheight'
]

# --- 4. 页面主体 ---
st.title("👓 中老年人视力障碍风险预测系统")
st.info("本系统已根据 SHAP 解释性分析更新，优先采用对预测结果影响最显著的 15 项核心指标。")

if model is None:
    st.error(f"❌ 资源加载失败。请检查路径。错误: {feature_list}")
    st.stop()

# --- 5. 模式选择 ---
st.subheader("第一步：选择筛查模式")
mode = st.selectbox(
    "请选择适合您的筛查版本：",
    options=["请选择...", "精简版 (基于 SHAP 核心 15 指标)", "完整版 (全量指标预测)"],
    index=0
)

if mode == "请选择...":
    st.warning("👈 请在上方下拉框中选择一个版本以开始录入数据。")
    st.stop()

st.markdown("---")
st.subheader("第二步：录入受试者数据")

user_inputs = {}
is_simplified = "精简版" in mode

# 选项卡布局：根据新的 15 指标重新组织
tab1, tab2, tab3 = st.tabs(["人口学与背景", "生理与感官", "认知与社会经济"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        user_inputs['age'] = st.number_input("年龄 (age)", 45, 120, 65)
        user_inputs['province'] = st.number_input("出生地区代码 (province)", 0, 100, 1)
        user_inputs['rural'] = st.selectbox("居住环境 (rural)", options=["1", "2"], format_func=lambda x: RURAL_MAP[x])
    with col2:
        user_inputs['edu'] = st.selectbox("教育情况 (edu)", options=["1", "2", "3", "4"], format_func=lambda x: EDU_MAP[x])
        user_inputs['pension'] = st.selectbox("退休/养老金状况 (pension)", options=["0", "1"], format_func=lambda x: "无" if x=="0" else "有")

with tab2:
    col3, col4 = st.columns(2)
    with col3:
        user_inputs['hear'] = st.selectbox("听力障碍 (hear)", options=["0", "1"], format_func=lambda x: HEAR_MAP[x])
        user_inputs['mweight'] = st.number_input("体重 (kg) (mweight)", 30.0, 150.0, 65.0)
    with col4:
        user_inputs['mheight'] = st.number_input("身高 (cm) (mheight)", 100.0, 220.0, 165.0)
        user_inputs['da042s_total'] = st.slider("疼痛/身体不适评分 (da042s_total)", 0, 50, 5)

with tab3:
    col5, col6 = st.columns(2)
    with col5:
        user_inputs['total_cognition'] = st.slider("总认知能力 (total_cognition)", 0, 40, 25)
        user_inputs['executive'] = st.slider("心智执行力 (executive)", 0, 20, 10)
        user_inputs['memeory'] = st.slider("记忆能力 (memeory)", 0, 20, 10)
    with col6:
        user_inputs['fcamt'] = st.number_input("子女经济支持金额 (fcamt)", 0, 100000, 1000)
        user_inputs['income_total'] = st.number_input("家庭总收入 (income_total)", 0, 500000, 20000)
        user_inputs['social_total'] = st.slider("社会交往评分 (social_total)", 0, 100, 50)

# 如果是完整版，展示其余变量
if not is_simplified:
    with st.expander("更多详细指标 (完整版选填)"):
        st.caption("以下特征将使用默认值填充：")
        remaining_features = [f for f in feature_list if f not in user_inputs]
        cols = st.columns(3)
        for idx, feat in enumerate(remaining_features):
            user_inputs[feat] = cols[idx % 3].number_input(f"{feat}", value=0.0)

# --- 6. 侧边栏配置 ---
with st.sidebar:
    st.header("⚙️ 系统配置")
    st.info(f"当前模式: {mode.split('(')[0]}")
    st.divider()
    optimal_threshold = st.number_input("风险判断阈值", 0.1, 0.9, 0.45, 0.01)
    st.markdown("---")
    st.markdown("### SHAP 特征重要性说明")
    st.caption("图中显示听力障碍、地区、年龄和教育程度是该模型最重要的四个预测因子。")

# --- 7. 预测执行 ---
st.markdown("---")
if st.button("🚀 开始风险评估"):
    with st.status("正在进行 AI 模型推理...", expanded=True) as status:
        st.write("数据对齐中...")
        final_data = {feat: user_inputs.get(feat, 0) for feat in feature_list}
        input_df = pd.DataFrame([final_data])[feature_list]
        
        st.write("标签编码与特征缩放...")
        for col, le in encoders.items():
            if col in input_df.columns:
                val = str(input_df[col].values[0])
                input_df[col] = le.transform([val])[0] if val in le.classes_ else 0
        
        input_scaled = scaler.transform(input_df)
        prob = model.predict_proba(input_scaled)[:, 1][0]
        is_high_risk = prob >= optimal_threshold
        status.update(label="评估完成！", state="complete", expanded=False)

    # --- 8. 结果展示 ---
    st.subheader("🔮 预测评估报告")
    c_res1, c_res2 = st.columns([1, 2])
    
    with c_res1:
        st.metric(label="视力障碍风险概率", value=f"{prob:.2%}")
        if is_high_risk:
            st.error("结论：高风险人群")
        else:
            st.success("结论：低风险人群")

    with c_res2:
        st.write("#### 风险可视化")
        st.progress(prob)
        st.caption(f"决策边界：{optimal_threshold:.2f} | 建议：{'请及时就医检查' if is_high_risk else '定期体检即可'}")

# --- 9. 底部说明 ---
with st.expander("🔬 SHAP 模型原理图解"):
    st.markdown("""
    ### 为什么选择这 15 个指标？
        
    我们通过 **SHAP (SHapley Additive exPlanations)** 方法对梯度提升模型进行了归因分析：
    - **横轴 (SHAP Value)**: 右侧点表示该因素增加了患病风险，左侧表示降低风险。
    - **颜色 (Feature Value)**: 红色代表该指标数值较高，蓝色代表数值较低。
    - **例如 `hear`**: 顶部的红色簇聚集在右侧，说明有听力障碍的人群患视力障碍的风险显著升高。
    """)

st.markdown("---")
st.caption("© 2025 牡丹江医科大学护理学院 | 仅供科研参考")
