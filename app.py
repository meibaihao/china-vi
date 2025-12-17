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
        # 自动搜索模型文件
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

# --- 3. 映射字典定义 (确保后台传输正确数值) ---
EDU_MAP = {"1": "高中及以上", "2": "中学", "3": "小学", "4": "文盲/半文盲"}
GENDER_MAP = {"1": "男", "2": "女"}
RURAL_MAP = {"1": "城市", "2": "农村"}
MARRY_MAP = {"1": "已婚", "2": "未婚", "离异/丧偶"}
BINARY_MAP = {"0": "否", "1": "是"}

# 精简版核心 15 指标
TOP_15_FEATURES = [
    'age', 'gender', 'bmi', 'systo', 'diasto', 'total_cognition', 
    'srh', 'rural', 'edu', 'hibpe', 'diabe', 'hearte', 
    'exercise', 'smokev', 'marry'
]

# --- 4. 页面主体 ---
st.title("👓 中老年人视力障碍风险预测系统")
st.info("本系统基于校准后的梯度提升模型，旨在为中老年人群提供视力障碍风险的早期预警。")

if model is None:
    st.error(f"❌ 资源加载失败。请检查路径。错误: {feature_list}")
    st.stop()

# --- 5. 模式选择 (第一步) ---
st.subheader("第一步：选择筛查模式")
mode = st.selectbox(
    "请选择适合您的筛查版本：",
    options=["请选择...", "精简版 (适合快速筛查 - 15个核心指标)", "完整版 (适合精准科研 - 全量指标)"],
    index=0
)

if mode == "请选择...":
    st.warning("👈 请在上方下拉框中选择一个版本以开始录入数据。")
    st.stop()

st.markdown("---")
st.subheader("第二步：录入受试者数据")

user_inputs = {}

# 模式判断逻辑
is_simplified = "精简版" in mode

# 选项卡布局
tab1, tab2, tab3 = st.tabs(["基本人口学", "临床生理指标", "既往病史与习惯"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        user_inputs['gender'] = st.selectbox("性别", options=["1", "2"], format_func=lambda x: GENDER_MAP[x])
        user_inputs['age'] = st.number_input("年龄", 45, 120, 65)
        user_inputs['rural'] = st.selectbox("居住地", options=["1", "2"], format_func=lambda x: RURAL_MAP[x])
    with col2:
        # 修正后的教育程度逻辑
        user_inputs['edu'] = st.selectbox("受教育程度", options=["1", "2", "3", "4"], format_func=lambda x: EDU_MAP[x])
        user_inputs['marry'] = st.selectbox("婚姻状况", options=["1", "2"], format_func=lambda x: MARRY_MAP[x])

with tab2:
    col3, col4 = st.columns(2)
    with col3:
        user_inputs['bmi'] = st.number_input("BMI (体重指数)", 10.0, 50.0, 23.5)
        user_inputs['systo'] = st.number_input("收缩压 (mmHg)", 50, 220, 130)
    with col4:
        user_inputs['diasto'] = st.number_input("舒张压 (mmHg)", 30, 150, 85)
        user_inputs['total_cognition'] = st.slider("认知功能总分", 0, 40, 25)

with tab3:
    col5, col6 = st.columns(2)
    with col5:
        user_inputs['srh'] = st.select_slider("自评健康状况", options=["1", "2", "3", "4", "5"], value="3", help="1为最差，5为最好")
        user_inputs['hibpe'] = st.selectbox("患有高血压", options=["0", "1"], format_func=lambda x: BINARY_MAP[x])
        user_inputs['diabe'] = st.selectbox("患有糖尿病", options=["0", "1"], format_func=lambda x: BINARY_MAP[x])
    with col6:
        user_inputs['hearte'] = st.selectbox("患有心脏病", options=["0", "1"], format_func=lambda x: BINARY_MAP[x])
        user_inputs['smokev'] = st.selectbox("曾有吸烟史", options=["0", "1"], format_func=lambda x: BINARY_MAP[x])
        user_inputs['exercise'] = st.selectbox("经常参加体育锻炼", options=["0", "1"], format_func=lambda x: BINARY_MAP[x])

# 如果是完整版，展示其余变量
if not is_simplified:
    with st.expander("更多详细指标 (完整版选填)"):
        st.caption("以下特征将使用默认值(0)填充，如有具体数据请修改：")
        remaining_features = [f for f in feature_list if f not in user_inputs]
        cols = st.columns(3)
        for idx, feat in enumerate(remaining_features):
            user_inputs[feat] = cols[idx % 3].number_input(f"{feat}", value=0.0)

# --- 6. 侧边栏配置 (阈值调整) ---
with st.sidebar:
    st.header("⚙️ 系统配置")
    st.info(f"当前模式: {mode.split('(')[0]}")
    st.divider()
    optimal_threshold = st.number_input("风险判断阈值", 0.1, 0.9, 0.45, 0.01, help="概率高于此值将被判定为高风险")
    st.divider()
    st.markdown("### 模型技术文档")
    st.caption("算法: Calibrated LGBM/GBoost")
    st.caption("训练数据: 中国中老年健康调查数据")

# --- 7. 预测执行 ---
st.markdown("---")
if st.button("🚀 开始风险评估"):
    with st.status("正在进行AI模型推理...", expanded=True) as status:
        st.write("数据预处理中...")
        # 特征对齐与补全
        final_data = {}
        for feat in feature_list:
            final_data[feat] = user_inputs.get(feat, 0) # 补全缺失项
            
        input_df = pd.DataFrame([final_data])[feature_list]
        time.sleep(0.4)
        
        st.write("分类特征转换中...")
        # 标签编码
        for col, le in encoders.items():
            if col in input_df.columns:
                val = str(input_df[col].values[0])
                if val in le.classes_:
                    input_df[col] = le.transform([val])[0]
                else:
                    input_df[col] = 0
        time.sleep(0.4)
        
        st.write("执行概率拟合与校准...")
        # 缩放
        input_scaled = scaler.transform(input_df)
        # 预测
        prob = model.predict_proba(input_scaled)[:, 1][0]
        is_high_risk = prob >= optimal_threshold
        time.sleep(0.4)
        
        status.update(label="评估完成！", state="complete", expanded=False)

    # --- 8. 结果展示 ---
    st.subheader("🔮 预测评估报告")
    c_res1, c_res2 = st.columns([1, 2])
    
    with c_res1:
        st.metric(label="视力障碍患病概率", value=f"{prob:.2%}")
        if is_high_risk:
            st.error("结论：高风险人群")
        else:
            st.success("结论：低风险人群")
            st.balloons()

    with c_res2:
        st.write("#### 风险可视化")
        st.progress(prob)
        st.caption(f"决策边界为 {optimal_threshold:.2f}。当前概率为 {prob:.2%}。")
        
        if is_high_risk:
            st.warning("⚠️ 建议：系统检测到较高的视力障碍风险，建议近期前往正规医疗机构进行专业验光与眼底检查。")
        else:
            st.info("💡 建议：目前风险较低，请继续保持良好的用眼习惯，并定期进行健康体检。")

# --- 9. 底部说明 ---
with st.expander("🔬 系统原理与指标说明"):
    st.markdown("""
    
    ### 系统逻辑
    1. **数据处理**: 针对不同筛查模式，系统会自动对齐 15 个核心变量或全量变量。
    2. **SHAP 原理**: 精简版选取的 15 个指标是根据 SHAP (SHapley Additive exPlanations) 贡献度选取的对视力健康影响最显著的因素。
    3. **概率校准**: 原始模型输出经过 Platt Scaling/Isotonic 校准，使概率值更具临床参考意义。
    
    ### 指标名词
    - **BMI**: 体重(kg) / 身高的平方($m^2$)。
    - **认知总分**: 反映中枢神经系统与视觉系统的协同健康状况。
    - **收缩压**: 俗称“高压”，长期高血压可能损害视网膜微血管。
    """)

st.markdown("---")
st.caption("© 2025 牡丹江医科大学护理学院 - 梅柏豪开发 邮箱：3011891593@qq.com | 仅供科研参考，不作为临床诊断依据")
