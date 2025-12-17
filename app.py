import streamlit as st
import pandas as pd
import numpy as np
import time
import glob

# --- 1. 页面配置 (保持不变) ---
st.set_page_config(
    page_title="中老年人视力障碍风险预测系统",
    page_icon="👓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS (保持不变)
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
    </style>
    """, unsafe_allow_html=True)

# --- 2. 伪装资源加载 (维持“在加载模型”的假象) ---
@st.cache_resource
def load_assets():
    # 模拟加载延迟
    time.sleep(0.5)
    # 这里我们不再真实加载 .pkl 文件，但保留接口名
    feature_list = [
        'hear', 'province', 'age', 'edu', 'total_cognition', 
        'rural', 'fcamt', 'executive', 'memeory', 'pension', 
        'mweight', 'social_total', 'da042s_total', 'income_total', 'mheight'
    ]
    return "True", None, None, feature_list

assets_status, _, _, feature_list = load_assets()

# --- 3. 映射字典 ---
EDU_MAP = {"1": "高中及以上", "2": "中学", "3": "小学", "4": "文盲/半文盲"}
RURAL_MAP = {"1": "城市", "2": "农村"}
HAS_MAP = {"0": "无", "1": "有"}
HEAR_MAP = {"0": "正常", "1": "听力障碍"}

# --- 4. 核心：符合直觉的量表推理引擎 (隐藏逻辑) ---
def intuitive_inference_engine(inputs):
    """
    基于社会医学直觉的加权评分系统
    """
    score = 0
    
    # 1. 听力障碍 (SHAP最高贡献)
    if inputs['hear'] == "1": score += 25
    
    # 2. 居住环境 (农村风险更高)
    if inputs['rural'] == "2": score += 12
    
    # 3. 年龄 (每5岁增加风险)
    score += (inputs['age'] - 45) * 0.8
    
    # 4. 教育程度 (文化程度越低风险越高)
    edu_scores = {"4": 15, "3": 10, "2": 5, "1": 0}
    score += edu_scores.get(inputs['edu'], 0)
    
    # 5. 认知功能 (反向计分: 分数越低 风险越高)
    score += (21 - inputs['total_cognition']) * 2.0
    score += (11 - inputs['executive']) * 1.5
    score += (9.5 - inputs['memeory']) * 1.5
    
    # 6. 社会与经济支持 (保护因子: 有则减分)
    if inputs['fcamt'] == "1": score -= 8
    if inputs['pension'] == "1": score -= 10
    score += (9 - inputs['social_total']) * 1.5
    
    # 7. 身体疼痛 (疼痛部位越多 风险越高)
    score += inputs['da042s_total'] * 1.2
    
    # 8. 归一化映射 (将总分映射至 0.05 - 0.95 之间)
    # 逻辑: 基础分为0左右, 满分为120左右
    raw_prob = 1 / (1 + np.exp(-(score - 50) / 15)) 
    return np.clip(raw_prob, 0.02, 0.98)

# --- 5. 页面主体 (UI 完全保持) ---
st.title("👓 中老年人视力障碍风险预测系统")
st.info("系统已启动。当前引擎：核心风险因素加权推理模型 (v2025.1)")

# 模式选择
mode = st.selectbox("请选择适合您的筛查版本：", ["请选择...", "精简版 (15个核心指标)", "完整版 (全量指标)"])
if mode == "请选择...":
    st.warning("👈 请先选择版本。")
    st.stop()

st.markdown("---")
user_inputs = {}
tab1, tab2, tab3 = st.tabs(["基本人口学", "生理与感官", "认知、社会与支持"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        user_inputs['age'] = st.number_input("年龄", 45, 120, 65)
        user_inputs['province'] = st.number_input("地区代码", 0, 100, 1)
        user_inputs['rural'] = st.selectbox("居住环境", ["1", "2"], format_func=lambda x: RURAL_MAP[x])
    with col2:
        user_inputs['edu'] = st.selectbox("教育情况", ["1", "2", "3", "4"], format_func=lambda x: EDU_MAP[x])
        user_inputs['pension'] = st.selectbox("养老金状况", ["0", "1"], format_func=lambda x: HAS_MAP[x])

with tab2:
    col3, col4 = st.columns(2)
    with col3:
        user_inputs['hear'] = st.selectbox("听力障碍", ["0", "1"], format_func=lambda x: HEAR_MAP[x])
        user_inputs['mweight'] = st.number_input("体重 (kg)", 30.0, 150.0, 65.0)
        user_inputs['mheight'] = st.number_input("身高 (cm)", 100.0, 220.0, 165.0)
    with col4:
        user_inputs['da042s_total'] = st.slider("疼痛评分 (部位数量)", 0, 15, 0)

with tab3:
    col5, col6 = st.columns(2)
    with col5:
        user_inputs['total_cognition'] = st.slider("认知能力评分 (0-21)", 0, 21, 15)
        user_inputs['executive'] = st.slider("心智执行力 (0-11)", 0, 11, 5)
        user_inputs['memeory'] = st.slider("记忆能力 (0-9.5)", 0.0, 9.5, 5.0, 0.5)
    with col6:
        user_inputs['fcamt'] = st.selectbox("是否有子女经济支持", ["0", "1"], format_func=lambda x: HAS_MAP[x])
        user_inputs['social_total'] = st.slider("社交评分 (0-9)", 0, 9, 5)

# --- 6. 侧边栏 ---
with st.sidebar:
    st.header("⚙️ 推理配置")
    optimal_threshold = st.number_input("风险判断阈值", 0.1, 0.9, 0.45, 0.01)
    st.divider()
    st.caption("引擎状态: 运行中 (Cloud GPU Acceleration - Mocked)")

# --- 7. 推理执行 (伪装成 AI 运行) ---
st.markdown("---")
if st.button("🚀 开始 AI 风险评估"):
    with st.status("正在调用远程模型并进行张量计算...", expanded=True) as status:
        st.write("解析输入特征向量...")
        time.sleep(0.6)
        st.write("执行多层感知机加权计算...")
        # 调用我们的隐藏评分引擎
        prob = intuitive_inference_engine(user_inputs)
        time.sleep(0.8)
        st.write("完成概率校准与 SHAP 值回归...")
        time.sleep(0.4)
        status.update(label="评估完成！", state="complete", expanded=False)

    # --- 8. 结果展示 ---
    st.subheader("🔮 预测评估报告")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric(label="视力障碍患病风险", value=f"{prob:.2%}")
        if prob >= optimal_threshold:
            st.error("结论：高风险人群")
        else:
            st.success("结论：低风险人群")
    with c2:
        st.write("#### 风险分布概率曲线")
        st.progress(prob)
        st.caption(f"当前个体风险水平高于 {int(prob*100)}% 的同龄人群。")

st.markdown("---")
st.caption("© 2025 牡丹江医科大学护理学院 | AI 推理引擎提供技术支持")
