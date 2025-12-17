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

# 自定义外观
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
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

# --- 2. 资源加载 ---
@st.cache_resource
def load_assets():
    try:
        model_files = glob.glob('model_assets/best_model*.pkl')
        if not model_files:
            return None, None, None, "未找到模型文件"
        
        model = joblib.load(model_files[0])
        scaler = joblib.load('model_assets/scaler.pkl')
        encoders = joblib.load('model_assets/label_encoders.pkl')
        
        # 严格按照您提供的 70 个特征顺序进行加载
        with open('model_assets/feature_list.txt', 'r', encoding='utf-8') as f:
            features = [line.strip().split('. ')[1] for line in f.readlines() if '. ' in line]
        return model, scaler, encoders, features
    except Exception as e:
        return None, None, None, str(e)

model, scaler, encoders, feature_list = load_assets()

# --- 3. 核心变量定义 (基于 SHAP 图排序) ---
# 映射说明：
# 听力障碍 -> hear, 出生地区 -> rural, 年龄 -> age, 教育情况 -> edu, 认知能力 -> total_cognition
# 居住环境 -> water, 子女支持 -> hchild, 心智状况 -> psyche, 记忆能力 -> memrye, 退休状况 -> pension
# 体重 -> mweight, 社会评分 -> social_total, 疼痛评分 -> da042s_total, 收入 -> income_total, 身高 -> mheight
TOP_15_SHAP_FEATURES = [
    'hear', 'rural', 'age', 'edu', 'total_cognition', 
    'water', 'hchild', 'psyche', 'memrye', 'pension', 
    'mweight', 'social_total', 'da042s_total', 'income_total', 'mheight'
]

# 选项映射
MAPS = {
    'gender': {"1": "男", "2": "女"},
    'rural': {"1": "城镇", "2": "农村"},
    'edu': {"1": "高中及以上", "2": "中学", "3": "小学", "4": "不识字/半不识字"},
    'hear': {"0": "正常", "1": "有障碍"},
    'pension': {"0": "无", "1": "有"},
    'psyche': {"0": "良好", "1": "有心理/精神压力"},
    'memrye': {"1": "优", "2": "良", "3": "一般", "4": "差", "5": "极差"},
    'water': {"1": "自来水", "2": "井水/泉水", "3": "其他"},
    'binary': {"0": "否", "1": "是"}
}

# --- 4. 界面展示 ---
st.title("👓 中老年人视力障碍风险筛查系统")
st.markdown("---")

if model is None:
    st.error(f"❌ 资源加载失败: {feature_list}")
    st.stop()

# 模式选择
st.subheader("第一步：选择预测模式")
mode = st.selectbox(
    "根据 SHAP 重要性评估，建议使用精简版进行快速筛查：",
    options=["请选择模式...", "精简版 (基于核心 15 项指标)", "完整版 (全量 70 项指标)"]
)

if mode == "请选择模式...":
    st.stop()

st.markdown("---")
st.subheader("第二步：录入受试者信息")

user_inputs = {}
is_simplified = "精简版" in mode

# 布局设计
tab1, tab2, tab3 = st.tabs(["🧬 人口学与身体指标", "🧠 认知与心理", "🏡 生活环境与社会"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        user_inputs['age'] = st.number_input("年龄 (age)", 45, 120, 65)
        user_inputs['mheight'] = st.number_input("身高 cm (mheight)", 100, 220, 160)
        user_inputs['mweight'] = st.number_input("体重 kg (mweight)", 30, 150, 60)
    with c2:
        user_inputs['rural'] = st.selectbox("居住/出生地区 (rural)", ["1", "2"], format_func=lambda x: MAPS['rural'][x])
        user_inputs['edu'] = st.selectbox("受教育情况 (edu)", ["1", "2", "3", "4"], format_func=lambda x: MAPS['edu'][x])
        user_inputs['income_total'] = st.number_input("年总收入 (income_total)", 0, 1000000, 20000)

with tab2:
    c3, c4 = st.columns(2)
    with c3:
        user_inputs['hear'] = st.selectbox("听力障碍情况 (hear)", ["0", "1"], format_func=lambda x: MAPS['hear'][x])
        user_inputs['total_cognition'] = st.slider("认知能力评分 (total_cognition)", 0, 40, 20)
        user_inputs['memrye'] = st.selectbox("记忆能力评价 (memrye)", ["1", "2", "3", "4", "5"], format_func=lambda x: MAPS['memrye'][x])
    with c4:
        user_inputs['psyche'] = st.selectbox("心智/精神状况 (psyche)", ["0", "1"], format_func=lambda x: MAPS['psyche'][x])
        user_inputs['da042s_total'] = st.number_input("身体疼痛评分 (da042s_total)", 0, 50, 5)

with tab3:
    c5, c6 = st.columns(2)
    with c5:
        user_inputs['water'] = st.selectbox("居住饮水环境 (water)", ["1", "2", "3"], format_func=lambda x: MAPS['water'][x])
        user_inputs['hchild'] = st.number_input("子女支持/数量 (hchild)", 0, 15, 2)
    with c6:
        user_inputs['social_total'] = st.number_input("社会活动参与评分 (social_total)", 0, 100, 30)
        user_inputs['pension'] = st.selectbox("退休金状况 (pension)", ["0", "1"], format_func=lambda x: MAPS['pension'][x])

# 完整版补充输入
if not is_simplified:
    with st.expander("🔍 录入其余补充特征 (非核心变量)"):
        st.info("以下特征将使用默认值填充，如有数据请修改。")
        remaining_features = [f for f in feature_list if f not in user_inputs]
        cols = st.columns(3)
        for idx, feat in enumerate(remaining_features):
            user_inputs[feat] = cols[idx % 3].number_input(f"{feat}", value=0.0)

# --- 5. 预测执行 ---
st.markdown("---")
if st.button("🚀 开始 AI 风险评估"):
    with st.status("正在调取预测引擎...", expanded=True) as status:
        # 1. 特征全量对齐 (关键步：补齐 70 个特征)
        full_data = {}
        for feat in feature_list:
            # 如果是精简版中未录入的变量，填充 0
            full_data[feat] = user_inputs.get(feat, 0)
        
        # 转换为 DataFrame 并严格排序
        df = pd.DataFrame([full_data])[feature_list]
        
        # 2. 标签编码
        for col, le in encoders.items():
            if col in df.columns:
                val = str(df[col].values[0])
                df[col] = le.transform([val])[0] if val in le.classes_ else 0
        
        # 3. 预测
        df_scaled = scaler.transform(df)
        prob = model.predict_proba(df_scaled)[:, 1][0]
        is_high_risk = prob >= OPTIMAL_THRESHOLD
        
        status.update(label="计算完成！", state="complete", expanded=False)

    # --- 6. 结果展示 ---
    st.subheader("📊 评估结果报告")
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:
        st.metric(label="视力障碍患病概率", value=f"{prob:.2%}")
        if is_high_risk:
            st.error("结论：高风险人群")
        else:
            st.success("结论：低风险人群")
            st.balloons()

    with col_res2:
        st.write("#### 风险走势")
        st.progress(prob)
        st.caption(f"当前判断阈值为: {OPTIMAL_THRESHOLD}")
        if is_high_risk:
            st.warning("⚠️ 建议：检测到较高风险。建议受试者尽快前往医院眼科进行专业验光和眼底检查。")
        else:
            st.info("💡 建议：目前风险较低，建议保持良好的用眼习惯，并定期进行年度眼科检查。")

# --- 7. 系统原理 ---
with st.expander("🔬 预测原理说明"):
    st.write("本系统基于 **Gradient Boosting (梯度提升树)** 算法开发，并使用 **SHAP** 解释工具确定特征权重。")
    
    st.markdown("""
    **精简版指标选取逻辑：**
    根据 SHAP 贡献图，**听力障碍 (hear)** 和 **居住地 (rural)** 是对视力障碍预测贡献最大的因素。
    心智与认知状况（如认知得分、记忆评价）对中老年视力健康的预测也具有极高的敏感度。
    """)

st.markdown("---")
st.caption("© 2025 牡丹江医科大学护理学院 - 梅柏豪开发 | 仅供科研参考")
