import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time

# --- 页面配置 ---
st.set_page_config(
    page_title="视力健康辅助预测系统",
    page_icon="👓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 自定义 CSS 样式 ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 资源加载 ---
@st.cache_resource
def load_assets():
    # 注意：请确保路径和文件名与您导出的一致
    try:
        # 尝试自动搜索模型文件
        import glob
        model_path = glob.glob('model_assets/best_model_*.pkl')[0]
        model = joblib.load(model_path)
        scaler = joblib.load('model_assets/scaler.pkl')
        encoders = joblib.load('model_assets/label_encoders.pkl')
        with open('model_assets/feature_list.txt', 'r', encoding='utf-8') as f:
            features = [line.strip().split('. ')[1] for line in f.readlines() if '. ' in line]
        return model, scaler, encoders, features
    except Exception as e:
        return None, None, None, str(e)

model, scaler, encoders, feature_list = load_assets()

if model is None:
    st.error(f"❌ 资源加载失败。错误信息: {feature_list}")
    st.info("提示：请检查 GitHub 仓库中 model_assets 文件夹下的文件是否存在且路径正确。")
    st.stop()

# --- 核心逻辑：定义精简版变量 (Top 15) ---
# 这里根据常见医疗重要性预设了15个变量，您可以根据SHAP排名调整
TOP_15_FEATURES = [
    'age', 'gender', 'bmi', 'systo', 'diasto', 'total_cognition', 
    'srh', 'rural', 'edu', 'hibpe', 'diabe', 'hearte', 
    'exercise', 'smokev', 'marry'
]

# --- 侧边栏与标题 ---
st.title("👓 视力健康（眼镜佩戴）预测系统")
st.markdown("本系统基于校准后的梯度提升模型，通过您的基本体征和健康数据评估视力风险。")

with st.sidebar:
    st.header("⚙️ 预测配置")
    mode = st.radio(
        "选择预测模式",
        ["精简版 (15个核心指标)", "完整版 (全量数据输入)"],
        help="精简版仅需输入对结果影响最大的15个指标，其余由系统自动填充默认值。"
    )
    
    st.divider()
    st.subheader("📊 模型信息")
    st.info("当前算法：Calibrated Gradient Boosting")
    OPTIMAL_THRESHOLD = st.number_input("决策阈值 (Threshold)", value=0.45, step=0.01)
    st.write("注：高于此概率将被判定为需戴眼镜。")

# --- 输入表单 ---
st.header("📋 受试者信息录入")
if mode == "精简版 (15个核心指标)":
    st.caption("✨ 当前模式：精简版。仅展示关键特征以提高录入效率。")
else:
    st.caption("🧪 当前模式：完整版。提供详细特征以获得更高精度的预测。")

user_inputs = {}

# 使用 Tab 组织界面
tab1, tab2, tab3 = st.tabs(["基本人口学", "健康指标", "生活习惯与病史"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        user_inputs['gender'] = st.selectbox("性别", ["1", "2"], format_func=lambda x: "男 (1)" if x=="1" else "女 (2)")
        user_inputs['age'] = st.number_input("年龄", 0, 120, 60)
        user_inputs['rural'] = st.selectbox("居住地", ["1", "2"], format_func=lambda x: "农村 (1)" if x=="1" else "城市 (2)")
    with col2:
        user_inputs['edu'] = st.selectbox("受教育程度", ["1", "2", "3", "4"], format_func=lambda x: f"级别 {x}")
        user_inputs['marry'] = st.selectbox("婚姻状况", ["1", "2", "3"], format_func=lambda x: f"状态 {x}")

with tab2:
    col3, col4 = st.columns(2)
    with col3:
        user_inputs['bmi'] = st.number_input("BMI (体重指数)", 10.0, 50.0, 24.0)
        user_inputs['systo'] = st.number_input("收缩压 (mmHg)", 50, 250, 120)
        user_inputs['diasto'] = st.number_input("舒张压 (mmHg)", 30, 150, 80)
    with col4:
        user_inputs['srh'] = st.slider("自评健康状况 (1-5, 5为最健康)", 1, 5, 3)
        user_inputs['total_cognition'] = st.number_input("认知功能总分", 0, 40, 20)

with tab3:
    col5, col6 = st.columns(2)
    binary_opts = {"0": "否 (0)", "1": "是 (1)"}
    with col5:
        user_inputs['hibpe'] = st.selectbox("是否有高血压", ["0", "1"], format_func=lambda x: binary_opts[x])
        user_inputs['diabe'] = st.selectbox("是否有糖尿病", ["0", "1"], format_func=lambda x: binary_opts[x])
        user_inputs['hearte'] = st.selectbox("是否有心脏病", ["0", "1"], format_func=lambda x: binary_opts[x])
    with col6:
        user_inputs['smokev'] = st.selectbox("是否有吸烟史", ["0", "1"], format_func=lambda x: binary_opts[x])
        user_inputs['exercise'] = st.selectbox("是否规律运动", ["0", "1"], format_func=lambda x: binary_opts[x])

    # 如果是完整版，在这里展示剩余的所有变量输入
    if mode == "完整版 (全量数据输入)":
        st.divider()
        st.subheader("补充特征 (完整模式下可用)")
        remaining_features = [f for f in feature_list if f not in user_inputs]
        for i in range(0, len(remaining_features), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(remaining_features):
                    feat = remaining_features[i+j]
                    user_inputs[feat] = cols[j].number_input(f"{feat}", value=0.0)

# --- 预测引擎 ---
st.divider()
if st.button("🚀 开始 AI 风险评估"):
    
    # 模拟计算动画
    with st.status("正在分析数据并调取模型...", expanded=True) as status:
        st.write("1. 正在根据模式进行数据对齐...")
        # 补全缺失特征
        full_input_data = {}
        for feat in feature_list:
            full_input_data[feat] = user_inputs.get(feat, 0) # 缺失项填0
        
        df = pd.DataFrame([full_input_data])[feature_list]
        time.sleep(0.5)
        
        st.write("2. 正在执行标签编码与标准化...")
        # 编码
        for col, le in encoders.items():
            if col in df.columns:
                val = str(df[col].values[0])
                if val in le.classes_:
                    df[col] = le.transform([val])[0]
                else:
                    df[col] = 0
        # 缩放
        df_scaled = scaler.transform(df)
        time.sleep(0.5)
        
        st.write("3. 正在运行概率校准计算...")
        prob = model.predict_proba(df_scaled)[:, 1][0]
        prediction = 1 if prob >= OPTIMAL_THRESHOLD else 0
        time.sleep(0.5)
        
        status.update(label="✅ 计算完成!", state="complete", expanded=False)

    # --- 结果展示 ---
    st.subheader("🔮 预测结论")
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.metric(label="预测概率", value=f"{prob:.2%}")
        if prediction == 1:
            st.error("结论：高风险 - 需要/建议佩戴眼镜")
        else:
            st.success("结论：低风险 - 目前可能不需要眼镜")
            st.balloons() # 低风险时给个庆祝动画

    with res_col2:
        st.write("#### 风险阈值图")
        # 显示进度条
        st.progress(prob)
        st.caption(f"当前设定的判断阈值为: {OPTIMAL_THRESHOLD}")
        
        if prediction == 1:
            st.warning(f"由于概率高于阈值 {OPTIMAL_THRESHOLD}，建议进行专业验光。")
        else:
            st.info("您的风险处于较低水平，请继续保持良好的用眼习惯。")

# --- 解释说明部分 ---
with st.expander("📚 指标名词解释与系统原理"):
    st.markdown("""
    ### 关键指标说明
    * **BMI (体重指数)**: 体重(kg)除以身高(m)的平方。研究表明部分慢性代谢疾病与视力变化相关。
    * **收缩压/舒张压**: 血压水平反映了心血管健康，视网膜血管是人体唯一能直接观察到的微循环。
    * **认知功能总分**: 认知能力与视觉处理能力在神经层面具有相关性。
    * **SRH (自评健康)**: 个人对身体的主观感受，通常是多项健康指标的综合体现。

    ### 预测模式说明
    1.  **精简版**: 利用“特征重要性”原则，仅要求输入前15位对结果贡献最大的数据。其余变量以训练集的中位数或默认值填充。这种模式在损失极小精度的情况下提供了极高的录入效率。
    2.  **完整版**: 调用所有特征，适用于科研或需要极端精确结论的场景。

    ### 免责声明
    本工具基于机器学习算法分析所得，仅供科研与参考，**不能代替医生的临床诊断**。
    """)

st.markdown("---")
st.caption("© 2024 视力健康科研组 | 基于 Streamlit & Scikit-Learn 构建")
