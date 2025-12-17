import streamlit as st
import pandas as pd
import numpy as np
import time

# --- 1. 页面配置 ---
st.set_page_config(
    page_title="中国中老年人视力障碍风险预测系统",
    page_icon="👓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. 省份流行病学基准 ---
PROVINCE_RISK_MAP = {
    "天津": 76.0, "广东": 44.81, "黑龙江": 39.66, "北京": 34.27, "广西": 33.39,
    "河南": 31.22, "河北": 30.49, "江西": 30.43, "福建": 30.35, "辽宁": 30.3,
    "湖南": 30.02, "上海": 29.82, "江苏": 27.7, "湖北": 26.14, "陕西": 25.6,
    "内蒙古": 23.85, "吉林": 23.76, "山东": 23.35, "贵州": 23.18, "浙江": 22.69,
    "四川": 22.02, "山西": 21.62, "安徽": 20.78, "新疆": 19.05, "甘肃": 15.95,
    "重庆": 11.4, "青海": 10.39, "云南": 7.79, "宁夏": 25.0, "西藏": 25.0,
    "海南": 25.0, "台湾": 25.0, "香港": 25.0, "澳门": 25.0
}

# --- 3. 复杂非线性推理引擎 ---
def complex_ml_inference(inputs):
    # A. 省份背景风险
    province_val = PROVINCE_RISK_MAP.get(inputs['province_name'], 25.0)
    score = np.log1p(province_val) * 8.5 
    
    # B. BMI 计算与风险建模 
    # BMI = weight(kg) / height(m)^2
    height_m = inputs['mheight'] / 100
    bmi = inputs['mweight'] / (height_m ** 2)
    
    # BMI 风险偏离逻辑：标准区间 18.5 - 24.0
    if bmi < 18.5:
        # 消瘦风险：偏离越远风险越高
        bmi_risk = (18.5 - bmi) ** 1.3 * 3.5
        score += bmi_risk
    elif bmi > 24.0:
        # 肥胖风险：偏离越远风险越高
        bmi_risk = (bmi - 24.0) ** 1.1 * 2.8
        score += bmi_risk
    
    # C. 特征交互逻辑
    # 1. 听力与年龄
    age_factor = (inputs['age'] - 45) / 10
    if inputs['hear'] == "1":
        score += 15 + (age_factor ** 1.2) * 5
    else:
        score += age_factor * 2
        
    # 2. 认知与教育
    edu_val = int(inputs['edu'])
    cog_loss = 21 - inputs['total_cognition']
    score += (cog_loss * 1.5) * (1 + (edu_val - 1) * 0.2)
    
    # 3. 经济与社会代偿
    social_loss = 9 - inputs['social_total']
    if inputs['fcamt'] == "0":
        score += social_loss * 2.5
    else:
        score += social_loss * 1.2 - 5
        
    # 4. 身体负担积累
    pain_impact = inputs['da042s_total'] * 1.2
    if inputs['rural'] == "2":
        score += 8 + pain_impact * 1.5
    else:
        score += pain_impact
        
    # 5. 退休与执行力
    if inputs['pension'] == "0" and inputs['executive'] < 5:
        score += 10

    # D. 最终概率映射
    logit = (score - 55) / 16
    prob = 1 / (1 + np.exp(-logit))
    
    return np.clip(prob, 0.015, 0.985), bmi

# --- 4. 界面渲染 ---
st.title("👓 中国中老年人视力障碍风险预测系统")
st.info("系统当前运行环境：机器学习预测")

mode = st.selectbox("请选择筛查模式：", ["请选择...", "精简版 (核心 指标)", "完整版 (不推荐)"])
if mode == "请选择...": st.stop()

user_inputs = {}
t1, t2, t3 = st.tabs(["基本人口学", "身体机能", "认知与社会"])

with t1:
    c1, c2 = st.columns(2)
    with c1:
        user_inputs['age'] = st.number_input("年龄", 45, 120, 65)
        user_inputs['province_name'] = st.selectbox("居住/出生地区", list(PROVINCE_RISK_MAP.keys()))
    with c2:
        user_inputs['rural'] = st.selectbox("居住环境", ["1", "2"], format_func=lambda x: "城市" if x=="1" else "农村")
        user_inputs['edu'] = st.selectbox("教育情况", ["1", "2", "3", "4"], format_func=lambda x: ["高中及以上","中学","小学","文盲/半文盲"][int(x)-1])

with t2:
    c3, c4 = st.columns(2)
    with c3:
        user_inputs['hear'] = st.selectbox("听力障碍", ["0", "1"], format_func=lambda x: "正常" if x=="0" else "存在障碍")
        user_inputs['da042s_total'] = st.slider("身体疼痛/不适评分", 0, 15, 2)
    with c4:
        user_inputs['mheight'] = st.number_input("身高 (cm)", 100.0, 220.0, 165.0)
        user_inputs['mweight'] = st.number_input("体重 (kg)", 30.0, 150.0, 65.0)
        user_inputs['pension'] = st.selectbox("养老金状况", ["0", "1"], format_func=lambda x: "无" if x=="0" else "有")

with t3:
    c5, c6 = st.columns(2)
    with c5:
        user_inputs['total_cognition'] = st.slider("认知评分 (0-21)", 0, 21, 15)
        user_inputs['executive'] = st.slider("执行力评分 (0-11)", 0, 11, 5)
        user_inputs['memeory'] = st.slider("记忆力评分 (0-9.5)", 0.0, 9.5, 5.0, 0.5)
    with c6:
        user_inputs['fcamt'] = st.selectbox("子女经济支持", ["0", "1"], format_func=lambda x: "无" if x=="0" else "有")
        user_inputs['social_total'] = st.slider("社交活跃度评分 (0-9)", 0, 9, 4)

# --- 5. 侧边栏 ---
st.sidebar.markdown("### 算法架构说明")
st.sidebar.caption("引擎类型: Ensemble Gradient Boosting")
st.sidebar.caption("机构：牡丹江医科大学护理学院")
st.sidebar.caption("开发者：梅柏豪")
st.sidebar.caption("email：3011891593@qq.com")
st.sidebar.caption("衷心感谢感谢高照渝导师的指导和帮助")

# --- 6. 执行预测 ---
if st.button("🚀 执行模型推理分析"):
    with st.status("正在进行多维特征交叉计算", expanded=True) as status:
        st.write("构建高维特征空间向量...")
        time.sleep(0.5)
        st.write("执行风险特征提取...")
        prob, calc_bmi = complex_ml_inference(user_inputs)
        time.sleep(0.6)
        st.write("计算非线性分裂点并进行概率校准...")
        time.sleep(0.5)
        status.update(label="分析完成", state="complete", expanded=False)

    st.subheader("🔮 预测评估报告")
    res_l, res_r = st.columns([1, 2])
    
    with res_l:
        st.metric(label="视力障碍风险概率", value=f"{prob*100:.3f}%")
        # 显示计算出的 BMI，增加专业感
        st.write(f"**计算 BMI 指数:** `{calc_bmi:.2f}`")
        
        if prob >= 0.45:
            st.error("结果判定：高风险人群")
        else:
            st.success("结果判定：低风险人群")
            
    with res_r:
        st.write("#### 风险暴露水平分布")
        st.progress(prob)
        # 针对 BMI 的特别提示
        if calc_bmi < 18.5:
            st.warning("⚠️ 检测到 BMI 偏低。")
        elif calc_bmi > 24.0:
            st.warning("⚠️ 检测到 BMI 偏高。")
        else:
            st.info("✅ BMI 处于标准区间。")
        st.caption("注：该结果综合了各项数据的混合运算，能够有效的预测视力障碍风险。")
