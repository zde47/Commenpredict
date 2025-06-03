import streamlit as st
from transformers import pipeline

# 建立情緒分析 pipeline（我們預設使用 distilbert-base-uncased-finetuned-sst-2-english）
st.info("載入模型中，請稍候...")
classifier = pipeline("sentiment-analysis")
st.success("模型已載入！")

# 頁面標題與說明
st.title("評論留言情緒分析 Web")
st.write("請在下方輸入你的文字，系統將評估該文字的情緒（正向或負向）")

# 輸入區：使用者輸入文字
user_input = st.text_area("請輸入影評或文字：", height=150)

# 點擊按鈕後進行預測
if st.button("開始分析"):
    if user_input.strip() == "":
        st.error("請先輸入一些文字再進行分析！")
    else:
        st.info("正在分析中...")
        result = classifier(user_input)
        st.write("分析結果：")
        st.json(result)