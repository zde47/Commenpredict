import streamlit as st
from transformers import pipeline
from transformers.utils import logging

# 關閉 transformers 訊息提示
logging.set_verbosity_error()

# 頁面設定
st.set_page_config(
    page_title="留言情感分析器",
    page_icon="🧠",
    layout="centered"
)

# 載入模型
st.info("載入模型中，請稍候...")
try:
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1,  # 使用 CPU 避免 meta tensor 問題
        framework="pt"
    )
    st.success("✅ 模型載入完成！")
    st.caption("模型來源：Hugging Face（distilbert-base-uncased-finetuned-sst-2-english）")
except Exception as e:
    st.error(f"❌ 模型載入失敗：{e}")
    st.stop()

# 標題與說明
st.title("📝 留言情感分析 Web App")
st.write("請在下方輸入留言文字，系統將判斷其情緒為 **正面** 或 **負面**，並顯示信心分數。")

# 使用者輸入
user_input = st.text_area("✏️ 請輸入一段留言內容：", height=150)

# 分析按鈕
if st.button("開始分析"):
    if user_input.strip() == "":
        st.warning("⚠️ 請先輸入一些文字再進行分析！")
    else:
        with st.spinner("模型正在分析中，請稍候..."):
            try:
                result = classifier(user_input)
                if isinstance(result, list) and len(result) > 0:
                    label = result[0].get("label", "未知")
                    score = float(result[0].get("score", 0.0))
                    emoji = "👍" if label == "POSITIVE" else "👎"
                    st.subheader("📊 分析結果")
                    st.success(f"{emoji} 預測情緒：**{label}**（信心值：{score:.2%}）")
                else:
                    st.warning("⚠️ 模型沒有回傳有效結果，請重新嘗試。")
            except Exception as e:
                st.error(f"❌ 發生錯誤：{e}")
