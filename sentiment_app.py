import streamlit as st
from transformers import pipeline


st.set_page_config(
    page_title="留言情感分析器",
    page_icon="🧠",
    layout="centered"
)


st.info("載入模型中，請稍候...")
try:
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1  # 使用 CPU
    )

    st.success("✅ 模型載入完成！")
    st.caption("模型來源：Hugging Face（distilbert-base-uncased-finetuned-sst-2-english）")
except Exception as e:
    st.error(f"❌ 模型載入失敗：{e}")
    st.stop()

st.title("📝 留言情感分析 Web App")
st.write("請在下方輸入留言文字，系統將判斷其情緒為 **正面** 或 **負面**，並顯示信心分數。")

user_input = st.text_area("✏️ 請輸入一段留言內容：", height=150)


if st.button("開始分析"):
    if user_input.strip() == "":
        st.warning("⚠️ 請先輸入一些文字再進行分析！")
    else:
        with st.spinner("模型正在分析中，請稍候..."):
            try:
                result = classifier(user_input)
                label = result[0]["label"]
                score = result[0]["score"]
                emoji = "👍" if label == "POSITIVE" else "👎"
                st.success(f"{emoji} 預測結果：**{label}**（信心值：{score:.2%}）")
            except Exception as e:
                st.error(f"❌ 發生錯誤：{e}")
