from transformers import pipeline
import gradio as gr


def load_sentiment_analysis_model():
    """
    加载 Hugging Face 的情感分析模型
    :return: 情感分析 pipeline
    """
    # 使用预训练的 DistilBERT 模型进行情感分析
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return sentiment_pipeline


def analyze_sentiment(text):
    """
    对输入的文本进行情感分析
    :param text: 用户输入的文本
    :return: 情感分析结果
    """
    try:
        # 加载模型
        sentiment_pipeline = load_sentiment_analysis_model()
        # 进行情感分析
        result = sentiment_pipeline(text)
        # 格式化输出
        output = []
        for res in result:
            output.append(f"文本: {text}")
            output.append(f"情感: {res['label']}")
            output.append(f"置信度: {res['score']:.4f}")
        return "\n".join(output)
    except Exception as e:
        return f"情感分析时出错: {e}"


def main():
    # 创建 Gradio 界面
    interface = gr.Interface(
        fn=analyze_sentiment,  # 处理函数
        inputs=gr.Textbox(lines=2, placeholder="请输入一段文本...", label="输入文本"),  # 输入组件
        outputs=gr.Textbox(label="情感分析结果"),  # 输出组件
        title="情感分析工具",  # 界面标题
        description="输入一段文本，工具将返回情感分析结果（正面/负面）及其置信度。",  # 界面描述
        examples=[["I love this product! It's amazing."], ["This is the worst experience I've ever had."]]  # 示例输入
    )
    # 启动界面
    interface.launch(share=True)


if __name__ == "__main__":
    main()
