import tensorflow as tf
from transformers import BertTokenizer, TFBertForTokenClassification
import numpy as np

# 載入模型與 tokenizer
def load_model():
    """
    載入已微調的 BERT 模型和對應的標記器（tokenizer）。
    模型有 5 個標籤，分別對應實體類型（包括非實體）。
    """
    model = TFBertForTokenClassification.from_pretrained('bert-base-chinese', num_labels=5)  # 加載 BERT 模型
    model.load_weights('ner_model_weights.h5')  # 載入已訓練的模型權重
    tokenizer = BertTokenizer.from_pretrained('./bert_tokenizer')  # 載入 BERT 標記器
    return model, tokenizer

# 萃取關鍵字
def extract_keywords(text, model, tokenizer):
    """
    根據輸入文字，使用模型預測實體並萃取關鍵字。
    
    Args:
        text (str): 要進行實體識別的文字。
        model: 預訓練的 BERT 模型。
        tokenizer: 對應的 BERT 標記器。
    
    Returns:
        dict: 各類實體的關鍵字字典。
    """
    # 將輸入文字轉換為 BERT 可處理的格式（包含 token 的 ID 和 attention mask）
    encoded = tokenizer(text, return_tensors='tf', padding=True, truncation=True, max_length=128)
    
    # 使用模型進行預測，取得 logits（每個 token 的標籤機率分數）
    predictions = model(encoded).logits
    
    # 取出每個 token 預測的標籤（取機率最大的位置）
    pred_labels = tf.argmax(predictions, axis=-1).numpy()[0]
    
    # 將 token ID 轉回對應的文字（token）
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
    
    # 初始化關鍵字字典，存放實體類型對應的關鍵字
    keywords = {
        "search_keyword": "",
        "price_category": "",
        "review_category": "",
        "star_rating_category": ""
    }
    
    # 初始化變數，用於暫存當前實體
    current_entity = []  # 暫存當前實體的 token
    current_label = 0    # 暫存當前實體的標籤類型

    # 遍歷 tokens 與對應的標籤
    for token, label in zip(tokens, pred_labels):
        if label > 0:  # 如果標籤大於 0，表示該 token 是某個實體的一部分
            if label != current_label:  # 如果標籤改變，代表新實體的開始
                if current_entity:  # 如果有已累積的實體，將其加入結果字典
                    entity = ''.join(current_entity).replace('##', '')  # 合併 token 並去掉子詞標記 ##
                    keywords[list(keywords.keys())[current_label - 1]] = entity  # 將實體放入對應的關鍵字類型
                current_entity = [token]  # 開始新實體
                current_label = label  # 更新當前實體標籤
            else:
                current_entity.append(token)  # 實體未結束，繼續累積 token
        else:  # 如果標籤為 0，表示當前 token 不屬於任何實體
            if current_entity:  # 如果有已累積的實體，將其加入結果字典
                entity = ''.join(current_entity).replace('##', '')  # 合併 token 並去掉子詞標記 ##
                keywords[list(keywords.keys())[current_label - 1]] = entity  # 將實體放入對應的關鍵字類型
                current_entity = []  # 清空暫存的實體
                current_label = 0  # 重置當前實體標籤
    
    return keywords  # 返回關鍵字字典

# 測試程式
def main():
    """
    測試 BERT 模型的 NER 功能，輸入句子並萃取出關鍵字。
    """
    # 載入模型和標記器
    model, tokenizer = load_model()
    
    # 測試輸入的句子
    test_query = "我家的貓最近吃太多貓飼料了，貓飼料都吃完了，你能推薦價格較低、評論多的貓飼料嗎?"
    
    # 執行關鍵字萃取
    keywords = extract_keywords(test_query, model, tokenizer)
    
    # 輸出結果
    print("萃取的關鍵字與分類:")
    print(keywords)

# 執行主函數
if __name__ == "__main__":
    main()
