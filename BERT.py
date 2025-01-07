# 導入所需的模組
import pandas as pd  # 用於處理資料
import tensorflow as tf  # 用於深度學習模型
from transformers import BertTokenizer, TFBertForTokenClassification  # 用於BERT模型與標記器
from tensorflow.keras.losses import SparseCategoricalCrossentropy  # 損失函數
from tensorflow.keras.optimizers import Adam  # 優化器
from sklearn.model_selection import train_test_split  # 用於分割訓練和驗證資料
import numpy as np  # 用於數值處理
import random  # 用於隨機選取模板
import json  # 用於讀取JSON格式的模板
from tensorflow.keras import mixed_precision  # 用於啟用混合精度
import matplotlib.pyplot as plt  # 用於繪製圖形

# 啟用混合精度以加速訓練並減少顯存使用
def enable_mixed_precision():
    mixed_precision.set_global_policy('mixed_float16')
    print("混合精度已啟用。")

# 建立BERT模型進行命名實體識別（NER）
def create_ner_model(num_labels=5, max_length=128):
    model = TFBertForTokenClassification.from_pretrained('bert-base-chinese', num_labels=num_labels)  # 加載BERT模型
    optimizer = Adam(learning_rate=2e-5)  # 設定學習率
    loss = SparseCategoricalCrossentropy(from_logits=True)  # 使用損失函數
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])  # 編譯模型
    return model

# 從JSON文件中載入模板
def load_templates(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 讀取並解析JSON
    return data['templates']  # 返回模板列表

# 根據模板生成訓練數據
def generate_training_data(df, templates):
    training_data = []  # 用於儲存生成的訓練數據
    for _, row in df.iterrows():  # 遍歷資料集的每一行
        template = random.choice(templates)  # 隨機選擇一個模板
        # 填充模板
        query = template.format(
            keyword=row['search_keyword'],
            price=row['price_category'],
            review=row['review_category'],
            rating=row['star_rating_category']
        )
        # 新增到訓練數據
        training_data.append({
            'query': query,
            'search_keyword': row['search_keyword'],
            'price_category': row['price_category'],
            'review_category': row['review_category'],
            'star_rating_category': row['star_rating_category']
        })
    return pd.DataFrame(training_data)  # 返回生成的訓練數據作為DataFrame

# 準備資料（將句子轉換為BIO格式）
def prepare_data(data, tokenizer, max_length=128):
    input_ids, attention_masks, label_ids = [], [], []  # 初始化儲存數據的列表
    
    for _, row in data.iterrows():
        tokens = tokenizer(row['query'], padding='max_length', truncation=True, max_length=max_length, return_tensors='tf')  # 將查詢文字轉換為token
        labels = [0] * max_length  # 初始化標籤為 0 表示 O
        
        # 對每個實體類型進行編碼
        for entity, label_id in zip(
            ['search_keyword', 'price_category', 'review_category', 'star_rating_category'], 
            [1, 2, 3, 4]  # 對應的標籤ID
        ):
            entity_tokens = tokenizer.tokenize(row[entity])  # 將實體轉換為token
            entity_ids = tokenizer.convert_tokens_to_ids(entity_tokens)  # 將token轉換為ID
            # 在input_ids中尋找實體的匹配位置
            for i in range(len(tokens['input_ids'][0]) - len(entity_ids) + 1):
                if tokens['input_ids'][0][i:i+len(entity_ids)].numpy().tolist() == entity_ids:
                    labels[i] = label_id  # 設定 B-Label
                    for j in range(1, len(entity_ids)):
                        labels[i + j] = label_id  # 設定 I-Label
                    break
        
        input_ids.append(tokens['input_ids'][0].numpy())
        attention_masks.append(tokens['attention_mask'][0].numpy())
        label_ids.append(labels)
    
    return np.array(input_ids), np.array(attention_masks), np.array(label_ids)  # 返回處理好的數據

# 繪製訓練與驗證的損失與準確度曲線
def plot_training_history(history):
    # 繪製損失曲線
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='訓練損失')
    plt.plot(history.history['val_loss'], label='驗證損失')
    plt.title('損失曲線')
    plt.xlabel('Epoch')
    plt.ylabel('損失')
    plt.legend()
    
    # 繪製準確度曲線
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='訓練準確度')
    plt.plot(history.history['val_accuracy'], label='驗證準確度')
    plt.title('準確度曲線')
    plt.xlabel('Epoch')
    plt.ylabel('準確度')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 主訓練函數
def main():
    # 啟用混合精度
    enable_mixed_precision()
    
    # 讀取資料集
    df = pd.read_csv('products.csv')  # 請確保資料集有相應欄位
    
    # 讀取模板
    print("讀取模板...")
    templates = load_templates('templates.json')
    
    # 隨機生成訓練資料
    print("生成訓練資料...")
    training_data = generate_training_data(df, templates)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')  # 加載中文BERT標記器
    
    # 準備訓練資料
    print("準備資料...")
    input_ids, attention_masks, label_ids = prepare_data(training_data, tokenizer)
    
    # 將資料分成訓練集與驗證集
    X_train_ids, X_val_ids, X_train_masks, X_val_masks, y_train, y_val = train_test_split(
        input_ids,
        attention_masks,
        label_ids,
        test_size=0.2,
        random_state=42
    )

    train_data = {
        'input_ids': X_train_ids,
        'attention_mask': X_train_masks
    }
    val_data = {
        'input_ids': X_val_ids,
        'attention_mask': X_val_masks
    }

    # 確保標籤為 numpy 數組
    y_train = np.array(y_train)
    y_val = np.array(y_val)

    # 建立 NER 模型
    print("建立模型...")
    model = create_ner_model(num_labels=5)
    
    # 訓練模型並捕捉歷史
    print("開始訓練...")
    history = model.fit(
        train_data,
        y_train,
        validation_data=(val_data, y_val),
        epochs=5,
        batch_size=16
    )

    # 繪製訓練曲線
    print("繪製訓練曲線...")
    plot_training_history(history)
    
    # 保存模型
    print("保存模型...")
    model.save_pretrained('./ner_model')  # 保存整個模型（包括 config.json 和權重）
    tokenizer.save_pretrained('./bert_tokenizer')  # 保存 tokenizer
    print("模型訓練完成，模型和 tokenizer 已保存！")

if __name__ == "__main__":
    main()
