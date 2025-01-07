# BERT_PCHomeProductKeywordExtraction
此項目使用BERT來做PCHome商品關鍵字萃取，包含爬蟲程式

## 專案簡介
待更新...

---

## 檔案架構
- 🔹 BERT.py => 模型訓練
- 🔹 BERTtest.py => 測試程式
- 🔹 webcrawler.py => PCHome 爬蟲程式
- 🔹 teplates.json => 文本樣板(可根據需要添加或修改)
- 🔹 tfTest.py => 測試當前 tensorflow 是否支援 GPU 加速

---

## 安裝方法
### 1. 克隆儲存庫
```bash
git clone https://github.com/TeemoCaption/BERT_PCHomeProductKeywordExtraction.git
```

### 2. 安裝Anaconda
下載網址： https://www.anaconda.com/download/success
安裝後檢查 Conda 是否已安裝： 
```bash
conda --version
```
如果顯示版本號，代表已安裝。

### 3. 建立 Conda 虛擬環境
執行以下指令
```bash
conda create --name tf_gpu python=3.9.21
```
tf_gpu 是虛擬環境名稱，可以替換成其他名稱（如專案名稱）。
python=3.9.21 是指定的 Python 版本，可根據需求更改。

### 4. 啟動虛擬環境
執行以下指令啟動剛建立的虛擬環境：
conda activate tf_gpu

### 5. 安裝套件
執行下列指令安裝所需套件
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn
pip install "tensorflow<2.11"
pip install pandas transformers scikit-learn numpy matplotlib
```

### 6. 環境測試
(需先 cd 進入工作目錄)
然後執行 tfTest.py 測試當前 tensorflow 是否支援 GPU 加速，如果支援 GPU 會印出「可用的 GPU： [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]」

```bash
python tfTest.py
```

### 7. 執行你想執行的程式
方法和第六步一樣。


