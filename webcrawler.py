import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import random

# 定義關鍵字列表
keywords = ["桌上型電腦", "筆記型電腦","平板電腦","手機","滑鼠","耳機","家電","衛生紙","奶粉","尿布","零食","泡麵","鍋具","文具","營養品","香水","保養品","洗髮精","顯示卡","小說"]

# PChome 搜尋基礎網址
base_url = "https://24h.pchome.com.tw/search/?q="

# 設定 headers 模擬瀏覽器
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# 定義 CSV 檔案路徑
csv_file = 'products.csv'

# 初始化存放商品資訊的清單
all_products = []

for keyword in keywords:
    print(f"開始爬取關鍵字: {keyword}")
    page = 1  # 初始化分頁
    max_pages = 10  # 每個關鍵字最多爬取 10 頁

    while page <= max_pages:
        try:
            # 構造 URL（包含關鍵字與分頁）
            url = f"{base_url}{keyword}&p={page}"
            print(f"正在爬取: {url}")
            
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"無法連接至目標網站，狀態碼: {response.status_code}")
                break
            
            # 解析 HTML 內容
            soup = BeautifulSoup(response.text, 'html.parser')

            # 找到商品清單的區域
            product_list = soup.find_all("div", class_="c-prodInfoV2 c-prodInfoV2--gridCard")

            # 找到分類清單的區域
            category_list = soup.find_all("div", class_="c-inputLabel__text c-inputLabel__text--s400grayDarkest")

            if not product_list:  # 如果沒有商品，跳出分頁循環
                print("沒有更多商品，跳出分頁爬取。")
                break

            # 提取分類資訊
            category_texts = [cat.text.strip() for cat in category_list]

            for product in product_list:
                # 商品名稱
                title = product.find("div", class_="c-prodInfoV2__title")
                title_text = title.text.strip() if title else "無名稱"

                # 商品價格
                price = product.find("div", class_="c-prodInfoV2__priceValue")
                price_text = price.text.strip() if price else "0"
                try:
                    price_value = int(price_text.replace("$", "").replace(",", ""))
                except ValueError:
                    price_value = 0

                # 商品星星數
                rating = product.find("div", class_="c-ratingIcon__list")
                stars = len(rating.find_all("div", class_="c-ratingIcon__item")) if rating else 0

                # 商品評論數
                reviews = product.find("div", class_="c-prodInfoV2__text c-prodInfoV2__text--xs500GrayDark")
                reviews_text = reviews.text.strip() if reviews else "0"
                try:
                    reviews_value = int(reviews_text.replace("(", "").replace(")", ""))
                except ValueError:
                    reviews_value = 0

                # 分類資訊
                category = category_texts[0] if category_texts else "無分類"

                # 商品超連結
                link_tag = product.find("a", class_="c-prodInfoV2__link gtmClickV2")
                product_link = f"https://24h.pchome.com.tw{link_tag['href']}" if link_tag else "無連結"

                # 將商品資訊加入清單
                all_products.append({
                    "product_name": title_text,
                    "price": price_value,
                    "review_count": reviews_value,
                    "star_rating": stars,
                    "category": category,
                    "product_link": product_link,
                    "search_keyword": keyword,
                })

            # 下一頁
            page += 1

            # 隨機延遲，模擬人工操作
            time.sleep(random.uniform(1, 3))

        except Exception as e:
            print(f"在爬取關鍵字 '{keyword}' 的第 {page} 頁時發生錯誤: {e}")
            break

# 將爬取的商品資訊轉換為 DataFrame
new_data = pd.DataFrame(all_products)

# 檢查 CSV 檔案是否存在
if os.path.exists(csv_file):
    # 讀取現有的 CSV 檔案
    existing_data = pd.read_csv(csv_file)
    # 將新資料追加到現有資料後面
    combined_data = pd.concat([existing_data, new_data], ignore_index=True)
else:
    # 如果 CSV 檔案不存在，使用新資料作為初始資料
    combined_data = new_data

# 移除重複的商品（根據商品名稱）
combined_data.drop_duplicates(subset=['product_name'], inplace=True)

# 計算非 0 值的四分位數
price_quartiles = combined_data[combined_data['price'] > 0]['price'].quantile([0.25, 0.5, 0.75]).values
review_quartiles = combined_data[combined_data['review_count'] > 0]['review_count'].quantile([0.25, 0.5, 0.75]).values
star_quartiles = combined_data[combined_data['star_rating'] > 0]['star_rating'].quantile([0.25, 0.5, 0.75]).values

# 定義分類函數
def categorize(value, quartiles, categories):
    if value == 0:  # 如果值為 0，直接歸類為最低區間
        return categories[0]
    elif value < quartiles[0]:
        return categories[0]
    elif quartiles[0] <= value < quartiles[1]:
        return categories[1]
    elif quartiles[1] <= value < quartiles[2]:
        return categories[2]
    else:
        return categories[3]

# 定義類別標籤
price_categories = ['低', '中低', '中高', '高']
review_categories = ['少', '中', '多', '非常多']
star_categories = ['低', '中', '高', '非常高']

# 分類價格
combined_data['price_category'] = combined_data['price'].apply(
    lambda x: categorize(x, price_quartiles, price_categories)
)

# 分類評論數
combined_data['review_category'] = combined_data['review_count'].apply(
    lambda x: categorize(x, review_quartiles, review_categories)
)

# 分類星級評分
combined_data['star_rating_category'] = combined_data['star_rating'].apply(
    lambda x: categorize(x, star_quartiles, star_categories)
)

# 儲存資料到 CSV 檔案（不包含索引）
combined_data.to_csv(csv_file, index=False, encoding='utf-8-sig')

print("所有資料已成功儲存到 products.csv")
