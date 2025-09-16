# -*- coding: utf-8 -*-
"""
CÔNG CỤ CỐ VẤN CỦA LÝ VĂN HIỆP(KIOSOO)
Xây dựng bằng Streamlit.
"""
import streamlit as st
from pytrends.request import TrendReq
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import time
from requests.exceptions import ReadTimeout

# --- Dữ liệu tĩnh ---
COUNTRIES = {
    'Vietnam': 'VN', 'United States': 'US', 'Japan': 'JP', 'South Korea': 'KR',
    'United Kingdom': 'GB', 'Germany': 'DE', 'France': 'FR', 'Canada': 'CA',
    'Australia': 'AU', 'India': 'IN', 'Brazil': 'BR', 'Russia': 'RU',
    'Thailand': 'TH', 'Singapore': 'SG', 'Malaysia': 'MY',
    'Indonesia': 'ID', 'Philippines': 'PH', 'Taiwan': 'TW', 'Hong Kong': 'HK',
    'Spain': 'ES', 'Portugal': 'PT'
}
SORTED_COUNTRIES = dict(sorted(COUNTRIES.items()))

# --- Cấu hình trang web ---
st.set_page_config(page_title="CÔNG CỤ CỐ VẤN CỦA LÝ VĂN HIỆP(KIOSOO)", page_icon="🧠", layout="wide")

# --- Khởi tạo Session State để quản lý API Key ---
if 'youtube_key_index' not in st.session_state:
    st.session_state.youtube_key_index = 0

# --- Các hàm tính toán và tiện ích ---
# @st.cache_data(ttl=3600) # Tạm thời tắt cache để kiểm tra lỗi 429 dễ hơn, sau có thể bật lại
def analyze_trends_data(keywords, country_code, timeframe, gprop):
    """
    HÀM ĐƯỢC NÂNG CẤP VỚI LOGIC TỰ ĐỘNG THỬ LẠI (EXPONENTIAL BACKOFF)
    """
    retries = 3  # Số lần thử lại tối đa
    delay = 1    # Thời gian chờ ban đầu (giây)
    
    for i in range(retries):
        try:
            # Thêm timeout để tránh ứng dụng bị treo
            pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25)) 
            pytrends.build_payload(kw_list=keywords, timeframe=timeframe, geo=country_code, gprop=gprop)
            
            interest_df = pytrends.interest_over_time()
            related_queries = pytrends.related_queries()

            if interest_df.empty:
                return None, None, "Không tìm thấy dữ liệu."
            
            interest_df.drop(columns=['isPartial'], inplace=True, errors='ignore')
            return interest_df, related_queries, None # Trả về kết quả nếu thành công

        except ReadTimeout:
             if i < retries - 1:
                st.toast(f"Google Trends phản hồi chậm, đang thử lại lần {i+1}/{retries-1}...")
                time.sleep(delay)
                delay *= 2 # Gấp đôi thời gian chờ
             else:
                return None, None, "Lỗi: Google Trends không phản hồi kịp thời. Vui lòng thử lại sau."

        except Exception as e:
            # Chỉ thử lại nếu gặp lỗi 429
            if 'response with code 429' in str(e):
                if i < retries - 1:
                    st.toast(f"Bị giới hạn truy cập, đang thử lại sau {delay} giây...")
                    time.sleep(delay)
                    delay *= 2 # Gấp đôi thời gian chờ cho lần thử tiếp theo
                else:
                    # Nếu thử hết số lần mà vẫn lỗi
                    return None, None, f"Lỗi: Google Trends đang tạm thời giới hạn truy cập. Vui lòng thử lại sau ít phút. (Code: 429)"
            else:
                # Nếu là lỗi khác, báo lỗi ngay lập tức
                return None, None, f"Đã xảy ra lỗi không xác định: {e}"
    
    return None, None, "Đã thử lại nhiều lần nhưng không thành công."


def search_youtube_videos_with_rotation(query):
    api_keys_str = st.secrets.get("YOUTUBE_API_KEYS", "")
    if not api_keys_str:
        return None, "Lỗi cấu hình: Vui lòng thêm YOUTUBE_API_KEYS vào Streamlit Secrets."
    
    api_keys = [key.strip() for key in api_keys_str.split(',')]
    start_index = st.session_state.youtube_key_index
    
    for i in range(len(api_keys)):
        current_key_index = (start_index + i) % len(api_keys)
        api_key = api_keys[current_key_index]
        
        try:
            youtube = build('youtube', 'v3', developerKey=api_key)
            request = youtube.search().list(q=query, part='snippet', maxResults=5, type='video', order='relevance')
            response = request.execute()
            st.session_state.youtube_key_index = (current_key_index + 1) % len(api_keys)
            videos = [{'id': item['id']['videoId'], 'title': item['snippet']['title'], 'channel': item['snippet']['channelTitle'], 'thumbnail': item['snippet']['thumbnails']['high']['url'], 'url': f'https://www.youtube.com/watch?v={item["id"]["videoId"]}'} for item in response.get('items', [])]
            return videos, None
        except HttpError as e:
            error_details = e.error_details
            is_quota_error = any(detail.get('reason') in ['quotaExceeded', 'dailyLimitExceeded'] for detail in error_details) if error_details else False
            if is_quota_error:
                print(f"Key {current_key_index + 1} đã hết quota. Đang chuyển sang key tiếp theo...")
                continue
            else:
                return None, f"Lỗi API với Key {current_key_index + 1}: {e}. Vui lòng kiểm tra lại key."
        except Exception as e:
            return None, f"Đã xảy ra lỗi không xác định: {e}"
    return None, "Tất cả các API key đều đã hết hạn ngạch hoặc không hợp lệ."

def calculate_potential_score(interest_series, related_queries_data):
    if interest_series.empty or interest_series.sum() == 0:
        return {'score': 0, 'label': "Không có dữ liệu", 'color': "#6c757d"}
    avg_interest = interest_series.mean()
    avg_interest_score = min((avg_interest / 80), 1) * 30 
    X = np.arange(len(interest_series)).reshape(-1, 1)
    y = interest_series.values
    model = LinearRegression().fit(X, y)
    slope = model.coef_[0]
    normalized_slope = (slope / (avg_interest + 1e-6)) * 100
    trend_score = 10 + normalized_slope
    trend_score = np.clip(trend_score, 0, 20)
    interest_score = avg_interest_score + trend_score
    growth_score = 0
    rising_df = related_queries_data.get('rising')
    if rising_df is not None and not rising_df.empty:
        def process_rising_value(val):
            if isinstance(val, str) and 'Breakout' in val: return 10000
            numeric_val = pd.to_numeric(str(val).replace('+','').replace('%','').replace(',',''), errors='coerce')
            return numeric_val if pd.notnull(numeric_val) else 0
        rising_df['value_numeric'] = rising_df['value'].apply(process_rising_value)
        num_rising_score = min(len(rising_df) / 10, 1) * 25
        avg_growth_score = min(rising_df['value_numeric'].mean() / 1000, 1) * 25
        growth_score = num_rising_score + avg_growth_score
    total_score = int(interest_score + growth_score)
    if total_score >= 70: label, color = 'Tiềm năng cao', '#28a745'
    elif total_score >= 40: label, color = 'Trung bình', '#ffc107'
    else: label, color = 'Bão hòa / Cạnh tranh cao', '#dc3545'
    return {'score': total_score, 'label': label, 'color': color, 'interest_score': interest_score, 'growth_score': growth_score, 'slope': slope, 'avg_interest': avg_interest, 'rising_df': rising_df}

def generate_advice(kw, metrics):
    score = metrics['score']
    full_advice = {
        "gold": f"**🟢 ĐÂY LÀ MỘT CƠ HỘI VÀNG!** ...",
        "popular": f"**🟢 CHỦ ĐỀ ĐANG RẤT THỊNH HÀNH!** ...",
        "sustainable": f"**🟡 CƠ HỘI NGÁCH BỀN VỮNG.** ...",
        "evergreen": f"**🟡 CHỦ ĐỀ 'EVERGREEN' CẦN TÌM NGÁCH.** ...",
        "saturated": f"**🔴 CẨN TRỌNG - THỊ TRƯỜNG BÃO HÒA.** ...",
        "low_interest": f"**🔴 CHỦ ĐỀ ÍT QUAN TÂM.** ..."
    } # Giữ ngắn gọn để tập trung vào logic
    if score >= 70: return full_advice["gold"] if metrics.get('growth_score', 0) > metrics.get('interest_score', 0) else full_advice["popular"]
    elif score >= 40: return full_advice["sustainable"] if metrics.get('slope', 0) > 0 else full_advice["evergreen"]
    else: return full_advice["saturated"] if metrics.get('avg_interest', 0) > 30 else full_advice["low_interest"]

# --- Giao diện người dùng ---
st.title("🧠 CÔNG CỤ CỐ VẤN CỦA LÝ VĂN HIỆP(KIOSOO)")
st.markdown("Phân tích, chấm điểm và đưa ra lời khuyên chiến lược cho các chủ đề của bạn.")
st.sidebar.header("⚙️ Tùy chọn Phân tích")
timeframe = st.sidebar.selectbox("1. Khung thời gian", [('7 ngày qua', 'now 7-d'), ('30 ngày qua', 'today 1-m'), ('90 ngày qua', 'today 3-m'), ('12 tháng qua', 'today 12-m'), ('5 năm qua', 'today 5-y'), ('Từ 2004', 'all')], format_func=lambda x: x[0])[1]
gprop = st.sidebar.selectbox("2. Nền tảng tìm kiếm", [('Web Search', ''), ('YouTube', 'youtube'), ('Google Images', 'images')], format_func=lambda x: x[0])[1]

with st.form("input_form"):
    col1, col2 = st.columns([3, 1])
    with col1: keywords_str = st.text_input("Nhập các từ khóa (cách nhau bằng dấu phẩy)", "nồi chiên không dầu, máy ép chậm, máy làm sữa hạt")
    with col2: country_name = st.selectbox("Chọn quốc gia", options=list(SORTED_COUNTRIES.keys()), index=0); country_code = SORTED_COUNTRIES[country_name]
    submitted = st.form_submit_button("💡 Phân tích & Tư vấn")

if submitted:
    keywords = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
    if not keywords or not country_code: st.warning("Vui lòng nhập từ khóa và chọn quốc gia.")
    else:
        with st.spinner("Đang phân tích và chấm điểm..."):
            interest_data, related_data, error = analyze_trends_data(keywords, country_code, timeframe, gprop)
        if error: st.error(error)
        elif interest_data is not None:
            st.header("1. Bảng điểm Tiềm năng")
            # Code hiển thị giữ nguyên
            score_cards = st.columns(len(keywords)); all_metrics = {}
            for i, kw in enumerate(keywords):
                metrics = calculate_potential_score(interest_data.get(kw, pd.Series()), related_data.get(kw, {}))
                all_metrics[kw] = metrics
                with score_cards[i]: st.markdown(f"""<div style="background-color:{metrics['color']}; color:white; padding: 20px; border-radius: 10px; text-align: center; height: 180px; display: flex; flex-direction: column; justify-content: center;"><h4 style="color: white; margin-bottom: 5px;">{kw.upper()}</h4><h1 style="color: white; font-size: 3.5rem; margin: 0;">{metrics['score']}</h1><p style="color: white; margin-top: 5px;">{metrics['label']}</p></div>""", unsafe_allow_html=True)
            
            st.header("2. Phân tích & Lời khuyên từ Cố vấn")
            for kw in keywords:
                with st.expander(f"**Xem phân tích chi tiết cho từ khóa: '{kw}'**"):
                    metrics = all_metrics[kw]
                    if 'avg_interest' in metrics:
                        advice = generate_advice(kw, metrics)
                        st.markdown(advice)
                    else:
                        st.info(f"Không có đủ dữ liệu xu hướng cho từ khóa '{kw}' để đưa ra lời khuyên chi tiết.")

            st.header("3. Biểu đồ so sánh Mức độ quan tâm")
            # Code hiển thị giữ nguyên
            fig, ax = plt.subplots(figsize=(15, 7))
            for kw in keywords:
                if kw in interest_data.columns:
                    ax.plot(interest_data.index, interest_data[kw], label=kw)
            ax.set_title(f"So sánh xu hướng tại '{country_name}'", fontsize=16); ax.legend(); ax.grid(True); st.pyplot(fig)

            st.header("4. Dữ liệu chi tiết (Insight Tăng trưởng)")
            # Code hiển thị giữ nguyên
            for kw in keywords:
                if kw in related_data and related_data[kw]:
                    with st.expander(f"**Xem Insight và Video hàng đầu cho: '{kw}'**"):
                        tab1, tab2, tab3 = st.tabs(["📊 Truy vấn Hàng đầu", "📈 Truy vấn Tăng trưởng", "🎬 Videos YouTube"])
                        with tab1: top_df = related_data[kw].get('top'); st.markdown("##### **Các từ khóa được tìm kiếm nhiều nhất liên quan**"); st.dataframe(top_df)
                        with tab2: rising_df = all_metrics[kw].get('rising_df'); st.markdown("##### **Các từ khóa có mức tăng trưởng tìm kiếm đột phá**"); st.dataframe(rising_df)
                        with tab3:
                            st.markdown("##### **5 Video YouTube hàng đầu cho từ khóa liên quan**")
                            if top_df is not None and not top_df.empty:
                                top_query = top_df['query'].iloc[0]
                                with st.spinner(f"Đang tìm video cho '{top_query}'..."): videos, error = search_youtube_videos_with_rotation(top_query)
                                if error: st.error(error)
                                elif videos:
                                    for video in videos:
                                        v_col1, v_col2 = st.columns([1, 4]);
                                        with v_col1: st.image(video['thumbnail'])
                                        with v_col2: st.markdown(f"**[{video['title']}]({video['url']})**"); st.caption(f"Kênh: {video['channel']}")
                                        st.markdown("---") 
                                else: st.info(f"Không tìm thấy video nào cho từ khóa '{top_query}'.")
                            else: st.info("Không có từ khóa hàng đầu để tìm kiếm video.")
```

---
### ## Giải pháp 2: Sử dụng Proxy xoay vòng (Rotating Proxies) - Tốt nhất & Chuyên nghiệp (Có phí)

Đây là giải pháp triệt để nhất, được sử dụng bởi các hệ thống chuyên nghiệp.

* **Logic:** Thay vì tất cả yêu cầu đều xuất phát từ một địa chỉ IP duy nhất của máy chủ Streamlit, chúng ta sẽ sử dụng một dịch vụ proxy để "che giấu" và thay đổi địa chỉ IP cho mỗi yêu cầu. Từ góc nhìn của Google, các yêu cầu này đến từ hàng trăm địa điểm khác nhau trên thế giới, khiến việc phát hiện và chặn trở nên gần như không thể. 

* **Ưu điểm:**
    * Hiệu quả gần như 100% trong việc chống lỗi 429.
    * Tốc độ có thể nhanh hơn vì các dịch vụ proxy thường được tối ưu hóa.
    * Là giải pháp có thể mở rộng (scalable) khi ứng dụng của bạn có nhiều người dùng.

* **Nhược điểm:**
    * **Có chi phí.** Bạn sẽ phải trả tiền hàng tháng cho một dịch vụ cung cấp proxy (ví dụ: Bright Data, ScraperAPI, Smartproxy).

* **Cách triển khai (Ví dụ):**
    1.  Đăng ký một dịch vụ proxy và lấy thông tin kết nối (host, port, username, password).
    2.  Lưu các thông tin này vào **Secrets** của Streamlit.
    3.  Cấu hình `pytrends` để sử dụng proxy đó.

    ```python
    # Ví dụ cách cấu hình pytrends với proxy
    
    # Lấy thông tin từ secrets
    proxy_host = st.secrets["proxy"]["host"]
    proxy_port = st.secrets["proxy"]["port"]
    proxy_user = st.secrets["proxy"]["user"]
    proxy_pass = st.secrets["proxy"]["password"]

    # Tạo chuỗi proxy
    proxy_url = f"http://{proxy_user}:{proxy_pass}@{proxy_host}:{proxy_port}"
    
    # Tạo dictionary proxies
    proxies = {
       'http': proxy_url,
       'https': proxy_url,
    }

    # Truyền vào pytrends
    pytrends = TrendReq(hl='en-US', tz=360, requests_args={'proxies': proxies})
    

