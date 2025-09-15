# -*- coding: utf-8 -*-
"""
CÔNG CỤ CỐ VẤN TỪ KHÓA CỦA LVH-KIOSOO
Xây dựng bằng Streamlit.
Để chạy, bạn cần cài đặt: pip install streamlit pandas pytrends matplotlib scikit-learn google-api-python-client
Sau đó chạy lệnh: streamlit run streamlit_trend_app.py
"""
import streamlit as st
from pytrends.request import TrendReq
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

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
st.set_page_config(page_title="Công Cụ Cố Vấn Từ Khóa của LVH-Kiosoo", page_icon="🧠", layout="wide")

# --- Khởi tạo Session State để quản lý API Key ---
if 'youtube_key_index' not in st.session_state:
    st.session_state.youtube_key_index = 0

# --- Các hàm tính toán và tiện ích ---
@st.cache_data(ttl=3600)
def analyze_trends_data(keywords, country_code, timeframe, gprop):
    try:
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload(kw_list=keywords, timeframe=timeframe, geo=country_code, gprop=gprop)
        interest_df = pytrends.interest_over_time()
        related_queries = pytrends.related_queries()
        if interest_df.empty:
            return None, None, "Không tìm thấy dữ liệu."
        interest_df.drop(columns=['isPartial'], inplace=True, errors='ignore')
        return interest_df, related_queries, None
    except Exception as e:
        return None, None, f"Đã xảy ra lỗi khi lấy dữ liệu Google Trends: {e}"

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
            request = youtube.search().list(
                q=query, part='snippet', maxResults=5, type='video', order='relevance'
            )
            response = request.execute()
            
            st.session_state.youtube_key_index = (current_key_index + 1) % len(api_keys)
            
            videos = []
            for item in response.get('items', []):
                videos.append({
                    'id': item['id']['videoId'],
                    'title': item['snippet']['title'],
                    'channel': item['snippet']['channelTitle'],
                    'thumbnail': item['snippet']['thumbnails']['high']['url'],
                    'url': f'https://www.youtube.com/watch?v={item["id"]["videoId"]}'
                })
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
        num_rising = len(rising_df)
        avg_growth = rising_df['value_numeric'].mean()
        num_rising_score = min(num_rising / 10, 1) * 25
        avg_growth_score = min(avg_growth / 1000, 1) * 25
        growth_score = num_rising_score + avg_growth_score
    total_score = int(interest_score + growth_score)
    if total_score >= 70: label, color = 'Tiềm năng cao', '#28a745'
    elif total_score >= 40: label, color = 'Trung bình', '#ffc107'
    else: label, color = 'Bão hòa / Cạnh tranh cao', '#dc3545'
    return {'score': total_score, 'label': label, 'color': color, 'interest_score': interest_score, 'growth_score': growth_score, 'slope': slope, 'avg_interest': avg_interest, 'rising_df': rising_df}

def generate_advice(kw, metrics):
    score = metrics['score']
    if score >= 70:
        if metrics['growth_score'] > metrics['interest_score']: advice = f"**🟢 ĐÂY LÀ MỘT CƠ HỘI VÀNG!** Điểm tiềm năng cao của **'{kw}'** chủ yếu đến từ **sự bùng nổ của các thị trường ngách** liên quan... \n\n**Chiến lược:** **Hành động nhanh!**..."
        else: advice = f"**🟢 CHỦ ĐỀ ĐANG RẤT THỊNH HÀNH!** **'{kw}'** đang có mức độ quan tâm rất cao và ổn định... \n\n**Chiến lược:** Cạnh tranh sẽ rất cao..."
    elif score >= 40:
        if metrics['slope'] > 0: advice = f"**🟡 CƠ HỘI NGÁCH BỀN VỮNG.** **'{kw}'** là một chủ đề có sự tăng trưởng ổn định... \n\n**Chiến lược:** Tập trung vào việc **giải quyết các vấn đề cụ thể**..."
        else: advice = f"**🟡 CHỦ ĐỀ 'EVERGREEN' CẦN TÌM NGÁCH.** **'{kw}'** có một lượng khán giả ổn định... \n\n**Chiến lược:** Hãy **đào thật sâu** vào một khía cạnh rất nhỏ..."
    else:
        if metrics['avg_interest'] > 30: advice = f"**🔴 CẨN TRỌNG - THỊ TRƯỜNG BÃO HÒA.** **'{kw}'** có thể có lượng tìm kiếm cao... \n\n**Chiến lược:** **Nên tránh** nếu bạn là người mới..."
        else: advice = f"**🔴 CHỦ ĐỀ ÍT QUAN TÂM.** Lượng tìm kiếm cho **'{kw}'** hiện tại rất thấp... \n\n**Chiến lược:** Hãy sử dụng công cụ để tìm kiếm các từ khóa khác..."
    return advice

# --- Giao diện người dùng ---
st.title("🧠 Công Cụ Cố Vấn Từ Khóa của LVH-Kiosoo")
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
        with st.spinner("Đang phân tích và chấm điểm..."): interest_data, related_data, error = analyze_trends_data(keywords, country_code, timeframe, gprop)
        if error: st.error(error)
        elif interest_data is not None:
            st.header("1. Bảng điểm Tiềm năng")
            score_cards = st.columns(len(keywords)); all_metrics = {}
            for i, kw in enumerate(keywords):
                metrics = calculate_potential_score(interest_data[kw], related_data.get(kw, {})); all_metrics[kw] = metrics
                with score_cards[i]: st.markdown(f"""<div style="background-color:{metrics['color']}; color:white; padding: 20px; border-radius: 10px; text-align: center; height: 180px; display: flex; flex-direction: column; justify-content: center;"><h4 style="color: white; margin-bottom: 5px;">{kw.upper()}</h4><h1 style="color: white; font-size: 3.5rem; margin: 0;">{metrics['score']}</h1><p style="color: white; margin-top: 5px;">{metrics['label']}</p></div>""", unsafe_allow_html=True)
            st.header("2. Phân tích & Lời khuyên từ Cố vấn")
            for kw in keywords:
                with st.expander(f"**Xem phân tích chi tiết cho từ khóa: '{kw}'**"): st.markdown(generate_advice(kw, all_metrics[kw]))
            st.header("3. Biểu đồ so sánh Mức độ quan tâm")
            fig, ax = plt.subplots(figsize=(15, 7))
            for kw in keywords: ax.plot(interest_data.index, interest_data[kw], label=kw)
            ax.set_title(f"So sánh xu hướng tại '{country_name}'", fontsize=16); ax.legend(); ax.grid(True); st.pyplot(fig)
            st.header("4. Dữ liệu chi tiết (Insight Tăng trưởng)")
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
                                        v_col1, v_col2 = st.columns([1, 4])
                                        with v_col1: st.image(video['thumbnail'])
                                        with v_col2: st.markdown(f"**[{video['title']}]({video['url']})**"); st.caption(f"Kênh: {video['channel']}")
                                        st.markdown("---") 
                                else: st.info(f"Không tìm thấy video nào cho từ khóa '{top_query}'.")
                            else: st.info("Không có từ khóa hàng đầu để tìm kiếm video.")