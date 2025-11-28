import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import datetime
from PIL import Image
from ultralytics import YOLO
import os

# --- 頁面設定 ---
st.set_page_config(page_title="甘藷田間智慧監測系統 Pro", layout="wide", page_icon="🍠")

# ==========================================
# 側邊欄：參數設定
# ==========================================
st.sidebar.title("⚙️ 參數設定控制台")

# 1. 田區設定 (矩形 + 周邊佈署)
st.sidebar.subheader("1. 田區與佈署")
field_lat = st.sidebar.number_input("田區中心緯度", value=23.9500, format="%.4f")
field_lon = st.sidebar.number_input("田區中心經度", value=120.4500, format="%.4f")

col_w, col_h = st.sidebar.columns(2)
field_width = col_w.number_input("田區寬度 (公尺)", 50, 500, 100) # 東西向
field_length = col_h.number_input("田區長度 (公尺)", 50, 500, 150) # 南北向

# 計算周長與建議陷阱數
perimeter = (field_width + field_length) * 2
suggested_traps = int(perimeter / 15) # 每15公尺一支
min_traps = 4 # 每邊至少一支

st.sidebar.info(f"田區周長: {perimeter}m | 建議陷阱數 (15m間隔): {suggested_traps} 支")
trap_count = st.sidebar.slider("實際設置陷阱數", min_traps, max(suggested_traps + 5, 20), suggested_traps)

# 2. 作物與環境
st.sidebar.subheader("2. 環境模擬參數")
planting_date = st.sidebar.date_input("種植日期", datetime.date(2023, 9, 1))
sim_days = st.sidebar.slider("模擬天數", 30, 150, 120)
pest_source_direction = st.sidebar.selectbox("主要蟲源方向 (模擬入侵)", ["無特定", "北方", "東方", "南方", "西方", "東北角"])

# ==========================================
# 核心邏輯：矩形周邊佈點演算法
# ==========================================
def generate_perimeter_traps(center_lat, center_lon, width, length, num_traps):
    # 簡易座標轉換: 1度緯度 ~= 111km, 1度經度 ~= 111km * cos(lat)
    meters_per_lat = 111000
    meters_per_lon = 111000 * np.cos(np.radians(center_lat))
    
    half_w = (width / 2) / meters_per_lon
    half_l = (length / 2) / meters_per_lat
    
    # 定義四個角點 (逆時針: 左上, 左下, 右下, 右上)
    corners = [
        (center_lon - half_w, center_lat + half_l), # NW
        (center_lon - half_w, center_lat - half_l), # SW
        (center_lon + half_w, center_lat - half_l), # SE
        (center_lon + half_w, center_lat + half_l), # NE
        (center_lon - half_w, center_lat + half_l), # Close loop
    ]
    
    traps = []
    # 沿著周長均勻分布
    total_len = (width + length) * 2
    step = total_len / num_traps
    
    # 這裡使用簡化的邏輯將陷阱分配到四邊
    n_side1 = int(num_traps * (width / total_len))
    n_side2 = int(num_traps * (length / total_len))
    n_side3 = int(num_traps * (width / total_len))
    n_side4 = num_traps - n_side1 - n_side2 - n_side3 
    
    # 生成座標函數
    def make_line(start_p, end_p, n, side_name):
        pts = []
        for i in range(n):
            r = i / max(n, 1)
            lon = start_p[0] + (end_p[0] - start_p[0]) * r
            lat = start_p[1] + (end_p[1] - start_p[1]) * r
            pts.append({"lat": lat, "lon": lon, "side": side_name})
        return pts

    traps.extend(make_line(corners[0], corners[3], n_side1, "北邊 (North)"))
    traps.extend(make_line(corners[3], corners[2], n_side2, "東邊 (East)"))
    traps.extend(make_line(corners[2], corners[1], n_side3, "南邊 (South)"))
    traps.extend(make_line(corners[1], corners[0], n_side4, "西邊 (West)"))
    
    # 賦予 ID 與風險係數
    res = []
    for i, t in enumerate(traps):
        risk = 1.0
        if pest_source_direction == "北方" and "North" in t['side']: risk = 3.0
        if pest_source_direction == "東方" and "East" in t['side']: risk = 3.0
        if pest_source_direction == "南方" and "South" in t['side']: risk = 3.0
        if pest_source_direction == "西方" and "West" in t['side']: risk = 3.0
        if pest_source_direction == "東北角" and ("North" in t['side'] or "East" in t['side']): risk = 2.5

        res.append({
            "id": f"T-{i+1:02d}",
            "lat": t['lat'],
            "lon": t['lon'],
            "side": t['side'],
            "risk_factor": risk
        })
    return res, corners

# ==========================================
# 應用程式本體
# ==========================================

st.title("🍠 甘藷田間智慧監測系統 Pro")
st.caption("整合田區邊界模擬、風險熱圖與 AI 模型實測")

tab1, tab2 = st.tabs(["📊 田區模擬與風險監測", "🤖 AI 模型辨識驗證"])

# --- TAB 1: 模擬器 ---
with tab1:
    if st.button("🚀 執行田區模擬", type="primary"):
        # 1. 生成陷阱
        traps, corners = generate_perimeter_traps(field_lat, field_lon, field_width, field_length, trap_count)
        
        # 2. 生成時間序列數據
        data = []
        for day in range(sim_days):
            curr_date = planting_date + datetime.timedelta(days=day)
            
            # 生長週期係數
            growth_factor = 0.5
            if 60 <= day <= 120: growth_factor = 2.0 
            
            # 氣候係數
            weather_factor = np.random.uniform(0.8, 1.2)
            
            for t in traps:
                count = int(5 * growth_factor * weather_factor * t['risk_factor'] * np.random.uniform(0.5, 1.5))
                data.append({
                    "date": curr_date,
                    "day": day,
                    "trap_id": t['id'],
                    "latitude": t['lat'],
                    "longitude": t['lon'],
                    "side": t['side'],
                    "count": count
                })
        
        df = pd.DataFrame(data)
        
        # [關鍵修正]：將日期物件轉為文字，避免地圖繪製時發生 JSON Error
        df['date'] = df['date'].astype(str)
        
        st.session_state['sim_df'] = df
        st.session_state['corners'] = corners 

    # 顯示結果
    if 'sim_df' in st.session_state:
        df = st.session_state['sim_df']
        corners = st.session_state['corners']
        latest_day = df['day'].max()
        latest_df = df[df['day'] == latest_day]
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("📍 田區風險熱點圖 (最新數據)")
            
            # PyDeck 地圖
            polygon_layer = pdk.Layer(
                "PolygonLayer",
                data=[{"polygon": [[p[0], p[1]] for p in corners]}],
                get_polygon="polygon",
                filled=True,
                get_fill_color=[144, 238, 144, 50],
                get_line_color=[0, 100, 0],
                get_line_width=2,
                line_width_min_pixels=1,
            )
            
            scatter_layer = pdk.Layer(
                "ScatterplotLayer",
                data=latest_df,
                get_position='[longitude, latitude]',
                get_radius=8,
                get_fill_color='[count > 30 ? 255 : 0, count > 30 ? 0 : 128, 0, 200]',
                pickable=True,
                auto_highlight=True
            )
            
            text_layer = pdk.Layer(
                "TextLayer",
                data=latest_df,
                get_position='[longitude, latitude]',
                get_text='trap_id',
                get_color=[0, 0, 0],
                get_size=12,
                get_alignment_baseline="'bottom'",
            )

            view_state = pdk.ViewState(latitude=field_lat, longitude=field_lon, zoom=16)
            st.pydeck_chart(pdk.Deck(
                map_style='mapbox://styles/mapbox/light-v9',
                initial_view_state=view_state,
                layers=[polygon_layer, scatter_layer, text_layer],
                tooltip={"html": "<b>{trap_id}</b> ({side})<br/>蟲數: {count}"}
            ))

        with col2:
            st.subheader("📋 蟲害重點")
            total = latest_df['count'].sum()
            avg = latest_df['count'].mean()
            st.metric("全區總蟲數", f"{total}")
            st.metric("平均單一陷阱", f"{avg:.1f}")
            
            risk_side = latest_df.groupby('side')['count'].mean().idxmax()
            st.error(f"⚠️ 高風險方位: **{risk_side}**")
            st.markdown("建議檢查該方位之外部蟲源。")

        st.subheader("📈 自家田區趨勢分析")
        
        # 繪製折線圖
        trend_data = df.pivot_table(index='date', columns='side', values='count', aggfunc='mean')
        st.line_chart(trend_data)

# --- TAB 2: AI 驗證 ---
with tab2:
    st.header("🔬 AI 模型辨識與驗證")
    st.markdown("""
    在此上傳您的 **模型 (.pt)** 與 **陷阱照片**，系統將進行計數，並讓您輸入實際數量以驗證準確度。
    """)

    col_model, col_img = st.columns(2)
    
    with col_model:
        model_file = st.file_uploader("1. 上傳訓練好的模型 (best.pt)", type=['pt'])
    
    with col_img:
        img_file = st.file_uploader("2. 上傳陷阱照片", type=['jpg', 'png', 'jpeg'])

    if model_file and img_file:
        with open("temp_model.pt", "wb") as f:
            f.write(model_file.getbuffer())
        
        try:
            model = YOLO("temp_model.pt")
            
            image = Image.open(img_file)
            st.image(image, caption="原始照片", use_container_width=True)
            
            if st.button("🔍 開始辨識計數"):
                with st.spinner("AI 正在數蟲..."):
                    results = model.predict(image)
                    ai_count = len(results[0].boxes)
                    res_plotted = results[0].plot()
                    
                    st.image(res_plotted, caption=f"AI 辨識結果: {ai_count} 隻", use_container_width=True)
                    
                    st.markdown("---")
                    st.subheader("📝 準確度驗證")
                    real_count = st.number_input("請輸入人工清點的真實數量 (Ground Truth)", min_value=0, value=ai_count)
                    
                    if st.button("計算誤差"):
                        diff = abs(ai_count - real_count)
                        accuracy = 100 * (1 - diff / max(real_count, 1)) if real_count > 0 else 0
                        
                        c1, c2, c3 = st.columns(3)
                        c1.metric("AI 計數", ai_count)
                        c2.metric("人工計數", real_count)
                        c3.metric("準確率", f"{accuracy:.1f}%")
                        
                        if accuracy > 90:
                            st.success("模型表現優異！")
                        elif accuracy > 70:
                            st.warning("模型表現尚可，建議增加更多樣本訓練。")
                        else:
                            st.error("準確度較低，請檢查模型或照片清晰度。")
                            
        except Exception as e:
            st.error(f"模型載入失敗，請確認檔案是否為 YOLOv8 格式。錯誤訊息: {e}")
