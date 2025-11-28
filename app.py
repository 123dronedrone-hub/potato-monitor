import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import datetime

# --- 頁面設定 ---
st.set_page_config(page_title="甘藷田間模擬系統", layout="wide", page_icon="🍠")

# --- 側邊欄設定 ---
st.sidebar.title("🛠️ 田區環境模擬器")
st.sidebar.subheader("1. 田區與陷阱設定")
field_center_lat = st.sidebar.number_input("田區中心緯度", value=23.9500, format="%.4f")
field_center_lon = st.sidebar.number_input("田區中心經度", value=120.4500, format="%.4f")
field_size = st.sidebar.slider("田區範圍半徑 (公尺)", 50, 500, 200)
trap_count = st.sidebar.slider("設置陷阱數量", 3, 20, 10)

st.sidebar.subheader("2. 作物參數")
planting_date = st.sidebar.date_input("甘藷種植日期", datetime.date(2023, 9, 1))
sim_duration = st.sidebar.slider("模擬天數 (從種植日開始)", 30, 150, 120)

st.sidebar.subheader("3. 蟲害與環境變數")
base_temp = st.sidebar.slider("平均氣溫 (°C)", 15, 35, 25)
pest_pressure = st.sidebar.select_slider("外部蟲源壓力", options=["低", "中", "高", "爆發"], value="中")
spray_day = st.sidebar.number_input("第幾天施藥 (0為不施藥)", 0, 150, 0)

# --- 核心邏輯 ---
def generate_simulation_data():
    data = []
    traps = []
    for i in range(trap_count):
        lat_offset = np.random.uniform(-1, 1) * (field_size / 111000)
        lon_offset = np.random.uniform(-1, 1) * (field_size / 111000)
        traps.append({
            'id': f'Trap_{i+1:02d}',
            'lat': field_center_lat + lat_offset,
            'lon': field_center_lon + lon_offset,
            'risk_factor': np.random.uniform(0.5, 1.5) 
        })
    pressure_map = {"低": 0.5, "中": 1.0, "高": 2.0, "爆發": 5.0}
    p_factor = pressure_map[pest_pressure]

    start_datetime = datetime.datetime.combine(planting_date, datetime.time(0,0))
    for day in range(sim_duration):
        sim_date = start_datetime + datetime.timedelta(days=day)
        daily_temp = base_temp + np.random.normal(0, 2) 
        
        # 生長階段
        growth_stage = ""
        crop_attraction = 1.0
        if day < 30:
            growth_stage = "建立期 (緩慢生長)"
            crop_attraction = 0.2
        elif 30 <= day < 60:
            growth_stage = "分枝期 (莖葉生長)"
            crop_attraction = 0.5
        elif 60 <= day < 90:
            growth_stage = "結薯期 (塊根開始膨大)"
            crop_attraction = 1.5 
        else:
            growth_stage = "塊根肥大期 (採收前)"
            crop_attraction = 2.5 

        # 施藥
        chemical_effect = 1.0
        if spray_day > 0 and day >= spray_day:
            days_after_spray = day - spray_day
            if days_after_spray < 14:
                chemical_effect = 0.1 + (days_after_spray * 0.05) 
        
        for trap in traps:
            temp_effect = max(0, (daily_temp - 15) * 0.5) 
            base_count = (temp_effect * crop_attraction * trap['risk_factor'] * p_factor)
            final_count = int(base_count * np.random.uniform(0.8, 1.2) * chemical_effect)
            final_count = max(0, final_count)
            alert = final_count > 30

            data.append({
                'days_after_planting': day,
                'trap_id': trap['id'], 'latitude': trap['lat'], 'longitude': trap['lon'],
                'temp': daily_temp, 'growth_stage': growth_stage,
                'count': final_count, 'alert': alert
            })
    return pd.DataFrame(data)

# --- UI 顯示 ---
if st.button("🚀 執行田間模擬運算", type="primary"):
    st.session_state['sim_data'] = generate_simulation_data()

if 'sim_data' in st.session_state:
    df = st.session_state['sim_data']
    latest_day = df['days_after_planting'].max()
    latest_df = df[df['days_after_planting'] == latest_day]
    
    st.markdown("---")
    st.header("📊 田間戰情儀表板")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("生長階段", latest_df.iloc[0]['growth_stage'])
    c2.metric("今日氣溫", f"{latest_df['temp'].mean():.1f} °C")
    c3.metric("總蟲數", f"{latest_df['count'].sum()} 隻")
    c4.metric("警報陷阱", f"{latest_df['alert'].sum()} 個")

    if latest_df['alert'].sum() > 0:
        st.error("⚠️ 警報！部分區域蟲數過高，請參考下方熱區圖進行防治。")

    t1, t2, t3 = st.tabs(["🗺️ 風險熱點圖", "📈 趨勢分析", "📋 數據表"])
    with t1:
        sel_day = st.slider("選擇日期", 0, sim_duration-1, latest_day)
        day_df = df[df['days_after_planting'] == sel_day]
        layer = pdk.Layer("ColumnLayer", data=day_df, get_position='[longitude, latitude]',
            get_elevation='count', elevation_scale=10, radius=15, get_fill_color='[count*4, 255-count*4, 0, 180]', pickable=True)
        st.pydeck_chart(pdk.Deck(initial_view_state=pdk.ViewState(latitude=field_center_lat, longitude=field_center_lon, zoom=16, pitch=50), layers=[layer], tooltip={"html": "蟲數: {count}"}))
    with t2:
        st.line_chart(df.groupby('days_after_planting')['count'].mean())
    with t3: st.dataframe(df)
else:
    st.info("👈 請在左側點擊「執行田間模擬運算」")