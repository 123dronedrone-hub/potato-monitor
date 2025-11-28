import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import datetime
from PIL import Image
from ultralytics import YOLO
import os

# --- 頁面設定 ---
st.set_page_config(page_title="甘藷田間智慧戰情室", layout="wide", page_icon="🍠")

# ==========================================
# 側邊欄：參數設定
# ==========================================
st.sidebar.title("⚙️ 參數設定")

st.sidebar.subheader("1. 田區幾何設定")
field_lat = st.sidebar.number_input("中心緯度", value=23.9500, format="%.4f")
field_lon = st.sidebar.number_input("中心經度", value=120.4500, format="%.4f")
field_width = st.sidebar.slider("田區寬度 (m)", 50, 500, 100)
field_length = st.sidebar.slider("田區長度 (m)", 50, 500, 150)

# 計算周長與建議陷阱數
perimeter = (field_width + field_length) * 2
suggested_traps = int(perimeter / 15)
st.sidebar.caption(f"周長 {perimeter}m | 建議陷阱數: {suggested_traps} 支")
trap_count = st.sidebar.number_input("實際陷阱數", 4, 100, suggested_traps)

st.sidebar.subheader("2. 模擬參數")
pest_direction = st.sidebar.selectbox("外部蟲源模擬方向", ["無", "北方", "東方", "南方", "西方", "東北角", "西南角"])
base_risk = st.sidebar.slider("基礎環境風險值", 0, 100, 10)

# ==========================================
# 函式庫
# ==========================================

def get_field_corners(lat, lon, w, l):
    # 簡易座標轉換
    meters_per_lat = 111000
    meters_per_lon = 111000 * np.cos(np.radians(lat))
    half_w = (w / 2) / meters_per_lon
    half_l = (l / 2) / meters_per_lat
    return [
        [lon - half_w, lat + half_l], # NW
        [lon - half_w, lat - half_l], # SW
        [lon + half_w, lat - half_l], # SE
        [lon + half_w, lat + half_l], # NE
        [lon - half_w, lat + half_l]  # Close
    ]

def generate_traps(lat, lon, w, l, n):
    corners = get_field_corners(lat, lon, w, l)
    traps = []
    
    # 沿著周長分佈 (簡化版：均勻分佈在四邊)
    # 這裡為了展示方便，直接在邊界上生成點
    poly_path = corners[:-1] # 去掉重複的終點
    
    total_len = (w + l) * 2
    step = total_len / n
    
    current_dist = 0
    # 簡單邏輯：將周長拉直，均勻撒點，再映射回座標 (這裡做簡易近似)
    # 為了確保四邊都有，我們強制分配
    side_counts = [int(n * (w/total_len)), int(n * (l/total_len)), int(n * (w/total_len))]
    side_counts.append(n - sum(side_counts)) # 剩餘給最後一邊
    
    trap_idx = 1
    sides_name = ["北邊 (North)", "西邊 (West)", "南邊 (South)", "東邊 (East)"] # 順序依 corners 定義調整
    
    # NW -> SW (West), SW -> SE (South), SE -> NE (East), NE -> NW (North)
    # 修正 corner 順序對應邊的名稱
    # corners: 0(NW), 1(SW), 2(SE), 3(NE)
    
    # Side 1: NW to SW (West)
    for i in range(side_counts[1]): # West
        r = i / max(side_counts[1], 1)
        t_lat = corners[0][1] + (corners[1][1] - corners[0][1]) * r
        t_lon = corners[0][0] + (corners[1][0] - corners[0][0]) * r
        traps.append({"id": f"T-{trap_idx:02d}", "lat": t_lat, "lon": t_lon, "side": "西方"})
        trap_idx += 1

    # Side 2: SW to SE (South)
    for i in range(side_counts[2]): # South
        r = i / max(side_counts[2], 1)
        t_lat = corners[1][1] + (corners[2][1] - corners[1][1]) * r
        t_lon = corners[1][0] + (corners[2][0] - corners[1][0]) * r
        traps.append({"id": f"T-{trap_idx:02d}", "lat": t_lat, "lon": t_lon, "side": "南方"})
        trap_idx += 1
        
    # Side 3: SE to NE (East)
    for i in range(side_counts[3]): # East
        r = i / max(side_counts[3], 1)
        t_lat = corners[2][1] + (corners[3][1] - corners[2][1]) * r
        t_lon = corners[2][0] + (corners[3][0] - corners[2][0]) * r
        traps.append({"id": f"T-{trap_idx:02d}", "lat": t_lat, "lon": t_lon, "side": "東方"})
        trap_idx += 1

    # Side 4: NE to NW (North)
    for i in range(side_counts[0]): # North
        r = i / max(side_counts[0], 1)
        t_lat = corners[3][1] + (corners[0][1] - corners[3][1]) * r
        t_lon = corners[3][0] + (corners[0][0] - corners[3][0]) * r
        traps.append({"id": f"T-{trap_idx:02d}", "lat": t_lat, "lon": t_lon, "side": "北方"})
        trap_idx += 1
        
    return pd.DataFrame(traps)

# ==========================================
# APP 主畫面
# ==========================================

st.title("🍠 甘藷田間智慧戰情室")

# 初始化 Session State (用於儲存數據)
if 'trap_data' not in st.session_state:
    st.session_state['trap_data'] = generate_traps(field_lat, field_lon, field_width, field_length, trap_count)
    st.session_state['trap_data']['count'] = 0 # 初始數量
    st.session_state['prev_count'] = 0 # 上期數量 (用於比較)

# 1. 數據管理區塊 (Hybrid Data Input)
with st.expander("📝 陷阱數據管理 (模擬生成 / 手動修改 / AI 辨識)", expanded=True):
    col_mgmt_1, col_mgmt_2 = st.columns([1, 2])
    
    with col_mgmt_1:
        st.subheader("1. 數據來源")
        mode = st.radio("選擇模式", ["全自動模擬生成", "手動/AI 修正模式"])
        
        if mode == "全自動模擬生成":
            if st.button("🎲 生成本期模擬數據"):
                df = st.session_state['trap_data'].copy()
                # 備份舊數據
                st.session_state['prev_count'] = df['count'].copy()
                
                # 生成新數據
                for index, row in df.iterrows():
                    risk = base_risk
                    if pest_direction in row['side']: risk *= 3
                    if pest_direction == "東北角" and row['side'] in ["北方", "東方"]: risk *= 2.5
                    if pest_direction == "西南角" and row['side'] in ["南方", "西方"]: risk *= 2.5
                    
                    # 隨機生成
                    new_val = int(np.random.normal(risk, risk*0.5))
                    df.at[index, 'count'] = max(0, new_val)
                
                st.session_state['trap_data'] = df
                st.success("模擬數據已更新！")
                
        else: # 手動模式
            st.info("請在右側表格直接修改數值，或使用下方 AI 輔助填入。")
            
            # AI 輔助區塊
            st.markdown("---")
            st.markdown("##### 🤖 AI 影像辨識填入")
            
            # 模型上傳
            model_file = st.file_uploader("步驟 A: 載入模型 (best.pt)", type=['pt'], key="model_uploader")
            if model_file:
                with open("temp_best.pt", "wb") as f:
                    f.write(model_file.getbuffer())
                st.success("模型已載入")
            
            # 選擇陷阱與上傳照片
            target_trap = st.selectbox("步驟 B: 選擇要更新的陷阱", st.session_state['trap_data']['id'].unique())
            trap_img = st.file_uploader(f"步驟 C: 上傳 {target_trap} 的照片", type=['jpg', 'png'])
            
            if trap_img and os.path.exists("temp_best.pt"):
                if st.button("📸 執行 AI 計數並寫入"):
                    try:
                        model = YOLO("temp_best.pt")
                        img = Image.open(trap_img)
                        res = model.predict(img)
                        count = len(res[0].boxes)
                        
                        # 更新 Session State
                        idx = st.session_state['trap_data'].index[st.session_state['trap_data']['id'] == target_trap].tolist()[0]
                        st.session_state['trap_data'].at[idx, 'count'] = count
                        st.success(f"辨識成功！{target_trap} 數量已更新為 {count} 隻。")
                        
                    except Exception as e:
                        st.error(f"辨識失敗: {e}")
            elif trap_img and not os.path.exists("temp_best.pt"):
                st.warning("請先完成步驟 A (上傳模型)。")

    with col_mgmt_2:
        st.subheader("2. 當前陷阱數據表")
        # 使用 Data Editor 允許直接修改
        edited_df = st.data_editor(
            st.session_state['trap_data'],
            column_config={
                "count": st.column_config.NumberColumn("蟻象數量 (可編輯)", help="點擊修改數值", min_value=0, step=1),
                "lat": st.column_config.NumberColumn("緯度", format="%.5f"),
                "lon": st.column_config.NumberColumn("經度", format="%.5f"),
            },
            disabled=["id", "side", "lat", "lon"],
            use_container_width=True,
            key="data_editor_table"
        )
        # 即使在手動模式，Data Editor 的變更也會同步
        if not edited_df.equals(st.session_state['trap_data']):
             st.session_state['trap_data'] = edited_df
             st.rerun()

# 2. 警報分析區塊 (Alert Logic)
df_curr = st.session_state['trap_data']
total_curr = df_curr['count'].sum()
try:
    total_prev = st.session_state['prev_count'].sum() if isinstance(st.session_state['prev_count'], pd.Series) else 0
except:
    total_prev = 0

growth_rate = ((total_curr - total_prev) / total_prev * 100) if total_prev > 0 else 0

st.markdown("---")
col_metric_1, col_metric_2, col_metric_3 = st.columns(3)
col_metric_1.metric("本期全區總蟲數", f"{total_curr} 隻")
col_metric_2.metric("較上期增減", f"{growth_rate:.1f} %", delta_color="inverse")
col_metric_3.metric("高風險陷阱數 (>30隻)", f"{len(df_curr[df_curr['count']>30])} 個")

# --- 警報邏輯 ---
if growth_rate > 100:
    st.error(f"🚨 **嚴重警報：蟲數暴增！** 本期增長率 ({growth_rate:.1f}%) 超過 100%，請立即檢查環境！")
elif growth_rate > 50:
    st.warning(f"⚠️ **警戒：** 蟲數顯著增加 ({growth_rate:.1f}%)，請密切注意。")

# 3. 視覺化區塊 (Heatmap Visualization)
st.subheader("📍 田區風險熱力圖 (Risk Heatmap)")
st.caption("說明：藍色=安全/低密度，綠色=警戒/中密度，紅色=危險/高密度。色塊向外延伸代表潛在風險範圍。")

corners = get_field_corners(field_lat, field_lon, field_width, field_length)

# Heatmap 設定
layer = pdk.Layer(
    "HeatmapLayer",
    data=df_curr,
    get_position='[lon, lat]',
    get_weight="count",
    opacity=0.7,
    # 色彩漸層: 藍 -> 綠 -> 黃 -> 紅
    color_range=[
        [65, 105, 225],  # RoyalBlue (低)
        [0, 255, 127],   # SpringGreen (中)
        [255, 255, 0],   # Yellow (高)
        [220, 20, 60]    # Crimson (極高)
    ],
    threshold=0.1,      # 過濾掉極低值
    radiusPixels=60,    # 半徑 (像素)，調大可以讓顏色融合更連續，並顯示向外擴散的效果
    intensity=1.5,      # 強度
)

# 田區邊界框線 (Polygon)
poly_layer = pdk.Layer(
    "PolygonLayer",
    data=[{"polygon": corners}],
    get_polygon="polygon",
    filled=False,       # 不填滿，只畫框
    stroked=True,
    get_line_color=[255, 255, 255], # 白色框線
    get_line_width=3,
)

# 文字標籤 (顯示數量)
text_layer = pdk.Layer(
    "TextLayer",
    data=df_curr,
    get_position='[lon, lat]',
    get_text='count',
    get_color=[0, 0, 0],
    get_size=15,
    get_alignment_baseline="'bottom'",
    get_background_color=[255, 255, 255, 200], # 白底背景讓字清楚
    pickable=True
)

view_state = pdk.ViewState(
    latitude=field_lat, 
    longitude=field_lon, 
    zoom=16,
    pitch=0 # 俯視視角較適合看熱力圖
)

st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/satellite-v9', # 使用衛星地圖更有田間感
    initial_view_state=view_state,
    layers=[layer, poly_layer, text_layer], # 熱力圖在底層，邊框在上
    tooltip={"html": "<b>陷阱 ID:</b> {id}<br/><b>數量:</b> {count}"}
))
