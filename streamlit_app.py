# 실행: streamlit run streamlit_app.py

import io
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import requests
from requests.adapters import HTTPAdapter, Retry

# -----------------------------
# Streamlit 페이지 설정
# -----------------------------
st.set_page_config(
    page_title="미세먼지 종합 분석 대시보드",
    page_icon="🌫️",
    layout="wide"
)

# -----------------------------
# 메인 제목과 서론
# -----------------------------
st.title("🌫️ 미림마이스터고 1학년 4반 학생을 위한 미세먼지 종합 분석")
st.markdown("""
### 📖 문제 제기
최근 몇 년 동안 날씨 예보에서 가장 자주 등장하는 단어 중 하나가 ‘미세먼지’입니다.
등굣길 아침마다 마스크를 쓰고, 체육 수업이 실내로 바뀌는 일이 흔해졌습니다. 하지만 단순히 불편함을 넘어서, 미세먼지가 우리 건강과 생활에 어떤 영향을 주는지, 그리고 청소년인 우리가 스스로 어떻게 대응해야 하는지는 제대로 이야기되지 않습니다.
그래서 우리는 미세먼지 문제를 데이터와 실제 사례를 통해 확인하고, 학생으로서 실천할 수 있는 방법을 제시하기 위해 이 대시보드를 제작하게 되었습니다.
""")
st.markdown("---")

# -----------------------------
# 안정적 API 요청 세션
# -----------------------------
@st.cache_resource
def get_requests_session():
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session

# -----------------------------
# World Bank API 데이터 로더
# -----------------------------
@st.cache_data(ttl=3600)
def fetch_worldbank_data(country_code: str, indicator: str, start_year: int, end_year: int):
    """지정된 국가, 지표, 기간에 대한 World Bank 데이터를 가져옵니다."""
    session = get_requests_session()
    all_data = []
    url = f"http://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}?date={start_year}:{end_year}&format=json&per_page=100"
    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if len(data) > 1 and data[1]:
            for record in data[1]:
                if record["value"] is not None:
                    all_data.append({
                        "Year": int(record["date"]),
                        "Value": float(record["value"]),
                        "CountryCode": record["countryiso3code"],
                        "CountryName": record["country"]["value"]
                    })
        return pd.DataFrame(all_data)
    except requests.exceptions.RequestException:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# ============================================================================
# 사이드바 컨트롤
# ============================================================================
st.sidebar.header("📊 데이터 분석 옵션")
start_year, end_year = st.sidebar.slider(
    "기간 선택", 1990, 2022, (2000, 2021)
)
smooth = st.sidebar.checkbox("추세선 표시 (5년 이동평균)", value=True)

all_countries = {
    'KOR': '한국', 'CHN': '중국', 'JPN': '일본', 'USA': '미국', 'OED': 'OECD 평균',
    'DEU': '독일', 'GBR': '영국', 'FRA': '프랑스', 'IND': '인도', 'MNG': '몽골'
}
country_name_to_code = {v: k for k, v in all_countries.items()}

selected_countries_names = st.sidebar.multiselect(
    "비교할 국가 선택",
    options=list(all_countries.values()),
    default=['한국', '중국', '일본', '미국', 'OECD 평균']
)

# ============================================================================
# Section 1: 한국 연도별 미세먼지 데이터
# ============================================================================
st.header("📈 한국 연도별 미세먼지(PM2.5) 농도 추이")

pm25_indicator = "EN.ATM.PM25.MC.M3"
df = fetch_worldbank_data("KOR", pm25_indicator, start_year, end_year)
df = df.rename(columns={"Value": "PM2.5"}).sort_values("Year").reset_index(drop=True)

if df.empty:
    st.error("❌ 한국 데이터를 불러올 수 없습니다. 다른 기간을 선택해 주세요.")
    st.stop()

if smooth and len(df) >= 5:
    df["PM2.5_smooth"] = df["PM2.5"].rolling(5, center=True, min_periods=1).mean()

st.warning("📢 **세계보건기구(WHO) 권고 기준**: 연평균 초미세먼지(PM2.5) 농도는 **5㎍/㎥** 이하입니다.")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("최근 농도", f"{df['PM2.5'].iloc[-1]:.1f} ㎍/㎥", f"({df['Year'].iloc[-1]}년)")
with col2:
    st.metric("기간 내 평균 농도", f"{df['PM2.5'].mean():.1f} ㎍/㎥")
with col3:
    change = df['PM2.5'].iloc[-1] - df['PM2.5'].iloc[0]
    st.metric("농도 변화량", f"{change:.1f} ㎍/㎥", delta_color="inverse")
with col4:
    max_val_row = df.loc[df['PM2.5'].idxmax()]
    st.metric("기간 내 최고 농도", f"{max_val_row['PM2.5']:.1f} ㎍/㎥", f"({int(max_val_row['Year'])}년)")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Year"], y=df["PM2.5"], mode='lines+markers', name='연도별 PM2.5 농도'))
if smooth and 'PM2.5_smooth' in df.columns:
    fig.add_trace(go.Scatter(x=df["Year"], y=df["PM2.5_smooth"], mode='lines', name='추세선', line=dict(color='red', dash='dash')))
fig.update_layout(title=f"한국 연도별 미세먼지(PM2.5) 농도 ({start_year}-{end_year})", xaxis_title="연도", yaxis_title="농도 (㎍/㎥)", template='plotly_white', height=500)
st.plotly_chart(fig, use_container_width=True)

st.info("👨‍⚕️ **WHO**: '미세먼지는 모든 연령대의 사람들에게 영향을 미치는 '보이지 않는 살인자'이며, 특히 어린이와 청소년의 건강한 성장을 저해합니다.'")


# ============================================================================
# Section 2: 원인 분석 및 국가 비교
# ============================================================================
st.markdown("---")
st.header("🌏 원인 분석 및 국가 비교")

st.subheader("🔗 이산화탄소(CO2) 배출량과 미세먼지 농도의 관계")
st.markdown("""
미세먼지와 이산화탄소는 발생원이 상당 부분 겹칩니다. 화석연료(석탄, 석유 등)를 연소시킬 때 두 물질이 함께 배출되는 경우가 많기 때문입니다.
아래 그래프는 한국의 1인당 CO2 배출량과 PM2.5 농도 추이를 비교하여 두 지표 간의 관계를 보여줍니다.
""")
co2_indicator = "EN.ATM.CO2E.PC" # 1인당 CO2 배출량
df_co2 = fetch_worldbank_data("KOR", co2_indicator, start_year, end_year)

if not df_co2.empty and not df.empty:
    df_merged = pd.merge(df[['Year', 'PM2.5']], df_co2[['Year', 'Value']], on="Year")
    df_merged = df_merged.rename(columns={"Value": "CO2_per_capita"})

    fig_corr = make_subplots(specs=[[{"secondary_y": True}]])
    fig_corr.add_trace(go.Scatter(x=df_merged['Year'], y=df_merged['PM2.5'], name='PM2.5 농도 (㎍/㎥)', line=dict(color='#1f77b4')), secondary_y=False)
    fig_corr.add_trace(go.Scatter(x=df_merged['Year'], y=df_merged['CO2_per_capita'], name='1인당 CO2 배출량 (톤)', line=dict(color='darkred')), secondary_y=True)
    fig_corr.update_layout(title_text='PM2.5 농도와 1인당 CO2 배출량 추이 비교', template='plotly_white', height=500)
    fig_corr.update_xaxes(title_text="연도")
    fig_corr.update_yaxes(title_text="PM2.5 농도 (㎍/㎥)", secondary_y=False)
    fig_corr.update_yaxes(title_text="1인당 CO2 배출량 (톤)", secondary_y=True)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.success("📈 **핵심 관찰**: 2000년대 초반까지 두 지표가 동반 상승하는 경향을 보이다가, 이후 CO2 배출량은 정체되는 반면 PM2.5 농도는 변동성을 보입니다. 이는 산업 구조 변화와 환경 정책의 영향으로 해석될 수 있습니다.")

    correlation = df_merged['PM2.5'].corr(df_merged['CO2_per_capita'])
    
    c1, c2 = st.columns(2)
    with c1:
        st.metric("피어슨 상관계수", f"{correlation:.3f}")
        if correlation > 0.4:
            st.warning("양의 상관관계가 존재합니다.")
        else:
            st.info("상관관계가 약하거나 없습니다.")
    with c2:
        fig_scatter = px.scatter(
            df_merged, x='CO2_per_capita', y='PM2.5', 
            trendline="ols", trendline_color_override="red",
            title="CO2 배출량과 PM2.5 농도 산점도",
            labels={'CO2_per_capita': '1인당 CO2 배출량 (톤)', 'PM2.5': 'PM2.5 농도 (㎍/㎥)'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


st.markdown("---")
st.subheader("🌏 주요 국가 미세먼지 농도 비교")
if not selected_countries_names:
    st.warning("사이드바에서 비교할 국가를 1개 이상 선택해주세요.")
else:
    comparison_dfs = []
    for name in selected_countries_names:
        code = country_name_to_code[name]
        country_df = fetch_worldbank_data(code, pm25_indicator, start_year, end_year)
        if not country_df.empty:
            country_df['CountryName'] = name
            comparison_dfs.append(country_df)

    if comparison_dfs:
        df_comp = pd.concat(comparison_dfs)
        fig_comp = px.line(df_comp, x="Year", y="Value", color='CountryName', markers=True,
                           title=f"주요 국가별 PM2.5 농도 비교 ({start_year}-{end_year})",
                           labels={"Year": "연도", "Value": "PM2.5 농도 (㎍/㎥)", "CountryName": "국가"})
        fig_comp.update_layout(template='plotly_white', height=500)
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # 최신년도 데이터 바 차트
        latest_year = df_comp['Year'].max()
        df_latest = df_comp[df_comp['Year'] == latest_year].sort_values("Value", ascending=False)
        fig_bar = px.bar(df_latest, x='CountryName', y='Value', color='CountryName',
                         title=f"{latest_year}년 국가별 PM2.5 농도",
                         labels={"Value": "PM2.5 농도 (㎍/㎥)", "CountryName": "국가"})
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.error("선택된 국가들의 데이터를 불러오는 데 실패했습니다.")


# ============================================================================
# Section 3: 건강 영향과 위험성
# ============================================================================
st.markdown("---")
st.header("🩺 미세먼지가 청소년 건강에 미치는 영향")
st.error("""
### 🔬 작지만 치명적인 위협, 초미세먼지(PM2.5)
초미세먼지는 머리카락 굵기의 1/30 정도로 매우 작아 코나 기관지에서 걸러지지 않고 폐 깊숙이 침투하여 혈관으로 들어갈 수 있습니다.
특히 성장기 청소년은 면역 체계가 완성되지 않았고, 신체 활동량이 많아 미세먼지에 더 취약합니다.
""")

st.subheader("신체 부위별 영향 및 위험도")
health_impact_data = {
    '영향 부위': ['호흡기', '심혈관', '뇌/신경계', '피부', '눈'],
    '주요 질환': ['천식, 폐렴, 폐기능 저하', '심근경색, 뇌졸중', '학습능력 저하, 우울감', '아토피, 피부노화', '안구건조증, 결막염'],
    '위험도': ['🔴 매우 높음', '🔴 매우 높음', '🟠 높음', '🟡 보통', '🟡 보통'],
}
health_df = pd.DataFrame(health_impact_data)
st.dataframe(health_df, use_container_width=True)

st.subheader("🏫 학교 생활에 미치는 영향")
st.info("""
- **체육 활동 제한**: 실외 체육 수업이 취소되거나 실내 활동으로 대체되어 신체 발달 기회 감소
- **학업 집중도 저하**: 두통, 피로감, 산소 부족 등으로 인해 수업 집중력 저하
- **교우 관계 영향**: 야외 활동이 줄어들면서 친구들과의 교류 및 사회성 발달 기회 감소
""")

st.markdown("---")
st.subheader("😰 미세먼지와 청소년 정신건강 (환경 스트레스)")
st.warning("""
### '미세먼지 블루(Blue)'란?
미세먼지 문제가 장기화되면서 느끼는 우울감, 불안감, 무력감 등 정신적 스트레스를 의미합니다.
특히 학생들에게 다음과 같은 영향을 줍니다:

- **미래에 대한 불안감**: '어차피 공기도 안 좋은데...'라며 미래에 대한 희망을 잃고 무력감을 느낌
- **활동 제약으로 인한 스트레스**: 마음껏 뛰어놀지 못하는 상황에 대한 답답함과 스트레스
- **사회적 고립감**: 외출을 자제하게 되면서 친구들과 어울릴 기회가 줄어듦

**극복 방법**:
1. 작은 실천의 중요성 인지: 내가 하는 분리수거, 전기 절약이 환경에 도움이 된다는 긍정적 생각 갖기
2. 함께 행동하기: 친구들과 환경 동아리를 만들거나 캠페인에 참여하며 연대감 형성하기
3. 정확한 정보 얻기: 막연한 공포 대신, 신뢰할 수 있는 정보를 바탕으로 합리적으로 대응하기
""")

# ============================================================================
# Section 4: 보고서 및 실천 방안
# ============================================================================
st.markdown("---")
st.header("💡 결론 및 제언")
st.markdown("""
### 🎯 미세먼지 문제, 더 이상 외면할 수 없습니다.
데이터 분석을 통해 확인했듯이 미세먼지는 우리 건강과 학습 환경, 나아가 미래까지 위협하는 심각한 문제입니다.
정부와 사회 전체의 거시적인 노력도 중요하지만, 우리의 일상을 바꾸는 작은 실천들이 모일 때 가장 큰 변화를 만들 수 있습니다.
""")

st.subheader("🏠 제언 1. 일상 생활 속에서 시작하는 작은 변화")
col3, col4 = st.columns(2)
with col3:
    st.success("""
    #### ✅ 개인적 실천 방안 (나를 지키는 습관)
    1. **마스크는 필수**: 외출 시 KF80 이상 보건용 마스크를 올바르게 착용합니다.
    2. **수시로 확인**: '에어코리아' 등 앱을 통해 실시간 농도를 확인하고, '나쁨'일 땐 외출을 최소화합니다.
    3. **청결 유지**: 외출 후에는 반드시 손, 얼굴, 머리카락을 깨끗이 씻습니다.
    4. **환기는 현명하게**: 미세먼지가 좋은 날, 대기 순환이 활발한 시간에 짧게 환기합니다.
    """)
with col4:
    st.warning("""
    #### 🏫 학교 및 학급 단위 실천 (함께 만드는 변화)
    1. **'미세먼지 신호등' 운영**: 교실 입구에 농도별 색깔 깃발을 꽂아 모두의 경각심을 높입니다.
    2. **'공기정화 교실' 만들기**: 공기정화식물을 키우고, 공기청정기 필터를 주기적으로 관리합니다.
    3. **캠페인 활동**: '대중교통 이용의 날', '잔반 없는 날' 등 탄소 배출을 줄이는 캠페인을 주도합니다.
    4. **정책 제안**: '학생 미세먼지 감시단'을 만들어 학교 주변 대기질을 측정하고, 교육청에 안전한 통학로를 건의합니다.
    """)

st.markdown("---")
st.header("📄 미림마이스터고 1학년 4반 연구 보고서 요약")

with st.expander("**서론**: 우리가 이 보고서를 쓰게 된 이유"):
    st.markdown("""
    단순히 불편함을 넘어, 미세먼지가 우리 건강과 생활에 미치는 영향을 데이터로 직접 확인하고,
    청소년의 시각에서 실천 가능한 대응 방안을 제시하고자 이 연구를 시작했습니다.
    """)
    
with st.expander("**본론 1**: 데이터로 보는 미세먼지의 객관적 위협"):
    st.markdown("""
    World Bank 데이터 분석 결과, 한국의 PM2.5 농도는 WHO 권고 기준을 크게 상회하며, OECD 평균보다 높은 수준을 유지하고 있습니다.
    이는 미세먼지가 더 이상 감각적인 위협이 아닌, 데이터로 증명되는 과학적 사실임을 보여줍니다.
    """)

with st.expander("**본론 2**: 학생 건강과 학교 생활에 미치는 실제 영향"):
    st.markdown("""
    미세먼지는 호흡기, 심혈관 등 신체적 건강뿐만 아니라, '미세먼지 블루'라는 신조어가 생길 정도로 정신 건강에도 부정적 영향을 미칩니다.
    또한, 잦은 실외활동 제한은 학생들의 학습권과 사회성 발달을 저해하는 요인으로 작용합니다.
    """)
    
with st.expander("**결론**: 청소년은 피해자를 넘어 변화의 주체로"):
    st.markdown("""
    미세먼지 문제에 있어 청소년은 단순한 피해자가 아닙니다.
    일상 속 작은 실천을 통해 스스로를 보호하고, 학교와 지역사회에 긍정적 변화를 제안하는 능동적인 주체로서 행동해야 합니다.
    우리의 작은 날갯짓이 더 건강한 하늘을 만드는 태풍의 시작이 될 수 있습니다.
    """)
    
st.markdown("---")
with st.expander("📚 참고 자료"):
    st.markdown("""
    - World Bank Open Data (PM2.5, CO2 Emissions)
    - 세계보건기구(WHO) Air Quality Guidelines
    - 질병관리청 청소년 건강행태조사 (2019~2022)
    - 보건복지부 보건의료 빅데이터 개방시스템
    """)

