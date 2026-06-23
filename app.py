import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.font_manager as fm
font_path = './fonts/NanumBarunGothic.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'NanumBarunGothic'
else:
     plt.rcParams['font.family'] = 'Malgun Gothic' if 'Malgun Gothic' in fm.findSystemFonts(fontpaths=None, fontext='ttf') else 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
from scipy import stats
from scipy.fft import rfft, rfftfreq
import os

# --- 전역 변수 및 상수 (Colab 노트북과 동일하게 설정) ---
FS = 10000  # 샘플링 주파수
DATASET_NAME = "Case Western Reserve University Bearing Data Center"
DATASET_URL = "https://www.kaggle.com/datasets/brjapon/cwru-bearing-datasets"

# 한글 폰트 설정 (Streamlit Cloud에서는 폰트 설치가 필요할 수 있음)
# Colab 환경에서 테스트 시에는 별도의 설정 없이 `plt.rcParams['font.family'] = 'NanumBarunGothic'` 사용 가능
# Streamlit Cloud에서 한글 폰트 적용을 위해서는 'fonts' 디렉토리에 폰트 파일을 넣고 `.streamlit/config.toml`에 설정이 필요합니다.
# 예시: font_path = './fonts/NanumBarunGothic.ttf'

# matplotlib 한글 폰트 설정 (Streamlit Cloud 환경에서는 별도 처리 필요)
# 이 코드는 Colab 환경에서 폰트가 잘 설정되어 있다는 가정하에 작성되었습니다.
# Streamlit Cloud 배포 시에는 폰트 파일과 함께 설정 파일(`config.toml`)을 추가해야 합니다.
plt.rcParams['font.family'] = 'Malgun Gothic' if 'Malgun Gothic' in plt.matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf') else 'NanumBarunGothic' # 로컬 테스트용
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지


# --- 분석 함수 (Colab 노트북에서 가져옴) ---
def calculate_features(signal):
    signal = np.asarray(signal).ravel()
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    kurtosis = stats.kurtosis(signal, fisher=False)
    skewness = stats.skew(signal)
    crest_factor = peak / rms if rms > 0 else np.nan
    std = np.std(signal)
    mean_abs = np.mean(np.abs(signal))
    return {
        "mean": np.mean(signal),
        "std": std,
        "rms": rms,
        "peak": peak,
        "kurtosis": kurtosis,
        "skewness": skewness,
        "crest_factor": crest_factor,
        "mean_abs": mean_abs,
    }

def compute_fft(signal, fs):
    signal = np.asarray(signal).ravel()
    signal = signal - np.mean(signal)
    n = len(signal)
    window = np.hanning(n)
    spectrum = np.abs(rfft(signal * window)) / n
    freq = rfftfreq(n, 1 / fs)
    return freq, spectrum

def window_features(signal, fs, window_sec=0.2, step_sec=0.1):
    signal = np.asarray(signal).ravel()
    window = int(fs * window_sec)
    step = int(fs * step_sec)
    rows = []
    for start in range(0, len(signal) - window + 1, step):
        seg = signal[start:start + window]
        rows.append({
            "time_sec": start / fs,
            **calculate_features(seg),
        })
    return pd.DataFrame(rows)


# --- 그래프 함수 (Colab 노트북에서 가져옴) ---
def plot_time_waveform_ax(ax, signal, fs, title, seconds=0.2):
    n = min(len(signal), int(fs * seconds))
    x = np.arange(n) / fs
    ax.plot(x, signal[:n])
    ax.set_title(title)
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)

def plot_fft_ax(ax, signal, fs, title, max_freq=1000):
    freq, spectrum = compute_fft(signal, fs)
    mask = freq <= max_freq
    ax.plot(freq[mask], spectrum[mask])
    ax.set_title(title)
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)


# --- Streamlit 앱 레이아웃 ---
st.set_page_config(layout="wide", page_title="설비 이상 분석 보고서")
st.title("⚙️ 공개 진동 데이터 기반 설비 이상 분석 보고서")
st.markdown("이 애플리케이션은 공개 진동 데이터를 분석하여 설비의 상태를 진단하고 CBM(Condition-Based Maintenance) 관점의 의사결정을 돕기 위해 제작되었습니다.")


# --- 데이터 로드 및 처리 ---
@st.cache_data # 데이터 로딩 및 처리는 한 번만 수행하도록 캐싱
def load_and_process_data():
    # Colab에서 저장한 CSV 파일 로드
    feature_df = pd.read_csv('feature_summary.csv')
    trend_df = pd.read_csv('trend_analysis.csv')

    # MAT_FILES는 app.py가 직접 .mat 파일을 로드하지 않으므로, 
    # 파일 목록 정보만 재구성하여 시각화에 사용합니다.
    # 여기서는 예시로 두 개의 파일을 직접 지정합니다.
    # 실제 환경에서는 Streamlit File Uploader를 통해 .mat 파일을 업로드하도록 구현할 수도 있습니다.
    MAT_FILES_INFO = [
        {"label": "Time_Normal_1_098", "path": "Time_Normal_1_098.mat"},
        {"label": "OR007_6_1_136", "path": "OR007_6_1_136.mat"}
    ]
    
    # 여기서는 시각화를 위해 원래 노트북의 `all_analysis_data`와 유사한 구조를 재구성합니다.
    # 실제 신호 데이터는 포함되어 있지 않으므로, 파형/FFT 플롯은 표시되지 않을 수 있습니다.
    # .mat 파일을 직접 업로드 받아 처리하는 로직을 추가해야 파형/FFT를 그릴 수 있습니다.
    # 여기서는 feature_df와 trend_df만을 기반으로 작동하도록 합니다.

    # (선택 사항: .mat 파일을 Streamlit 앱에 업로드하여 처리하는 로직 추가)
    # uploaded_files = st.file_uploader("MAT 파일 업로드", type="mat", accept_multiple_files=True)
    # if uploaded_files:
    #    # ... 업로드된 .mat 파일을 처리하는 로직 ...
    #    # 이 예시에서는 미리 저장된 CSV만을 사용합니다.

    return feature_df, trend_df, MAT_FILES_INFO

feature_df, trend_df, MAT_FILES_INFO = load_and_process_data()


# --- 사이드바 메뉴 ---
st.sidebar.header("분석 파일 선택")
selected_file_label = st.sidebar.selectbox(
    "분석할 파일 선택:",
    options=[item["label"] for item in MAT_FILES_INFO]
)

current_file_features = feature_df[feature_df["file"] == selected_file_label]
current_file_trend = trend_df[trend_df["file"] == selected_file_label]

# --- 보고서 내용 생성 (Colab 노트북에서 가져옴) ---
def generate_report_content(file_label, current_file_features, current_file_trend):
    normal_rms = current_file_features.loc[current_file_features['state']=='normal', 'rms'].iloc[0]
    fault_rms = current_file_features.loc[current_file_features['state']=='fault', 'rms'].iloc[0]
    normal_kurtosis = current_file_features.loc[current_file_features['state']=='normal', 'kurtosis'].iloc[0]
    fault_kurtosis = current_file_features.loc[current_file_features['state']=='fault', 'kurtosis'].iloc[0]
    normal_crest_factor = current_file_features.loc[current_file_features['state']=='normal', 'crest_factor'].iloc[0]
    fault_crest_factor = current_file_features.loc[current_file_features['state']=='fault', 'crest_crest_factor'].iloc[0]

    # 해당 파일의 정상 신호로 기준값 계산 (diagnosis cell의 로직과 일치)
    normal_win_for_file = current_file_trend[current_file_trend["state"] == "normal"]
    normal_baseline = normal_win_for_file[["rms", "kurtosis", "crest_factor"]].agg(["mean", "std"])
    rms_threshold = normal_baseline.loc["mean", "rms"] + 3 * normal_baseline.loc["std", "rms"]
    kurtosis_threshold = 5.0
    crest_threshold = 4.0

    summary_report = f'''
# {file_label} - 공개 진동 데이터 분석 결과 보고서

## 1. 개요
본 보고서는 {DATASET_NAME} 데이터셋의 {file_label} 파일 관련 데이터를 활용하여 설비의 정상 및 이상 진동 신호를 분석하고, 상태진단 기준을 제안합니다. 시간 영역 파형, 주요 통계 특징값, 주파수 영역 스펙트럼 및 시간 경과에 따른 특징값 추세 분석을 통해 이상 상태의 징후를 식별합니다.

## 2. 사용 데이터
- 데이터셋: {DATASET_NAME}
- 파일명: {file_label}.mat
- 출처: {DATASET_URL}
- 샘플링 주파수: {FS} Hz

## 3. 분석 결과 요약

### 3.1. 시간 영역 파형 비교
*(이 앱에서는 원시 파형 데이터가 직접 로드되지 않으므로, 요약된 내용을 참조해 주세요.)*
- **정상 신호:** 안정적인 주기성을 보이며 비교적 낮은 진폭을 유지합니다.
- **이상 신호:** 전체적인 진폭이 증가하고, 명확한 충격성 피크들이 주기적으로 관찰됩니다. 이는 베어링 손상과 같은 충격성 결함의 전형적인 특징입니다.

### 3.2. 시간 영역 특징값 비교
| 특징값 | 정상 | 이상 | 변화량 | 의미 |
|---|---|---|---|---|
| RMS | {normal_rms:.4f} | {fault_rms:.4f} | {fault_rms - normal_rms:.4f} | 진동 에너지 변화 |
| Kurtosis | {normal_kurtosis:.4f} | {fault_kurtosis:.4f} | {fault_kurtosis - normal_kurtosis:.4f} | 신호의 뾰족함/충격성 변화 |
| Crest Factor | {normal_crest_factor:.4f} | {fault_crest_factor:.4f} | {fault_crest_factor - normal_crest_factor:.4f} | 충격성 이상에 대한 민감도 변화 |

위 표에서 볼 수 있듯이, 주요 특징값들이 정상과 이상 상태에서 변화를 보였습니다. **RMS, Kurtosis, Crest Factor**는 이상 상태를 구별하는 데 효과적임을 알 수 있습니다.

### 3.3. 주파수 영역 분석 (FFT)
*(이 앱에서는 원시 파형 데이터가 직접 로드되지 않으므로, 요약된 내용을 참조해 주세요.)*
- **정상 신호:** 특정 기본 주파수 성분만 뚜렷하게 나타납니다.
- **이상 신호:** 기본 주파수 외에 여러 고주파수 성분들이 높은 진폭으로 나타납니다. 이는 결함으로 인한 충격이 다양한 주파수 대역에서 발생하고 있음을 시사하며, 특정 결함 주파수 성분 분석을 통해 결함 유형을 진단할 수 있는 근거가 됩니다.

### 3.4. 구간별 특징값 추세 분석
- **RMS 추세:** 이상 신호의 RMS는 정상 신호에 비해 변화를 보입니다.
- **Kurtosis 추세:** 이상 신호의 Kurtosis는 정상 신호보다 현저히 높은 값을 유지하며, 이는 지속적인 충격성 결함의 존재를 나타냅니다.
- **Crest Factor 추세:** 이상 신호의 Crest Factor 역시 정상 신호보다 높은 값을 가지며, 신호의 뾰족함이 증가했음을 확인시켜 줍니다.

## 4. 규칙 기반 상태진단 기준 제안 및 CBM 해석

### 4.1. 진단 기준 예시
- RMS 주의 기준: 정상 RMS 평균 + 3σ = {rms_threshold:.4f}
- Kurtosis 주의 기준: {kurtosis_threshold}
- Crest Factor 주의 기준: {crest_threshold}

**진단 로직:**
- 한 가지 지표가 주의 기준을 초과하면 '주의' 상태로 진단합니다.
- 두 가지 이상의 지표가 주의 기준을 초과하면 '위험' 상태로 진단합니다.

### 4.2. CBM (Condition-Based Maintenance) 관점의 의사결정
- **이상 감지:** 현재 분석 결과, 이상 신호는 대부분 '주의' 또는 '위험' 상태로 진단되었습니다.
- **점검/정비 지시:**
    - '주의' 상태가 일정 기간 지속되거나, 증가 추세를 보일 경우 예방적 점검을 지시합니다.
    - '위험' 상태로 진단되거나, 특징값(특히 Kurtosis, Crest Factor)의 급격한 상승이 관찰될 경우 즉각적인 정밀 점검 및 정비를 지시하여 설비의 갑작스러운 고장을 방지하고 생산 손실을 최소화합니다.

### 4.3. 실제 현장 적용 시 한계점 및 고려사항
- **데이터 대표성:** 현재 사용된 데이터는 예시이므로, 실제 설비 데이터의 다양한 노이즈와 복잡성을 반영하지 못할 수 있습니다. 실제 적용을 위해서는 해당 설비의 충분한 정상 및 이상 데이터 수집이 필수적입니다.
- **임계값 설정:** 본 보고서의 임계값은 예시이며, 실제 현장에서는 설비의 종류, 운전 조건(부하, 속도), 센서 위치 등을 고려하여 통계적 방법 또는 전문가의 경험을 바탕으로 정교하게 설정해야 합니다.
- **결함 유형 분류:** RMS, Kurtosis, Crest Factor는 결함 존재 여부를 판단하는 데 유용하지만, 특정 결함 유형(예: 내륜, 외륜, 전동체 결함)을 정확히 분류하기 위해서는 스펙트럼 분석에서 특정 주파수 대역의 변화를 상세히 분석하거나 추가적인 특징값(예: BPFI, BPFO 등)을 활용해야 합니다.
- **환경 요인:** 온도, 습도 등 환경 변화가 진동 신호에 미치는 영향을 고려해야 합니다.

## 5. 결론
본 분석을 통해 진동 데이터의 시간 및 주파수 영역 분석, 그리고 통계적 특징값을 활용하여 설비의 이상 유무를 효과적으로 감지할 수 있음을 확인했습니다. CBM 전략의 성공적인 구현을 위해서는 실제 설비 데이터 기반의 지속적인 모니터링, 정교한 임계값 설정, 그리고 다양한 결함 유형에 대한 심층적인 분석이 요구됩니다.
'''
    return summary_report



# --- 메인 대시보드 ---
st.subheader(f"선택된 파일: {selected_file_label}")

# 특징값 비교 그래프
st.markdown("### 1. 정상/이상 특징값 비교")
plot_cols = ["rms", "peak", "kurtosis", "crest_factor"]
file_specific_df = feature_df[feature_df["file"] == selected_file_label]

if not file_specific_df.empty:
    fig_features = file_specific_df.set_index("state")[plot_cols].T.plot(kind="bar", figsize=(10, 4))
    plt.title(f"{selected_file_label} - 정상/이상 특징값 비교")
    plt.ylabel("Feature value")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)
    st.pyplot(fig_features.figure)
    plt.close(fig_features.figure)
else:
    st.warning(f"{selected_file_label}에 대한 특징값 데이터를 찾을 수 없습니다.")


# 구간별 특징값 추세 그래프
st.markdown("### 2. 구간별 특징값 추세")
if not current_file_trend.empty:
    fig_trend, axes_trend = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for i, col in enumerate(["rms", "kurtosis", "crest_factor"]):
        for state, group in current_file_trend.groupby("state"):
            axes_trend[i].plot(group["time_sec"], group[col], label=state)
        axes_trend[i].set_title(f"{selected_file_label} - 구간별 {col} 추세")
        axes_trend[i].set_ylabel(col)
        axes_trend[i].grid(alpha=0.3)
        if i == 0:
            axes_trend[i].legend()
    axes_trend[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    st.pyplot(fig_trend)
    plt.close(fig_trend)
else:
    st.warning(f"{selected_file_label}에 대한 구간별 추세 데이터를 찾을 수 없습니다.")


# 분석 결과 보고서
st.markdown("### 3. 분석 결과 보고서")
report_content = generate_report_content(selected_file_label, current_file_features, current_file_trend)
st.markdown(report_content)

# 사이드바에 파일 다운로드 링크 추가
st.sidebar.markdown("--- ")
st.sidebar.download_button(
    label="특징값 CSV 다운로드",
    data=feature_df.to_csv(index=False).encode('utf-8'),
    file_name="feature_summary_all.csv",
    mime="text/csv",
)
st.sidebar.download_button(
    label="추세 분석 CSV 다운로드",
    data=trend_df.to_csv(index=False).encode('utf-8'),
    file_name="trend_analysis_all.csv",
    mime="text/csv",
)
