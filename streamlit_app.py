import streamlit as st
import json
import os
from datetime import datetime

# JSON 파일 경로 설정
JSON_FILE = 'opinions.json'

# 1. 기존 데이터를 불러오는 함수
def load_data():
    """JSON 파일이 존재하면 데이터를 불러오고, 없으면 빈 리스트를 반환합니다."""
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                return data
            except json.JSONDecodeError:
                # 파일이 비어있거나 잘못된 형식일 경우
                return []
    return []

# 2. 새로운 의견을 저장하는 함수
def save_data(author, title, opinion):
    """새로운 의견을 기존 데이터에 추가하고 JSON 파일에 저장합니다."""
    # 현재 시간 기록
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    new_entry = {
        'author': author if author else '익명', # 작성자가 없으면 '익명'으로 처리
        'title': title,
        'opinion': opinion,
        'timestamp': timestamp
    }
    
    # 기존 데이터 불러오기
    all_opinions = load_data()
    # 새로운 의견 추가
    all_opinions.append(new_entry)
    
    # JSON 파일에 저장 (한글 깨짐 방지: ensure_ascii=False)
    with open(JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_opinions, f, ensure_ascii=False, indent=4)

# --- Streamlit 웹 앱 UI 구성 ---

# 3. 웹 페이지 제목 설정
st.title("📢 2026년도 학생회에게 바라는 점")
st.write("2026년을 이끌어갈 학생회를 위해 여러분의 소중한 의견을 남겨주세요!")
st.markdown("---")


# 4. 의견 입력을 위한 폼(Form) 생성
# st.form을 사용하면 '제출하기' 버튼을 누를 때만 전체 입력값이 한 번에 처리됩니다.
with st.form("opinion_form", clear_on_submit=True):
    author_input = st.text_input("닉네임 (선택 사항, 비워두시면 '익명'으로 제출됩니다)")
    title_input = st.text_input("제안 제목")
    opinion_input = st.text_area("내용을 입력해주세요", height=200)
    
    # 제출 버튼
    submitted = st.form_submit_button("✅ 의견 제출하기")

    # 제출 버튼을 눌렀을 때 실행될 로직
    if submitted:
        if not title_input or not opinion_input:
            st.warning("⚠️ 제목과 내용을 모두 입력해주세요!")
        else:
            save_data(author_input, title_input, opinion_input)
            st.success("🗳️ 소중한 의견이 성공적으로 제출되었습니다. 감사합니다!")

# 5. 제출된 의견 목록 보여주기
st.markdown("---")
st.header("📝 지금까지 제출된 의견들")

all_opinions = load_data()

if not all_opinions:
    st.info("아직 제출된 의견이 없습니다. 첫 번째 의견을 남겨주세요!")
else:
    # 최신순으로 정렬하여 보여주기
    for i, opinion in enumerate(reversed(all_opinions)):
        with st.expander(f"**{opinion.get('title', '제목 없음')}** - `{opinion.get('author', '익명')}` ({opinion.get('timestamp', '')})"):
            st.write(opinion.get('opinion', ''))