"""
Patent Analysis RAG System
특허 분석을 위한 Advanced Agentic RAG 시스템

This Streamlit application provides an intelligent patent analysis system
that uses RAG (Retrieval-Augmented Generation) technology to analyze patents
and answer questions about them.
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path
import re
from datetime import datetime

# Vector Store related imports (only when needed)
try:
    from src.vector_store.patent_vector_store import PatentVectorStore
    from src.vector_store.embedding_manager import EmbeddingManager
    from src.patent_search.patent_parser import PatentDataProcessor
except ImportError as e:
    print(f"Vector Store 관련 모듈 import 실패: {e}")
    PatentVectorStore = None
    EmbeddingManager = None
    PatentDataProcessor = None

# Set page configuration
st.set_page_config(
    page_title="특허 분석 RAG 시스템",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """Main application function"""

    # Title and description
    st.title("📄 특허 분석 RAG 시스템")
    st.markdown(
        """
    **Advanced Agentic RAG 기반 특허 분석 도구**
    
    이 시스템은 엑셀 파일에 포함된 특허 번호를 기반으로 특허 정보를 자동 수집하고, 
    AI를 활용하여 특허에 대한 질문에 답변해드립니다.
    """
    )

    # Sidebar for navigation
    with st.sidebar:
        st.header("🧭 메뉴")

        # Check if environment is properly set up
        env_status = check_environment()

        if env_status:
            st.success("✅ 환경 설정 완료")

            page = st.selectbox(
                "페이지 선택",
                [
                    "📤 파일 업로드",
                    "🔍 특허 검색",
                    "🧪 Vector Store 테스트",
                    "❓ 질문 관리",
                    "🚀 고도화된 CRAG",
                    "📊 결과 보기",
                    "⚙️ 설정",
                ],
            )
        else:
            st.error("❌ 환경 설정 필요")
            st.info("API 키를 .env 파일에 설정해주세요.")
            page = "⚙️ 설정"

    # Main content area
    if page == "📤 파일 업로드":
        show_file_upload_page()
    elif page == "🔍 특허 검색":
        show_patent_search_page()
    elif page == "🧪 Vector Store 테스트":
        show_vector_store_test_page()
    elif page == "❓ 질문 관리":
        show_question_management_page()
    elif page == "🚀 고도화된 CRAG":
        show_langgraph_crag_page()
    elif page == "📊 결과 보기":
        show_results_page()
    elif page == "⚙️ 설정":
        show_settings_page()


def check_environment():
    """Check if the environment is properly configured"""
    required_env_vars = ["SERPAPI_API_KEY", "OPENAI_API_KEY"]

    # Load environment variables
    from dotenv import load_dotenv

    load_dotenv()

    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var) or os.getenv(var) == f"your_{var.lower()}_here":
            missing_vars.append(var)

    return len(missing_vars) == 0


def detect_patent_column(df):
    """
    특허 번호가 포함된 컬럼을 자동으로 감지하는 함수
    """
    # 특허 번호 관련 키워드들
    patent_keywords = [
        "patent",
        "patents",
        "특허",
        "특허번호",
        "특허_번호",
        "patent_number",
        "patent_no",
        "patentno",
        "number",
        "no",
        "번호",
        "id",
        "patent_id",
    ]

    # 컬럼명으로 검색
    for col in df.columns:
        col_lower = str(col).lower().strip()
        for keyword in patent_keywords:
            if keyword.lower() in col_lower:
                return col

    # 컬럼명으로 찾지 못한 경우, 데이터 패턴으로 검색
    for col in df.columns:
        # 샘플 데이터에서 특허 번호 패턴을 찾아봄
        sample_data = df[col].dropna().astype(str).head(10)
        patent_pattern_count = 0

        for value in sample_data:
            if is_valid_patent_number(value):
                patent_pattern_count += 1

        # 샘플의 70% 이상이 특허 번호 패턴이면 해당 컬럼으로 판단
        if patent_pattern_count >= len(sample_data) * 0.7:
            return col

    return None


def is_valid_patent_number(patent_str):
    """
    SerpAPI Google Patents API 형식에 맞는 특허 번호인지 검증하는 함수
    형식: <country_code><patent_number><classification> (예: US11734097B1, KR10-2021-0123456, EP3456789A1)
    """
    if not isinstance(patent_str, str):
        patent_str = str(patent_str)

    patent_str = patent_str.strip().upper()

    # 빈 문자열이나 'INVALID_PATENT' 같은 명시적으로 잘못된 값 체크
    if not patent_str or patent_str.lower() in ["invalid_patent", "n/a", "na", "-", ""]:
        return False

    # SerpAPI Google Patents API에서 지원하는 특허 번호 패턴
    # 패턴: <국가코드><특허번호><분류기호(선택적)>
    patterns = [
        # US 특허: US + 숫자 + 문자+숫자 조합 (예: US11734097B1, US10123456A1)
        r"^US\d{7,10}[A-Z]?\d*$",
        # KR 특허: KR + 숫자 또는 KR + 숫자-숫자-숫자 (예: KR1234567890, KR10-2021-0123456)
        r"^KR\d{7,12}$",
        r"^KR\d{2}-\d{4}-\d{7}$",
        # EP 특허: EP + 숫자 + 문자+숫자 조합 (예: EP3456789A1, EP1234567B1)
        r"^EP\d{6,10}[A-Z]?\d*$",
        # JP 특허: JP + 숫자 + 문자+숫자 조합 (예: JP2023123456A, JP6789012B2)
        r"^JP\d{6,12}[A-Z]?\d*$",
        # CN 특허: CN + 숫자 + 문자+숫자 조합 (예: CN202310987654A, CN112345678B)
        r"^CN\d{8,15}[A-Z]?\d*$",
        # 기타 국가 코드 패턴 (2글자 국가코드 + 숫자 + 선택적 문자)
        r"^[A-Z]{2}\d{6,12}[A-Z]?\d*$",
    ]

    for pattern in patterns:
        if re.match(pattern, patent_str):
            return True

    return False


def validate_patent_data(df, patent_column):
    """
    특허 데이터의 유효성을 검증하는 함수
    """
    if patent_column not in df.columns:
        return False, f"선택된 컬럼 '{patent_column}'이 데이터에 존재하지 않습니다."

    patent_data = df[patent_column].dropna()

    if len(patent_data) == 0:
        return False, f"컬럼 '{patent_column}'에 유효한 데이터가 없습니다."

    # 특허 번호 형식 검증
    valid_patents = 0
    total_patents = len(patent_data)

    for patent in patent_data:
        if is_valid_patent_number(str(patent)):
            valid_patents += 1

    validity_ratio = valid_patents / total_patents

    if validity_ratio < 0.5:  # 50% 미만이 유효한 특허 번호
        return (
            False,
            f"컬럼 '{patent_column}'에서 유효한 특허 번호 형식이 {validity_ratio:.1%}만 발견되었습니다. 올바른 컬럼을 선택했는지 확인해주세요.",
        )

    return (
        True,
        f"유효성 검증 완료: {valid_patents}/{total_patents} ({validity_ratio:.1%}) 개의 유효한 특허 번호 발견",
    )


def load_file_safely(uploaded_file):
    """
    파일을 안전하게 로드하는 함수
    """
    try:
        file_extension = uploaded_file.name.split(".")[-1].lower()

        if file_extension == "csv":
            # CSV 파일 인코딩 자동 감지
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    uploaded_file.seek(0)  # 파일 포인터 리셋
                    df = pd.read_csv(uploaded_file, encoding="cp949")
                except:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding="latin-1")

        elif file_extension in ["xlsx", "xls"]:
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {file_extension}")

        # 기본적인 데이터 검증
        if df.empty:
            raise ValueError("업로드된 파일이 비어있습니다.")

        if len(df.columns) == 0:
            raise ValueError("파일에 컬럼이 없습니다.")

        return df, None

    except Exception as e:
        return None, str(e)


def show_file_upload_page():
    """File upload page with enhanced functionality"""
    st.header("📤 파일 업로드")

    st.markdown(
        """
    특허 번호가 포함된 파일을 업로드하세요.
    
    **지원 형식:**
    - 📊 Excel 파일: `.xlsx`, `.xls`
    - 📄 CSV 파일: `.csv`
    
    **요구사항:**
    - 특허 번호가 포함된 열이 있어야 합니다
    - 지원되는 특허 번호 형식: US1234567, KR1234567, EP1234567 등
    """
    )

    uploaded_file = st.file_uploader(
        "파일 선택",
        type=["xlsx", "xls", "csv"],
        help="특허 번호가 포함된 Excel 또는 CSV 파일을 업로드하세요.",
    )

    if uploaded_file is not None:
        with st.spinner("파일을 처리하는 중..."):
            # 파일 로드
            df, error = load_file_safely(uploaded_file)

            if error:
                st.error(f"❌ 파일 로드 오류: {error}")
                st.markdown(
                    """
                **해결 방법:**
                - 파일이 손상되지 않았는지 확인하세요
                - Excel 파일의 경우 다른 프로그램에서 열려있지 않은지 확인하세요
                - CSV 파일의 경우 UTF-8 인코딩으로 저장되었는지 확인하세요
                """
                )
                return

        st.success(f"✅ 파일 업로드 완료: {uploaded_file.name}")

        # 파일 정보 표시
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 행 수", f"{df.shape[0]:,}")
        with col2:
            st.metric("총 열 수", f"{df.shape[1]:,}")
        with col3:
            file_size = uploaded_file.size / 1024  # KB
            if file_size > 1024:
                st.metric("파일 크기", f"{file_size/1024:.1f} MB")
            else:
                st.metric("파일 크기", f"{file_size:.1f} KB")

        # 데이터 미리보기 (PyArrow 에러 방지를 위한 데이터 타입 정리)
        st.subheader("📋 데이터 미리보기")
        preview_df = df.head().copy()
        # 모든 컬럼을 문자열로 변환하여 PyArrow 호환성 확보
        for col in preview_df.columns:
            preview_df[col] = preview_df[col].astype(str)
        st.dataframe(preview_df, use_container_width=True)

        # 자동 컬럼 감지
        st.subheader("🎯 특허 번호 컬럼 선택")

        detected_column = detect_patent_column(df)

        if detected_column:
            st.success(f"🔍 자동 감지된 특허 번호 컬럼: **{detected_column}**")
            default_index = list(df.columns).index(detected_column)
        else:
            st.info(
                "🔍 특허 번호 컬럼을 자동으로 감지하지 못했습니다. 수동으로 선택해주세요."
            )
            default_index = 0

        patent_column = st.selectbox(
            "특허 번호가 포함된 컬럼을 선택하세요:",
            df.columns.tolist(),
            index=default_index,
            help="특허 번호가 포함된 컬럼을 선택하세요. 자동 감지가 틀렸다면 올바른 컬럼을 선택할 수 있습니다.",
        )

        if patent_column:
            # 선택된 컬럼의 데이터 검증
            is_valid, validation_message = validate_patent_data(df, patent_column)

            if is_valid:
                st.success(f"✅ {validation_message}")
            else:
                st.error(f"❌ {validation_message}")
                st.markdown(
                    """
                **해결 방법:**
                - 다른 컬럼을 선택해보세요
                - 특허 번호가 올바른 형식인지 확인하세요 (예: US1234567, KR1234567)
                - 특허 번호에 불필요한 공백이나 특수문자가 없는지 확인하세요
                """
                )

            # 샘플 데이터 표시
            st.write(f"**선택된 컬럼:** {patent_column}")

            sample_data = df[patent_column].dropna().head(10)

            col1, col2 = st.columns(2)
            with col1:
                st.write("**샘플 데이터:**")
                for i, value in enumerate(sample_data, 1):
                    validity_icon = "✅" if is_valid_patent_number(str(value)) else "❌"
                    st.write(f"{i}. {validity_icon} {value}")

            with col2:
                # 통계 정보
                total_count = len(df[patent_column].dropna())
                valid_count = sum(
                    1
                    for x in df[patent_column].dropna()
                    if is_valid_patent_number(str(x))
                )

                st.write("**데이터 통계:**")
                st.write(f"- 총 특허 수: {total_count:,}")
                st.write(f"- 유효한 특허: {valid_count:,}")
                st.write(f"- 유효성 비율: {valid_count/total_count*100:.1f}%")

            # 데이터 저장 버튼
            if st.button("💾 데이터 저장", type="primary", disabled=not is_valid):
                if is_valid:
                    try:
                        # 데이터 저장
                        data_dir = Path("data")
                        data_dir.mkdir(exist_ok=True)

                        # 원본 파일명 기반으로 저장
                        original_name = uploaded_file.name.split(".")[0]
                        file_path = data_dir / f"uploaded_patents_{original_name}.xlsx"
                        df.to_excel(file_path, index=False)

                        # 세션 상태에 저장
                        st.session_state["patent_data"] = df
                        st.session_state["patent_column"] = patent_column
                        st.session_state["data_file_path"] = str(file_path)
                        st.session_state["upload_stats"] = {
                            "total_patents": total_count,
                            "valid_patents": valid_count,
                            "file_name": uploaded_file.name,
                        }

                        st.success("✅ 데이터가 성공적으로 저장되었습니다!")
                        st.balloons()  # 축하 애니메이션

                        st.info(
                            "🚀 이제 '🔍 특허 검색' 페이지로 이동하여 특허 정보를 수집하세요."
                        )

                        # 자동으로 다음 페이지로 이동하는 옵션
                        if st.button("🔍 특허 검색 페이지로 이동"):
                            st.session_state.selected_page = "🔍 특허 검색"
                            st.rerun()

                    except Exception as e:
                        st.error(f"❌ 데이터 저장 중 오류 발생: {str(e)}")
                else:
                    st.error("❌ 유효하지 않은 데이터는 저장할 수 없습니다.")


def show_patent_search_page():
    """Patent search page with SerpAPI integration"""
    st.header("🔍 특허 검색")

    if "patent_data" not in st.session_state:
        st.warning("⚠️ 먼저 파일을 업로드해주세요.")
        st.info("'📤 파일 업로드' 페이지에서 특허 데이터를 업로드하세요.")
        return

    # SerpAPI 클라이언트와 웹 스크래핑 모듈 import
    try:
        from src.patent_search.serp_client import SerpPatentClient, PatentSearchError
        from src.patent_search.description_scraper import (
            GooglePatentsDescriptionScraper,
            ScrapingError,
        )
    except ImportError:
        st.error(
            "❌ 특허 검색 모듈을 불러올 수 없습니다. 시스템 관리자에게 문의하세요."
        )
        return

    st.success("✅ 업로드된 데이터가 있습니다.")

    df = st.session_state["patent_data"]
    patent_column = st.session_state["patent_column"]

    # 업로드 통계가 있으면 표시
    if "upload_stats" in st.session_state:
        stats = st.session_state["upload_stats"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("총 특허 수", f"{stats['total_patents']:,}")
        with col2:
            st.metric("유효한 특허", f"{stats['valid_patents']:,}")
        with col3:
            st.metric(
                "유효성 비율",
                f"{stats['valid_patents']/stats['total_patents']*100:.1f}%",
            )
        with col4:
            st.metric("원본 파일", stats["file_name"])

    st.write(f"**특허 번호 컬럼:** {patent_column}")

    # API 키 확인
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    if not serpapi_key or serpapi_key.startswith("your_"):
        st.error("❌ SERPAPI_API_KEY가 올바르게 설정되지 않았습니다.")
        st.info("'⚙️ 설정' 페이지에서 API 키를 확인하고 설정하세요.")
        return

    # 검색 옵션
    st.subheader("🔧 검색 설정")

    col1, col2, col3 = st.columns(3)
    with col1:
        search_mode = st.selectbox(
            "검색 모드 선택:",
            ["테스트 검색 (1개)", "부분 검색 (최대 5개)", "전체 검색"],
            help="테스트 검색으로 먼저 API 연결을 확인하세요.",
        )

    with col2:
        country_code = st.selectbox(
            "언어 코드:", ["en", "ko", "ja", "zh"], help="검색 결과 언어를 선택하세요."
        )

    with col3:
        enable_web_scraping = st.checkbox(
            "🕷️ 웹 스크래핑 활성화",
            value=False,
            help="Google Patents에서 상세한 특허 설명을 추가로 스크래핑합니다. (처리 시간이 길어집니다)",
        )

    # 샘플 특허 번호 표시
    st.subheader("📋 검색 대상 특허 번호")
    valid_patents = []
    invalid_patents = []

    all_patents = df[patent_column].dropna().tolist()
    for patent in all_patents:
        if is_valid_patent_number(str(patent)):
            valid_patents.append(str(patent))
        else:
            invalid_patents.append(str(patent))

    # 검색할 특허 수 결정
    if search_mode == "테스트 검색 (1개)":
        search_patents = valid_patents[:1]
    elif search_mode == "부분 검색 (최대 5개)":
        search_patents = valid_patents[:5]
    else:
        search_patents = valid_patents

    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(f"**검색할 특허 ({len(search_patents)}개):**")
        for i, patent in enumerate(search_patents[:10], 1):  # 최대 10개만 표시
            st.write(f"{i}. ✅ {patent}")

        if len(search_patents) > 10:
            st.write(f"... 외 {len(search_patents) - 10}개")

    with col2:
        st.write("**통계:**")
        st.write(f"- 총 특허: {len(all_patents):,}")
        st.write(f"- 유효한 특허: {len(valid_patents):,}")
        st.write(f"- 무효한 특허: {len(invalid_patents):,}")
        st.write(f"- 검색 예정: {len(search_patents):,}")

    # 검색 실행 버튼
    if st.button(
        "🚀 특허 검색 시작", type="primary", disabled=len(search_patents) == 0
    ):
        if len(search_patents) == 0:
            st.error("❌ 검색할 유효한 특허가 없습니다.")
            return

        # 검색 실행
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()

        def progress_callback(current, total, patent, success=True, error=None):
            progress = current / total
            progress_bar.progress(progress)

            if success:
                status_text.text(f"진행 중... {current}/{total} - {patent} ✅")
            else:
                status_text.text(
                    f"진행 중... {current}/{total} - {patent} ❌ ({error})"
                )

        try:
            with st.spinner("특허 정보를 검색하는 중..."):
                client = SerpPatentClient(serpapi_key)

                # 배치 검색 실행
                search_results = client.batch_search_patents(
                    search_patents,
                    country_code=country_code,
                    progress_callback=progress_callback,
                )

                # 웹 스크래핑 추가 처리 (선택사항)
                if enable_web_scraping and search_results["summary"]["success"] > 0:
                    status_text.text("🕷️ 웹 스크래핑으로 상세 설명 수집 중...")

                    # Description Link 기반 웹 스크래핑 URL 수집 (요구사항에 따라 description_link 우선 사용)
                    patent_urls = []
                    url_to_patent_mapping = {}  # URL과 특허 ID 매핑을 위한 딕셔너리

                    for patent_id, result in search_results["results"].items():
                        # 요구사항: description_link를 우선적으로 사용
                        if "description_link" in result and result["description_link"]:
                            url = result["description_link"]
                            patent_urls.append(url)
                            url_to_patent_mapping[url] = patent_id
                        elif (
                            "google_patents_url" in result
                            and result["google_patents_url"]
                        ):
                            # fallback: google_patents_url 사용
                            url = result["google_patents_url"]
                            patent_urls.append(url)
                            url_to_patent_mapping[url] = patent_id
                        else:
                            # 마지막 fallback: patent_id로부터 Google Patents URL 생성
                            base_url = "https://patents.google.com/patent/"
                            url = f"{base_url}{patent_id}/en"
                            patent_urls.append(url)
                            url_to_patent_mapping[url] = patent_id

                    if patent_urls:
                        # 웹 스크래핑 실행
                        scraper = GooglePatentsDescriptionScraper(
                            requests_per_second=1.0,  # 안전한 속도
                            timeout=30,
                            max_retries=2,
                        )

                        def scraping_progress_callback(completed, total, result):
                            progress = (len(search_patents) + completed) / (
                                len(search_patents) + total
                            )
                            progress_bar.progress(progress)

                            status = "✅ 성공" if result.success else "❌ 실패"
                            status_text.text(
                                f"🕷️ 웹 스크래핑 {completed}/{total} - {result.patent_id}: {status}"
                            )

                        scraping_results = scraper.batch_scrape_patents(
                            patent_urls,
                            max_workers=2,
                            progress_callback=scraping_progress_callback,
                        )

                        # 스크래핑 결과를 검색 결과에 통합 (URL 기반 매칭)
                        scraping_success = 0
                        for scraping_result in scraping_results:
                            if scraping_result.success:
                                # URL 기반으로 특허 ID 찾기
                                matched_patent_id = url_to_patent_mapping.get(
                                    scraping_result.url
                                )

                                if (
                                    matched_patent_id
                                    and matched_patent_id in search_results["results"]
                                ):
                                    search_result = search_results["results"][
                                        matched_patent_id
                                    ]
                                    search_result["scraped_title"] = (
                                        scraping_result.title
                                    )
                                    search_result["scraped_abstract"] = (
                                        scraping_result.abstract
                                    )
                                    search_result["scraped_description"] = (
                                        scraping_result.description
                                    )
                                    search_result["scraped_claims"] = (
                                        scraping_result.claims
                                    )
                                    search_result["scraping_success"] = True
                                    scraping_success += 1

                        # 스크래핑 통계 업데이트
                        search_results["scraping_summary"] = {
                            "total_urls": len(patent_urls),
                            "successful_scraping": scraping_success,
                            "scraping_rate": (
                                scraping_success / len(patent_urls) * 100
                                if patent_urls
                                else 0
                            ),
                        }

                        status_text.text(
                            f"✅ 웹 스크래핑 완료: {scraping_success}/{len(patent_urls)} 성공"
                        )

                # 결과 저장
                st.session_state["search_results"] = search_results
                st.session_state["last_search_timestamp"] = datetime.now().isoformat()
                st.session_state["web_scraping_enabled"] = enable_web_scraping

                # 결과 표시
                progress_bar.empty()
                status_text.empty()

                # 검색 요약
                summary = search_results["summary"]

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("총 검색", f"{summary['total']:,}")
                with col2:
                    st.metric("성공", f"{summary['success']:,}")
                with col3:
                    st.metric("실패", f"{summary['failed']:,}")
                with col4:
                    st.metric("성공률", f"{summary['success_rate']:.1f}%")

                # 웹 스크래핑 통계 표시 (있는 경우)
                if "scraping_summary" in search_results:
                    st.subheader("🕷️ 웹 스크래핑 결과")
                    scraping_summary = search_results["scraping_summary"]

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "스크래핑 대상", f"{scraping_summary['total_urls']:,}"
                        )
                    with col2:
                        st.metric(
                            "스크래핑 성공",
                            f"{scraping_summary['successful_scraping']:,}",
                        )
                    with col3:
                        st.metric(
                            "스크래핑 성공률",
                            f"{scraping_summary['scraping_rate']:.1f}%",
                        )

                if summary["success"] > 0:
                    st.success(f"✅ {summary['success']}개 특허 검색 완료!")
                    st.info("🔍 '📊 결과 보기' 페이지에서 상세 결과를 확인하세요.")

                    # 첫 번째 결과 미리보기
                    if search_results["results"]:
                        st.subheader("📋 검색 결과 미리보기")
                        first_patent = list(search_results["results"].keys())[0]
                        first_result = search_results["results"][first_patent]

                        with st.expander(
                            f"🔍 {first_patent} - {first_result['title'][:50]}..."
                        ):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**기본 정보:**")
                                st.write(f"- 제목: {first_result['title']}")
                                st.write(
                                    f"- 발명자: {', '.join(first_result['inventor'][:3])}"
                                )
                                st.write(
                                    f"- 출원일: {first_result['application_date']}"
                                )
                                st.write(
                                    f"- 공개일: {first_result['publication_date']}"
                                )

                            with col2:
                                st.write("**요약:**")
                                # 웹 스크래핑된 초록이 있으면 사용, 없으면 기본 초록 사용
                                if first_result.get(
                                    "scraping_success"
                                ) and first_result.get("scraped_abstract"):
                                    abstract = first_result["scraped_abstract"][:300]
                                    st.write("🕷️ *웹 스크래핑된 상세 초록:*")
                                else:
                                    abstract = first_result["abstract"][:300]

                                if len(abstract) > 300:
                                    abstract += "..."
                                st.write(abstract)

                        # 웹 스크래핑 성공 시 상세 정보 표시 (전체 너비 사용)
                        if first_result.get("scraping_success"):
                            st.write("---")
                            st.write("### 🕷️ 웹 스크래핑된 상세 정보")

                            # 스크래핑 소스 표시
                            if (
                                "description_link" in first_result
                                and first_result["description_link"]
                            ):
                                st.info(
                                    "📡 **스크래핑 소스**: SerpAPI Description Link (요구사항 준수)"
                                )
                            else:
                                st.info("📡 **스크래핑 소스**: Google Patents 웹페이지")

                            # 상세 설명 미리보기
                            if first_result.get("scraped_description"):
                                desc = first_result["scraped_description"]
                                desc_length = len(desc)
                                st.write(f"**📋 상세 설명** ({desc_length:,} 문자):")

                                # 처음 500자 표시
                                preview_desc = (
                                    desc[:500] + "..." if len(desc) > 500 else desc
                                )
                                with st.expander("상세 설명 미리보기", expanded=True):
                                    st.text_area(
                                        "", preview_desc, height=150, disabled=True
                                    )
                            else:
                                st.warning("📋 상세 설명을 스크래핑하지 못했습니다.")

                            # 청구항 미리보기 (SerpAPI 검색 결과에서 가져옴)
                            claims = first_result.get("claims", [])
                            if claims and len(claims) > 0:
                                claims_count = len(claims)
                                st.write(f"**⚖️ 청구항** ({claims_count}개):")
                                st.info("💡 청구항은 SerpAPI 검색 결과에서 가져옵니다.")

                                with st.expander("청구항 미리보기", expanded=True):
                                    # 첫 3개 청구항 표시
                                    for i, claim in enumerate(claims[:3]):
                                        st.write(f"**청구항 {i+1}:**")
                                        claim_preview = (
                                            claim[:300] + "..."
                                            if len(claim) > 300
                                            else claim
                                        )
                                        st.write(claim_preview)
                                        if i < min(2, claims_count - 1):
                                            st.write("---")

                                    if claims_count > 3:
                                        st.info(
                                            f"+ {claims_count - 3}개 청구항 더 있음"
                                        )
                            else:
                                st.warning("❌ 청구항 정보를 찾을 수 없습니다.")

                if summary["failed"] > 0:
                    st.warning(f"⚠️ {summary['failed']}개 특허 검색 실패")

                    with st.expander("실패한 특허 목록 보기"):
                        for patent, error in search_results["errors"].items():
                            st.write(f"❌ {patent}: {error}")

        except PatentSearchError as e:
            st.error(f"❌ 특허 검색 오류: {str(e)}")
        except Exception as e:
            st.error(f"❌ 예상치 못한 오류: {str(e)}")

    # 이전 검색 결과가 있는 경우 표시
    if "search_results" in st.session_state:
        st.subheader("📊 최근 검색 결과")
        last_results = st.session_state["search_results"]
        last_timestamp = st.session_state.get("last_search_timestamp", "알 수 없음")

        st.info(f"마지막 검색: {last_timestamp}")

        summary = last_results["summary"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("검색한 특허", f"{summary['total']:,}")
        with col2:
            st.metric("성공한 특허", f"{summary['success']:,}")
        with col3:
            st.metric("성공률", f"{summary['success_rate']:.1f}%")

        if st.button("📊 상세 결과 보기"):
            st.session_state.selected_page = "📊 결과 보기"
            st.rerun()


def show_vector_store_test_page():
    """Vector Store 테스트 페이지"""
    st.header("🧪 Vector Store 테스트")
    st.markdown("Task 5-6에서 구현된 Vector Store 기능을 테스트해보세요!")
    st.markdown(
        "**기존 워크플로우**: 📤 파일 업로드 → 🔍 특허 검색 → 📄 웹 스크래핑 → ⚙️ 청킹 → 💾 Vector Store 저장"
    )

    # Import Vector Store modules
    try:
        from src.patent_search.patent_parser import PatentParser, PatentDataProcessor
        from src.vector_store.patent_vector_store import PatentVectorStore
        from src.vector_store.embedding_manager import EmbeddingManager
        import time
    except ImportError as e:
        st.error(f"❌ Vector Store 모듈을 불러올 수 없습니다: {e}")
        st.info("Vector Store 관련 모듈이 올바르게 설치되었는지 확인해주세요.")
        return

    # Initialize session state for vector store
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "processed_patents" not in st.session_state:
        st.session_state.processed_patents = []

    # Load Vector Store function (without cache for debugging)
    def load_vector_store():
        """Vector Store를 초기화합니다."""
        try:
            # 필수 클래스들이 import되었는지 확인
            if PatentVectorStore is None:
                st.error("PatentVectorStore 클래스를 import할 수 없습니다.")
                return None

            # Vector Store 초기화
            vector_store = PatentVectorStore(
                collection_name="patent_chunks",
                persist_directory="./chroma_db",
                reset_collection=False,
                enable_preprocessing=True,
                enable_performance_monitoring=True,
            )
            return vector_store
        except Exception as e:
            st.error(f"Vector Store 초기화 실패: {e}")
            import traceback

            st.text(traceback.format_exc())
            return None

    # 자동 Vector Store 초기화 시도 (함수 정의 후)
    if st.session_state.vector_store is None:
        try:
            st.session_state.vector_store = load_vector_store()
        except Exception as e:
            print(f"자동 Vector Store 초기화 실패: {e}")
            # 오류 시 None으로 유지

    # 사이드바 설정
    with st.sidebar:
        st.subheader("⚙️ Vector Store 설정")

        # Vector Store 상태 표시
        if st.session_state.vector_store:
            st.success("✅ Vector Store 준비됨")

            # Vector Store 통계
            try:
                stats = st.session_state.vector_store.get_stats()
                st.metric("저장된 청크 수", stats.get("total_chunks", 0))
                st.metric("고유 특허 수", stats.get("unique_patents", 0))
            except:
                pass
        else:
            st.warning("❌ Vector Store 미초기화")

        # Vector Store 초기화/재초기화
        button_text = (
            "🔄 Vector Store 재초기화"
            if st.session_state.vector_store
            else "🚀 Vector Store 초기화"
        )
        if st.button(button_text):
            with st.spinner("Vector Store 초기화 중..."):
                st.session_state.vector_store = load_vector_store()
                if st.session_state.vector_store:
                    st.success("Vector Store 초기화 완료!")
                    st.rerun()  # 페이지 새로고침
                else:
                    st.error("Vector Store 초기화 실패!")

    # 탭 생성
    tab1, tab2, tab3 = st.tabs(
        ["📄 기존 데이터 저장", "🎯 유사도 검색", "📊 Vector Store 관리"]
    )

    with tab1:
        st.subheader("기존 특허 검색 결과를 Vector Store에 저장")
        st.info("🔍 특허 검색 페이지에서 처리된 결과를 Vector Store에 저장합니다.")

        # 검색 결과 확인
        if "search_results" in st.session_state:
            search_results = st.session_state.search_results
            st.success("✅ 기존 검색 결과가 있습니다!")

            col1, col2 = st.columns(2)
            with col1:
                search_count = len(search_results.get("results", {}))
                st.metric("검색된 특허 수", search_count)
            with col2:
                # 웹 스크래핑된 특허 수 계산
                scraping_count = 0
                for result in search_results.get("results", {}).values():
                    if result.get("scraping_success", False):
                        scraping_count += 1
                st.metric("스크래핑된 특허 수", scraping_count)

            if st.button(
                "💾 기존 검색 결과를 Vector Store에 저장",
                disabled=not st.session_state.vector_store,
            ):
                with st.spinner("데이터 처리 및 저장 중..."):
                    try:
                        # 필수 클래스 확인
                        if PatentDataProcessor is None:
                            st.error(
                                "PatentDataProcessor 클래스를 import할 수 없습니다."
                            )
                            return

                        # 데이터 처리 - search_results에서 스크래핑 데이터 추출
                        processor = PatentDataProcessor()

                        # 스크래핑 결과를 별도 구조로 생성
                        scraping_results = {"results": []}
                        for patent_id, result in search_results.get(
                            "results", {}
                        ).items():
                            if result.get("scraping_success", False):
                                scraping_result = {
                                    "patent_id": patent_id,
                                    "url": result.get("description_link", ""),
                                    "title": result.get(
                                        "scraped_title", result.get("title", "")
                                    ),
                                    "abstract": result.get(
                                        "scraped_abstract", result.get("abstract", "")
                                    ),
                                    "description": result.get(
                                        "scraped_description", ""
                                    ),
                                    "claims": result.get(
                                        "scraped_claims", result.get("claims", [])
                                    ),
                                    "success": True,
                                }
                                scraping_results["results"].append(scraping_result)

                        patents, chunks = processor.process_search_results(
                            search_results,
                            scraping_results,
                        )

                        if patents and chunks:
                            st.success(
                                f"✅ {len(patents)}개 특허, {len(chunks)}개 청크 처리 완료!"
                            )

                            # Vector Store에 저장
                            success_count, error_count = (
                                st.session_state.vector_store.add_document_chunks_batch(
                                    chunks
                                )
                            )

                            if success_count > 0:
                                st.success(
                                    f"🎉 {success_count}개 청크 저장 완료! (오류: {error_count}개)"
                                )
                                st.session_state.processed_patents.extend(patents)

                                # 결과 표시
                                with st.expander("📋 처리된 특허 목록", expanded=True):
                                    for i, patent in enumerate(patents, 1):
                                        st.markdown(f"**{i}. {patent.title}**")
                                        st.markdown(
                                            f"특허번호: `{patent.patent_number}`"
                                        )
                                        st.markdown(
                                            f"출원일: {patent.filing_date or 'N/A'}"
                                        )
                                        st.markdown(
                                            f"청크 수: {len([c for c in chunks if c.patent_number == patent.patent_number])}"
                                        )
                                        st.markdown("---")
                            else:
                                st.error(f"❌ 저장 실패! 오류: {error_count}개")
                        else:
                            st.warning("처리된 데이터가 없습니다.")

                    except Exception as e:
                        st.error(f"처리 중 오류 발생: {e}")

        else:
            st.warning("⚠️ 저장할 검색 결과가 없습니다.")
            st.info(
                "먼저 '🔍 특허 검색' 페이지에서 특허를 검색하고 스크래핑을 완료해주세요."
            )

        # 샘플 데이터 로드 옵션
        st.markdown("---")
        st.subheader("📁 샘플 데이터 사용")

        # 기존 샘플 데이터 파일 확인
        import json
        from pathlib import Path

        sample_files = []
        for file_path in [
            "test_data.json",
            "sample_patent_data.json",
            "data/test_patents.json",
        ]:
            if Path(file_path).exists():
                sample_files.append(file_path)

        if sample_files:
            selected_file = st.selectbox("샘플 데이터 파일 선택:", sample_files)

            if st.button("📂 샘플 데이터 로드 및 저장"):
                try:
                    with open(selected_file, "r", encoding="utf-8") as f:
                        sample_data = json.load(f)

                    # 데이터 형식에 따라 처리
                    if isinstance(sample_data, list) and len(sample_data) > 0:
                        # Patent 객체 리스트로 변환
                        from src.patent_search.patent_parser import Patent

                        patents = []
                        for patent_data in sample_data:
                            if isinstance(patent_data, dict):
                                patent = Patent(
                                    patent_number=patent_data.get(
                                        "patent_number", "UNKNOWN"
                                    ),
                                    title=patent_data.get("title", "제목 없음"),
                                    abstract=patent_data.get("abstract", ""),
                                    description_html=patent_data.get(
                                        "description_html", ""
                                    ),
                                    claims=patent_data.get("claims", []),
                                    filing_date=patent_data.get("filing_date"),
                                    inventors=patent_data.get("inventors", []),
                                    assignees=patent_data.get("assignees", []),
                                )
                                patents.append(patent)

                        if patents:
                            # 청킹 처리
                            processor = PatentDataProcessor()
                            chunks = processor.create_chunks_from_patents(patents)

                            if chunks:
                                st.success(
                                    f"✅ {len(patents)}개 특허, {len(chunks)}개 청크 로드 완료!"
                                )

                                # Vector Store에 저장
                                success_count, error_count = (
                                    st.session_state.vector_store.add_document_chunks_batch(
                                        chunks
                                    )
                                )

                                if success_count > 0:
                                    st.success(
                                        f"🎉 {success_count}개 청크 저장 완료! (오류: {error_count}개)"
                                    )
                                else:
                                    st.error(f"❌ 저장 실패! 오류: {error_count}개")
                            else:
                                st.warning("청킹 처리에 실패했습니다.")
                        else:
                            st.warning("유효한 특허 데이터가 없습니다.")
                    else:
                        st.warning("지원하지 않는 데이터 형식입니다.")

                except Exception as e:
                    st.error(f"샘플 데이터 로드 실패: {e}")
        else:
            st.info("사용 가능한 샘플 데이터 파일이 없습니다.")

    with tab2:
        st.subheader("Vector Store 유사도 검색")

        if not st.session_state.vector_store:
            st.warning("먼저 Vector Store를 초기화해주세요.")
            return

        col1, col2 = st.columns([3, 1])
        with col1:
            similarity_query = st.text_input(
                "검색 쿼리를 입력하세요:",
                placeholder="예: deep learning algorithm, neural network",
                help="저장된 특허 청크에서 유사한 내용을 찾습니다.",
            )
        with col2:
            search_limit = st.selectbox("결과 개수:", [5, 10, 20], index=1)

        if st.button("🎯 유사도 검색"):
            if similarity_query:
                with st.spinner("검색 중..."):
                    start_time = time.time()
                    results = st.session_state.vector_store.search_similar(
                        similarity_query, n_results=search_limit, include_distances=True
                    )
                    search_time = time.time() - start_time

                st.success(
                    f"✅ 검색 완료! ({search_time:.3f}초, {results['total_results']}개 결과)"
                )

                if results["results"]:
                    for i, result in enumerate(results["results"], 1):
                        with st.expander(
                            f"📄 결과 {i} (유사도: {result.get('similarity', 0):.3f})",
                            expanded=i <= 3,
                        ):
                            st.markdown(f"**청크 ID:** `{result['chunk_id']}`")

                            # 메타데이터 확인 및 디버깅
                            metadata = result.get("metadata", {})

                            # 다양한 가능한 필드명들을 시도
                            patent_number = (
                                metadata.get("patent_number")
                                or metadata.get("patent_id")
                                or metadata.get("id")
                                or "N/A"
                            )

                            section = (
                                metadata.get("section_type")
                                or metadata.get("section")
                                or metadata.get("chunk_type")
                                or metadata.get("type")
                                or "N/A"
                            )

                            st.markdown(f"**특허번호:** {patent_number}")
                            st.markdown(f"**섹션:** {section}")
                            st.markdown(
                                f"**거리:** {result.get('distance', 'N/A'):.4f}"
                            )

                            # 메타데이터 상세 정보 표시 (처음 10개 결과)
                            if i <= 10:
                                st.markdown("**📋 메타데이터 상세:**")
                                st.json(metadata)

                            st.markdown("**내용:**")
                            st.markdown(f"```\n{result['content'][:500]}...\n```")
                else:
                    st.info("검색 결과가 없습니다.")
            else:
                st.warning("검색 쿼리를 입력해주세요.")

    with tab3:
        st.subheader("Vector Store 관리")

        if not st.session_state.vector_store:
            st.warning("먼저 Vector Store를 초기화해주세요.")
            return

        # 통계 정보
        try:
            stats = st.session_state.vector_store.get_stats()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("총 청크 수", stats.get("total_chunks", 0))
            with col2:
                st.metric("고유 특허 수", stats.get("unique_patents", 0))
            with col3:
                st.metric(
                    "컬렉션 크기 (MB)", f"{stats.get('collection_size_mb', 0):.2f}"
                )

            # 성능 모니터링
            if (
                hasattr(st.session_state.vector_store, "performance_monitor")
                and st.session_state.vector_store.performance_monitor
            ):
                st.subheader("📈 성능 모니터링")
                monitor = st.session_state.vector_store.performance_monitor

                if monitor.operations:
                    performance_stats = monitor.get_stats()

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "총 작업 수", performance_stats.get("total_operations", 0)
                        )
                    with col2:
                        st.metric(
                            "성공률", f"{performance_stats.get('success_rate', 0):.1f}%"
                        )
                    with col3:
                        st.metric(
                            "평균 응답시간",
                            f"{performance_stats.get('avg_duration', 0):.3f}초",
                        )

                    # 작업별 통계
                    op_breakdown = performance_stats.get("operation_breakdown", {})
                    if op_breakdown:
                        st.subheader("작업별 통계")
                        for op_type, op_stats in op_breakdown.items():
                            with st.expander(f"{op_type} 통계"):
                                st.json(op_stats)
                else:
                    st.info("아직 성능 데이터가 없습니다.")

            # 관리 작업
            st.subheader("🛠️ 관리 작업")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Vector Store 재설정"):
                    if st.session_state.vector_store:
                        st.session_state.vector_store.reset_collection()
                        st.success("Vector Store가 재설정되었습니다.")
                        st.rerun()

            with col2:
                if st.button("📊 상세 통계 보기"):
                    st.json(stats)

        except Exception as e:
            st.error(f"통계 정보를 가져올 수 없습니다: {e}")


def show_question_management_page():
    """Question management page"""
    st.header("❓ 질문 관리")

    st.markdown(
        """
    특허 분석을 위한 질문을 관리하세요.
    각 특허에 대해 동일한 질문들이 적용됩니다.
    """
    )

    # Default questions
    default_questions = [
        "이 특허의 주요 기술적 특징은 무엇인가요?",
        "이 특허의 응용 분야는 어떻게 되나요?",
        "이 특허의 혁신성은 무엇인가요?",
        "이 특허의 상업적 가치는 어떻게 평가되나요?",
    ]

    # Initialize questions in session state
    if "custom_questions" not in st.session_state:
        st.session_state["custom_questions"] = default_questions.copy()

    st.subheader("📝 현재 질문 목록")

    # Display current questions
    for i, question in enumerate(st.session_state["custom_questions"]):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"{i+1}. {question}")
        with col2:
            if st.button("🗑️", key=f"delete_{i}", help="질문 삭제"):
                st.session_state["custom_questions"].pop(i)
                st.rerun()

    # Add new question
    st.subheader("➕ 새 질문 추가")
    new_question = st.text_input("새로운 질문을 입력하세요:")

    if st.button("질문 추가") and new_question:
        st.session_state["custom_questions"].append(new_question)
        st.success("✅ 질문이 추가되었습니다!")
        st.rerun()

    # Reset to defaults
    if st.button("🔄 기본 질문으로 초기화"):
        st.session_state["custom_questions"] = default_questions.copy()
        st.success("✅ 기본 질문으로 초기화되었습니다!")
        st.rerun()


def show_rag_analysis_page():
    """RAG analysis page"""
    st.header("🤖 RAG 분석")
    st.info("🚧 RAG 분석 기능은 다음 단계에서 구현됩니다.")


def show_results_page():
    """Results page"""
    st.header("📊 결과 보기")
    st.info("🚧 결과 보기 기능은 다음 단계에서 구현됩니다.")


def show_settings_page():
    """Settings page"""
    st.header("⚙️ 설정")

    st.markdown(
        """
    ### 환경 설정 확인
    
    API 키가 올바르게 설정되어 있는지 확인하세요:
    """
    )

    # Environment variables check
    from dotenv import load_dotenv

    load_dotenv()

    env_vars = {
        "SERPAPI_API_KEY": "SerpAPI (특허 검색용)",
        "OPENAI_API_KEY": "OpenAI (AI 분석용)",
    }

    for var, description in env_vars.items():
        value = os.getenv(var)
        if value and value != f"your_{var.lower()}_here":
            st.success(f"✅ {description}: 설정됨")
        else:
            st.error(f"❌ {description}: 설정 필요")

    st.markdown(
        """
    ### 설정 방법
    
    1. 프로젝트 루트 디렉토리의 `.env` 파일을 편집하세요.
    2. 각 API 키를 실제 값으로 변경하세요:
    
    ```
    SERPAPI_API_KEY=your_actual_serpapi_key
    OPENAI_API_KEY=your_actual_openai_key
    ```
    
    3. 파일을 저장하고 애플리케이션을 재시작하세요.
    """
    )

    # System information
    st.subheader("💻 시스템 정보")
    st.write(f"**Python 버전:** {st.__version__}")
    st.write(f"**작업 디렉토리:** {os.getcwd()}")


def show_langgraph_crag_page():
    """LangGraph 기반 Corrective RAG (CRAG) 분석 페이지 (하이브리드 검색 + 재랭킹)"""
    st.title("🚀 고도화된 LangGraph CRAG 분석")
    st.markdown("**하이브리드 검색 + Cross-encoder 재랭킹 기반 고급 RAG 파이프라인**")

    # CRAG Pipeline import (지연 로딩)
    try:
        from src.langgraph_crag.crag_pipeline import CorrectiveRAGPipeline

        crag_available = True
    except ImportError as e:
        st.error(f"LangGraph CRAG 모듈을 가져올 수 없습니다: {e}")
        crag_available = False
        return

    # Vector Store 확인
    if not st.session_state.get("vector_store"):
        st.warning(
            "🔍 Vector Store가 초기화되지 않았습니다. Vector Store 테스트 페이지에서 먼저 초기화해주세요."
        )
        return

    # Pipeline Configuration
    st.sidebar.header("🔧 고도화된 CRAG 설정")

    with st.sidebar.expander("모델 설정", expanded=True):
        model_name = st.selectbox(
            "LLM 모델",
            ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            index=0,
        )

    with st.sidebar.expander("검색 시스템 설정", expanded=True):
        # 하이브리드 검색 설정
        enable_hybrid_search = st.checkbox(
            "🔍 하이브리드 검색 활성화",
            value=True,
            help="Vector Search + BM25 키워드 검색 결합",
        )

        if enable_hybrid_search:
            hybrid_vector_weight = st.slider(
                "Vector Search 가중치",
                min_value=0.1,
                max_value=0.9,
                value=0.6,
                step=0.1,
                help="하이브리드 검색에서 Vector Search 비중 (나머지는 BM25)",
            )
        else:
            hybrid_vector_weight = 0.6

        # 재랭킹 설정
        enable_reranking = st.checkbox(
            "🎯 Jina AI 재랭킹 활성화",
            value=True,
            help="Jina AI의 고성능 reranker를 통한 정밀한 질문-문서 관련성 재평가",
        )

        if enable_reranking:
            rerank_top_k = st.slider(
                "재랭킹 대상 문서 수",
                min_value=5,
                max_value=20,
                value=10,
                step=1,
                help="상위 몇 개 문서를 재랭킹할지 설정",
            )

            # Jina 모델 선택
            jina_model = st.selectbox(
                "Jina Reranker 모델",
                [
                    "jina-reranker-v2-base-multilingual",
                    "jina-colbert-v2",
                    "jina-reranker-v1-base-en",
                    "jina-reranker-v1-tiny-en",
                ],
                index=0,
                help="사용할 Jina reranker 모델을 선택하세요",
            )
        else:
            rerank_top_k = 10
            jina_model = "jina-reranker-v2-base-multilingual"

    with st.sidebar.expander("파이프라인 설정", expanded=True):
        max_retries = st.slider("최대 재시도 횟수", 1, 5, 2)

    with st.sidebar.expander("API 키 확인", expanded=False):
        openai_key_status = "✅ 설정됨" if os.getenv("OPENAI_API_KEY") else "❌ 필요"
        jina_key_status = "✅ 설정됨" if os.getenv("JINA_API_KEY") else "❌ 필요"

        st.markdown(f"**OpenAI API:** {openai_key_status}")
        st.markdown(f"**Jina AI API:** {jina_key_status}")

        # 검색 시스템 정보 표시
        search_info = []
        if enable_hybrid_search:
            search_info.append("🔍 하이브리드 검색 (Vector + BM25)")
        else:
            search_info.append("📊 Vector Store 검색")

        if enable_reranking:
            search_info.append("🎯 Jina AI 재랭킹")

        st.info(f"💡 검색 모드: {' + '.join(search_info)}")

        if not os.getenv("OPENAI_API_KEY"):
            st.error("OpenAI API 키가 필요합니다!")
            return

        if enable_reranking and not os.getenv("JINA_API_KEY"):
            st.error("Jina AI 재랭킹을 위해서는 Jina API 키가 필요합니다!")
            return

    # Initialize pipeline
    pipeline_config_changed = (
        st.session_state.get("last_hybrid_search") != enable_hybrid_search
        or st.session_state.get("last_reranking") != enable_reranking
        or st.session_state.get("last_vector_weight") != hybrid_vector_weight
        or st.session_state.get("last_rerank_top_k") != rerank_top_k
        or st.session_state.get("last_jina_model") != jina_model
    )

    if (
        "crag_pipeline" not in st.session_state
        or pipeline_config_changed
        or st.button("🔄 CRAG Pipeline 재초기화")
    ):

        try:
            with st.spinner("고도화된 CRAG Pipeline 초기화 중..."):
                st.session_state.crag_pipeline = CorrectiveRAGPipeline(
                    vector_store=st.session_state.vector_store,
                    model_name=model_name,
                    max_retries=max_retries,
                    enable_hybrid_search=enable_hybrid_search,
                    enable_reranking=enable_reranking,
                    hybrid_vector_weight=hybrid_vector_weight,
                    rerank_top_k=rerank_top_k,
                    jina_api_key=os.getenv("JINA_API_KEY"),
                    jina_model=jina_model,
                )

                # 설정 저장
                st.session_state.last_hybrid_search = enable_hybrid_search
                st.session_state.last_reranking = enable_reranking
                st.session_state.last_vector_weight = hybrid_vector_weight
                st.session_state.last_rerank_top_k = rerank_top_k
                st.session_state.last_jina_model = jina_model

            st.success("✅ 고도화된 LangGraph CRAG Pipeline 초기화 완료!")
        except Exception as e:
            st.error(f"Pipeline 초기화 실패: {e}")
            st.exception(e)
            return

    # Pipeline Info
    if st.session_state.get("crag_pipeline"):
        with st.expander("📊 고도화된 Pipeline 정보", expanded=False):
            pipeline_info = st.session_state.crag_pipeline.get_pipeline_info()

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**파이프라인 유형:**")
                st.code(pipeline_info["pipeline_type"])
                st.markdown("**모델:**")
                st.code(pipeline_info["model_name"])
                st.markdown("**최대 재시도:**")
                st.code(pipeline_info["max_retries"])

            with col2:
                st.markdown("**주요 기능:**")
                for feature in pipeline_info["features"]:
                    st.markdown(f"• {feature}")

            st.markdown("**워크플로우:**")
            for step in pipeline_info["workflow"]:
                st.markdown(f"• {step}")

            # 검색 설정 정보 추가
            if "search_configuration" in pipeline_info:
                st.markdown("---")
                st.markdown("**🔍 검색 시스템 설정:**")
                search_config = pipeline_info["search_configuration"]

                config_col1, config_col2 = st.columns(2)
                with config_col1:
                    hybrid_status = (
                        "✅ 활성화"
                        if search_config.get("hybrid_search_enabled")
                        else "❌ 비활성화"
                    )
                    st.markdown(f"• **하이브리드 검색:** {hybrid_status}")

                    if search_config.get("hybrid_search_enabled"):
                        vector_weight = search_config.get("hybrid_vector_weight", 0.6)
                        bm25_weight = search_config.get("hybrid_bm25_weight", 0.4)
                        st.markdown(f"  - Vector 가중치: {vector_weight:.1f}")
                        st.markdown(f"  - BM25 가중치: {bm25_weight:.1f}")

                with config_col2:
                    rerank_status = (
                        "✅ 활성화"
                        if search_config.get("reranking_enabled")
                        else "❌ 비활성화"
                    )
                    st.markdown(f"• **Cross-encoder 재랭킹:** {rerank_status}")

                    if search_config.get("reranking_enabled"):
                        rerank_model = search_config.get("reranker_model", "N/A")
                        rerank_loaded = search_config.get("reranker_loaded", False)
                        st.markdown(f"  - 모델: {rerank_model}")
                        st.markdown(f"  - 로딩 상태: {'✅' if rerank_loaded else '⏳'}")
                        st.markdown(
                            f"  - 재랭킹 대상: 상위 {search_config.get('rerank_top_k', 10)}개"
                        )

    # Main Query Interface
    st.header("💬 특허 질의")

    # Pre-defined example queries
    example_queries = [
        "무선충전 기술의 주요 특허는 무엇인가요?",
        "배터리 관련 최신 특허 기술을 설명해주세요.",
        "OLED 디스플레이 특허의 기술적 특징은?",
        "AI 반도체 관련 특허 동향을 알려주세요.",
        "전기차 배터리 관리 시스템 특허는?",
    ]

    query_input_method = st.radio("질의 입력 방법:", ["직접 입력", "예시 선택"])

    if query_input_method == "직접 입력":
        user_query = st.text_area(
            "질의를 입력하세요:",
            placeholder="예: 무선충전 기술의 주요 특허는 무엇인가요?",
            height=100,
        )
    else:
        user_query = st.selectbox("예시 질의 선택:", [""] + example_queries)

    # Process Query
    if st.button("🎯 CRAG 분석 실행", disabled=not user_query.strip()):
        if not st.session_state.get("crag_pipeline"):
            st.error("CRAG Pipeline이 초기화되지 않았습니다.")
            return

        with st.spinner("LangGraph CRAG Pipeline 실행 중..."):
            try:
                # Process query through CRAG Pipeline
                result = st.session_state.crag_pipeline.process_query(
                    user_query.strip()
                )

                # Store result in session state
                st.session_state.last_crag_result = result

                # Display Results
                if result["success"]:
                    st.success("✅ CRAG 분석 완료!")

                    # Metadata display
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        retry_icon = (
                            "🔄"
                            if result["metadata"].get("retry_count", 0) > 0
                            else "✅"
                        )
                        st.metric(
                            "재시도 횟수",
                            f"{retry_icon} {result['metadata'].get('retry_count', 0)}",
                        )
                    with col2:
                        rewrite_icon = (
                            "✏️"
                            if result["metadata"].get("query_rewritten", False)
                            else "📝"
                        )
                        st.metric(
                            "질문 재작성",
                            f"{rewrite_icon} {'예' if result['metadata'].get('query_rewritten', False) else '아니오'}",
                        )
                    with col3:
                        st.metric(
                            "특허 문서 수",
                            f"📄 {result['metadata'].get('documents_found', 0)}",
                        )
                    with col4:
                        # 검색 방식 표시
                        search_method = (
                            "🔍 하이브리드"
                            if result["metadata"].get("hybrid_search_used", False)
                            else "📊 Vector"
                        )
                        rerank_used = result["metadata"].get("reranking_used", False)
                        if rerank_used:
                            search_method += " + 재랭킹"
                        st.metric("검색 방식", search_method)

                    # 검색 시스템 정보 배지
                    search_badges = []
                    if result["metadata"].get("hybrid_search_used", False):
                        search_badges.append("🔍 하이브리드 검색")
                    if result["metadata"].get("reranking_used", False):
                        search_badges.append("🎯 Cross-encoder 재랭킹")
                    if result["metadata"].get("vector_store_only"):
                        search_badges.append("🏛️ Vector Store 전용")

                    if search_badges:
                        st.info(" + ".join(search_badges))

                    # Final Answer
                    st.subheader("💡 생성된 답변")
                    with st.expander("답변 내용", expanded=True):
                        st.markdown(result["answer"])

                    # Process Log
                    st.subheader("🔍 처리 과정")
                    with st.expander("상세 로그", expanded=False):
                        for i, log_entry in enumerate(result["process_log"], 1):
                            st.markdown(f"**{i}.** {log_entry}")

                    # Retrieved Documents
                    if result["documents"]:
                        st.subheader("📚 검색된 특허 문서")

                        # 재랭킹 정보 요약
                        has_rerank_scores = any(
                            doc.get("rerank_score") is not None
                            for doc in result["documents"]
                        )
                        if has_rerank_scores:
                            avg_rerank_score = sum(
                                doc.get("rerank_score", 0)
                                for doc in result["documents"]
                            ) / len(result["documents"])
                            max_rerank_score = max(
                                doc.get("rerank_score", 0)
                                for doc in result["documents"]
                            )
                            min_rerank_score = min(
                                doc.get("rerank_score", 0)
                                for doc in result["documents"]
                            )

                            rerank_col1, rerank_col2, rerank_col3 = st.columns(3)
                            with rerank_col1:
                                st.metric(
                                    "평균 재랭킹 스코어", f"{avg_rerank_score:.3f}"
                                )
                            with rerank_col2:
                                st.metric("최고 스코어", f"{max_rerank_score:.3f}")
                            with rerank_col3:
                                st.metric("최저 스코어", f"{min_rerank_score:.3f}")

                        with st.expander(
                            f"특허 문서 목록 ({len(result['documents'])}개)",
                            expanded=False,
                        ):
                            for i, doc in enumerate(result["documents"], 1):
                                with st.container():
                                    # 문서 제목에 재랭킹 정보 포함
                                    rerank_score = doc.get("rerank_score")
                                    original_pos = doc.get("original_position")
                                    rerank_pos = doc.get("rerank_position")

                                    title_parts = [f"**특허 문서 {i}**"]
                                    if rerank_score is not None:
                                        score_emoji = (
                                            "🎯"
                                            if rerank_score > 0.7
                                            else "🔍" if rerank_score > 0.3 else "📄"
                                        )
                                        title_parts.append(
                                            f"{score_emoji} 관련도: {rerank_score:.3f}"
                                        )

                                    if (
                                        original_pos
                                        and rerank_pos
                                        and original_pos != rerank_pos
                                    ):
                                        if original_pos > rerank_pos:
                                            title_parts.append(
                                                f"📈 순위 상승: {original_pos}→{rerank_pos}"
                                            )
                                        else:
                                            title_parts.append(
                                                f"📉 순위 하락: {original_pos}→{rerank_pos}"
                                            )

                                    st.markdown(" | ".join(title_parts))

                                    # Document content preview
                                    content = doc.get("content", str(doc))
                                    if isinstance(content, dict):
                                        content = content.get(
                                            "page_content", str(content)
                                        )

                                    if len(content) > 500:
                                        preview = content[:500] + "..."
                                    else:
                                        preview = content

                                    st.text_area(
                                        f"내용 미리보기 {i}",
                                        value=preview,
                                        height=100,
                                        disabled=True,
                                        key=f"doc_preview_{i}",
                                        label_visibility="collapsed",
                                    )

                                    # Document metadata
                                    if isinstance(doc, dict) and "metadata" in doc:
                                        metadata = doc["metadata"]

                                        # 특허 번호와 섹션 정보 표시
                                        meta_col1, meta_col2, meta_col3 = st.columns(3)
                                        with meta_col1:
                                            st.markdown(
                                                f"**특허번호:** `{metadata.get('patent_number', 'N/A')}`"
                                            )
                                        with meta_col2:
                                            st.markdown(
                                                f"**섹션:** `{metadata.get('section_type', 'N/A')}`"
                                            )
                                        with meta_col3:
                                            if rerank_score is not None:
                                                st.markdown(
                                                    f"**재랭킹 스코어:** `{rerank_score:.3f}`"
                                                )

                                        # 전체 메타데이터는 접을 수 있는 형태로
                                        with st.expander(
                                            "전체 메타데이터", expanded=False
                                        ):
                                            # 재랭킹 관련 정보도 포함
                                            full_metadata = metadata.copy()
                                            if rerank_score is not None:
                                                full_metadata["rerank_score"] = (
                                                    rerank_score
                                                )
                                            if original_pos:
                                                full_metadata["original_position"] = (
                                                    original_pos
                                                )
                                            if rerank_pos:
                                                full_metadata["rerank_position"] = (
                                                    rerank_pos
                                                )
                                            st.json(full_metadata)

                                    st.divider()

                else:
                    st.error("❌ CRAG 분석 실패")
                    st.error(result["answer"])

                    if result.get("process_log"):
                        with st.expander("오류 로그", expanded=True):
                            for log_entry in result["process_log"]:
                                st.text(log_entry)

            except Exception as e:
                st.error(f"CRAG Pipeline 실행 중 오류 발생: {e}")
                st.exception(e)


def display_judgement_result(judge_name: str, judgement):
    """Display judgement result in a formatted way"""

    # Score display with color coding
    score_color = (
        "🟢" if judgement.score >= 0.8 else "🟡" if judgement.score >= 0.6 else "🔴"
    )
    pass_icon = "✅" if judgement.pass_threshold else "❌"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("점수", f"{score_color} {judgement.score:.3f}")
    with col2:
        st.metric("신뢰도", f"{judgement.confidence:.3f}")
    with col3:
        st.metric("통과", f"{pass_icon}")

    # Explanation
    st.markdown("**설명:**")
    st.info(judgement.explanation)

    # Details specific to each judge type
    if judge_name == "relevance" and judgement.details:
        if judgement.details.get("relevant_docs"):
            st.markdown("**관련 문서:**")
            st.code(", ".join(judgement.details["relevant_docs"]))
        if judgement.details.get("irrelevant_docs"):
            st.markdown("**무관한 문서:**")
            st.code(", ".join(judgement.details["irrelevant_docs"]))

    elif judge_name == "hallucination" and judgement.details:
        col1, col2 = st.columns(2)
        with col1:
            if judgement.details.get("supported_claims"):
                st.markdown("**지원되는 주장:**")
                st.success(judgement.details["supported_claims"])
        with col2:
            if judgement.details.get("unsupported_claims"):
                st.markdown("**지원되지 않는 주장:**")
                st.warning(judgement.details["unsupported_claims"])

        risk_level = judgement.details.get("hallucination_risk", "중간")
        risk_color = {"낮음": "🟢", "중간": "🟡", "높음": "🔴"}.get(risk_level, "🟡")
        st.markdown(f"**할루시네이션 위험도:** {risk_color} {risk_level}")

    elif judge_name == "quality" and judgement.details:
        quality_metrics = [
            "completeness",
            "clarity",
            "accuracy",
            "structure",
            "usefulness",
        ]
        metric_names = ["완성도", "명확성", "정확성", "구조", "유용성"]

        cols = st.columns(len(quality_metrics))
        for i, (metric, name) in enumerate(zip(quality_metrics, metric_names)):
            if metric in judgement.details:
                with cols[i]:
                    value = judgement.details[metric]
                    st.metric(name, f"{value:.2f}")

        if judgement.details.get("improvements"):
            st.markdown("**개선 제안:**")
            st.warning(judgement.details["improvements"])


if __name__ == "__main__":
    main()
