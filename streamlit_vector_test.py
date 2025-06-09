#!/usr/bin/env python3
"""
Vector Store 테스트용 Streamlit 웹앱
Task 1-6까지 구현된 기능들을 테스트합니다.
"""

import streamlit as st
import os
import sys
import json
import time
from typing import List, Dict, Any

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.patent_search.serp_client import SerpPatentClient
from src.patent_search.description_scraper import GooglePatentsDescriptionScraper
from src.patent_search.patent_parser import PatentParser, PatentDataProcessor
from src.vector_store.patent_vector_store import PatentVectorStore
from src.vector_store.embedding_manager import EmbeddingManager

# 페이지 설정
st.set_page_config(
    page_title="특허 Vector Store 테스트",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    """세션 상태 초기화"""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "processed_patents" not in st.session_state:
        st.session_state.processed_patents = []


@st.cache_resource
def load_vector_store():
    """Vector Store 초기화 및 로드"""
    try:
        vector_store = PatentVectorStore(
            persist_directory="./streamlit_chroma_db",
            collection_name="patent_test",
            reset_collection=False,  # 기존 데이터 유지
            enable_preprocessing=True,
            enable_performance_monitoring=True,
        )
        return vector_store
    except Exception as e:
        st.error(f"Vector Store 초기화 실패: {e}")
        return None


def search_and_process_patents(query: str, num_results: int = 5):
    """특허 검색 및 처리"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # 1. SERP API 검색
        status_text.text("🔍 특허 검색 중...")
        progress_bar.progress(20)

        serp_client = SerpPatentClient()
        search_results = serp_client.search_patents(query, num_results=num_results)

        if not search_results or "organic_results" not in search_results:
            st.error("검색 결과가 없습니다.")
            return []

        # 2. 웹 스크래핑
        status_text.text("📄 특허 상세 정보 수집 중...")
        progress_bar.progress(40)

        scraper = GooglePatentsDescriptionScraper()
        scraping_results = scraper.scrape_batch_descriptions(
            search_results["organic_results"], max_workers=2
        )

        # 3. 데이터 처리 및 청킹
        status_text.text("⚙️ 데이터 처리 및 청킹 중...")
        progress_bar.progress(60)

        processor = PatentDataProcessor()
        patents, chunks = processor.process_search_results(
            search_results, scraping_results
        )

        progress_bar.progress(100)
        status_text.text("✅ 처리 완료!")

        return patents, chunks

    except Exception as e:
        st.error(f"처리 중 오류 발생: {e}")
        return [], []


def main():
    """메인 앱"""
    init_session_state()

    st.title("🔬 특허 Vector Store 테스트")
    st.markdown("Task 1-6까지 구현된 기능들을 테스트해보세요!")

    # 사이드바 설정
    st.sidebar.header("⚙️ 설정")

    # Vector Store 초기화
    if st.sidebar.button("🚀 Vector Store 초기화"):
        st.session_state.vector_store = load_vector_store()
        if st.session_state.vector_store:
            st.sidebar.success("Vector Store 초기화 완료!")
        else:
            st.sidebar.error("Vector Store 초기화 실패!")

    # Vector Store 상태 표시
    if st.session_state.vector_store:
        st.sidebar.success("✅ Vector Store 준비됨")

        # Vector Store 통계
        try:
            stats = st.session_state.vector_store.get_stats()
            st.sidebar.metric("저장된 청크 수", stats.get("total_chunks", 0))
            st.sidebar.metric("고유 특허 수", stats.get("unique_patents", 0))
        except:
            pass
    else:
        st.sidebar.warning("❌ Vector Store 미초기화")

    # 탭 생성
    tab1, tab2, tab3 = st.tabs(
        ["🔍 특허 검색 & 저장", "🎯 유사도 검색", "📊 Vector Store 관리"]
    )

    with tab1:
        st.header("특허 검색 및 Vector Store 저장")

        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input(
                "검색어를 입력하세요:",
                placeholder="예: artificial intelligence, machine learning",
                help="영어로 입력하면 더 정확한 결과를 얻을 수 있습니다.",
            )
        with col2:
            num_results = st.selectbox("검색 결과 수:", [3, 5, 10], index=1)

        if st.button(
            "🔍 검색 및 처리 시작", disabled=not st.session_state.vector_store
        ):
            if search_query:
                with st.spinner("처리 중..."):
                    patents, chunks = search_and_process_patents(
                        search_query, num_results
                    )

                    if patents and chunks:
                        st.success(
                            f"✅ {len(patents)}개 특허, {len(chunks)}개 청크 처리 완료!"
                        )

                        # Vector Store에 저장
                        st.info("Vector Store에 저장 중...")
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
                        else:
                            st.error(f"❌ 저장 실패! 오류: {error_count}개")

                        # 결과 표시
                        with st.expander("📋 처리된 특허 목록", expanded=True):
                            for i, patent in enumerate(patents, 1):
                                st.markdown(f"**{i}. {patent.title}**")
                                st.markdown(f"특허번호: `{patent.patent_number}`")
                                st.markdown(f"출원일: {patent.filing_date or 'N/A'}")
                                st.markdown(
                                    f"청크 수: {len([c for c in chunks if c.patent_number == patent.patent_number])}"
                                )
                                st.markdown("---")
                    else:
                        st.warning("검색 결과가 없거나 처리에 실패했습니다.")
            else:
                st.warning("검색어를 입력해주세요.")

    with tab2:
        st.header("Vector Store 유사도 검색")

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
                            st.markdown(
                                f"**특허번호:** {result['metadata'].get('patent_number', 'N/A')}"
                            )
                            st.markdown(
                                f"**섹션:** {result['metadata'].get('section', 'N/A')}"
                            )
                            st.markdown(
                                f"**거리:** {result.get('distance', 'N/A'):.4f}"
                            )
                            st.markdown("**내용:**")
                            st.markdown(f"```\n{result['content'][:500]}...\n```")
                else:
                    st.info("검색 결과가 없습니다.")
            else:
                st.warning("검색 쿼리를 입력해주세요.")

    with tab3:
        st.header("Vector Store 관리")

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
                    "평균 청크 길이", f"{stats.get('avg_chunk_length', 0):.0f} 문자"
                )

        except Exception as e:
            st.error(f"통계 정보 로드 실패: {e}")

        # 성능 모니터링
        if (
            hasattr(st.session_state.vector_store, "performance_monitor")
            and st.session_state.vector_store.performance_monitor
        ):
            st.subheader("📈 성능 모니터링")

            monitor = st.session_state.vector_store.performance_monitor
            overall_stats = monitor.get_overall_stats()

            col1, col2 = st.columns(2)
            with col1:
                st.metric("총 작업 수", overall_stats["total_operations"])
                st.metric("성공률", f"{overall_stats['success_rate']:.1%}")
            with col2:
                st.metric("평균 응답 시간", f"{overall_stats['avg_duration']:.3f}초")

            # 작업별 성능
            if overall_stats["operations_by_type"]:
                st.subheader("작업별 성능")
                for op_name, op_stats in overall_stats["operations_by_type"].items():
                    st.markdown(
                        f"**{op_name}**: {op_stats['count']}회, 평균 {op_stats['avg_duration']:.3f}초"
                    )

            # 권장사항
            recommendations = monitor.generate_recommendations()
            if recommendations:
                st.subheader("🔧 최적화 권장사항")
                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f"{i}. {rec}")

        # 관리 작업
        st.subheader("🛠️ 관리 작업")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Vector Store 초기화", type="secondary"):
                if st.session_state.vector_store:
                    st.session_state.vector_store.reset_collection()
                    st.success("Vector Store가 초기화되었습니다.")
                    st.rerun()

        with col2:
            if st.button("💾 테스트 데이터 로드", type="secondary"):
                st.info("기존 테스트 데이터를 로드하는 기능은 추후 구현 예정입니다.")


if __name__ == "__main__":
    main()
