#!/usr/bin/env python3
"""전처리 통합 Vector Store 테스트 스크립트"""

import json
import logging
import os
import sys

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.vector_store.embedding_manager import EmbeddingManager
from src.vector_store.patent_vector_store import PatentVectorStore
from src.patent_search.patent_parser import DocumentChunk


def test_integrated_preprocessing():
    """전처리 통합 Vector Store 테스트"""
    print("=" * 80)
    print("🔧 전처리 통합 Vector Store 테스트 시작")
    print("=" * 80)

    # 1. 테스트 데이터 로드
    print("\n1️⃣ 테스트 데이터 로드")
    try:
        with open("description_chunking_test_result.json", "r", encoding="utf-8") as f:
            test_data = json.load(f)

        patent = test_data["patents"][0]

        # 샘플 청크 생성 (HTML 태그와 특수 문자가 포함된 텍스트로 시뮬레이션)
        sample_chunks = []

        # Title chunk (HTML 태그 추가)
        if patent.get("title"):
            title_chunk = DocumentChunk(
                patent_number=patent["patent_number"],
                section="title",
                chunk_index=0,
                content=f"<b>{patent['title']}</b> with <span>HTML tags</span>",
                metadata={
                    "patent_number": patent["patent_number"],
                    "section": "title",
                    "chunk_index": 0,
                },
            )
            sample_chunks.append(title_chunk)

        # Description chunks (도면 참조 추가)
        if patent.get("description_chunks"):
            for i, chunk_text in enumerate(patent["description_chunks"][:3]):
                # 도면 참조와 참조 번호 추가
                modified_text = f"{chunk_text} See FIG. {i+1}A for details. Component (100) is shown."
                desc_chunk = DocumentChunk(
                    patent_number=patent["patent_number"],
                    section="description",
                    chunk_index=i,
                    content=modified_text,
                    metadata={
                        "patent_number": patent["patent_number"],
                        "section": "description",
                        "chunk_index": i,
                    },
                )
                sample_chunks.append(desc_chunk)

        print(f"  ✅ 테스트 청크 생성: {len(sample_chunks)}개")

        # 원본 텍스트 길이 계산
        total_original_length = sum(len(chunk.content) for chunk in sample_chunks)
        print(f"  📊 총 원본 텍스트 길이: {total_original_length:,} 문자")

    except Exception as e:
        print(f"  ❌ 테스트 데이터 로드 실패: {e}")
        return

    # 2. Embedding Manager 초기화
    print("\n2️⃣ Embedding Manager 초기화")
    try:
        embedding_manager = EmbeddingManager()
        print("  ✅ EmbeddingManager 초기화 완료")
    except Exception as e:
        print(f"  ❌ EmbeddingManager 초기화 실패: {e}")
        return

    # 3. 전처리 비활성화 Vector Store 테스트
    print("\n3️⃣ 전처리 비활성화 Vector Store 테스트")
    try:
        vector_store_no_prep = PatentVectorStore(
            embedding_manager=embedding_manager,
            collection_name="test_no_preprocessing",
            persist_directory="./data/chroma_test",
            reset_collection=True,
            enable_preprocessing=False,
        )

        # 청크 저장
        success_count, error_count = vector_store_no_prep.add_document_chunks_batch(
            sample_chunks
        )

        print(
            f"  ✅ 전처리 없이 저장 완료: {success_count}개 성공, {error_count}개 실패"
        )

        # 검색 테스트
        search_results = vector_store_no_prep.search_similar(
            query="machine learning hardware monitoring",
            n_results=2,
            include_distances=True,
        )

        print(f"  📊 검색 결과: {len(search_results.get('results', []))}개")
        for i, result in enumerate(search_results.get("results", [])[:2]):
            print(f"    {i+1}. 거리: {result.get('distance', 'N/A'):.4f}")
            print(f"       내용: {result['content'][:80]}...")

    except Exception as e:
        print(f"  ❌ 전처리 비활성화 테스트 실패: {e}")

    # 4. 전처리 활성화 Vector Store 테스트
    print("\n4️⃣ 전처리 활성화 Vector Store 테스트")
    try:
        vector_store_with_prep = PatentVectorStore(
            embedding_manager=embedding_manager,
            collection_name="test_with_preprocessing",
            persist_directory="./data/chroma_test",
            reset_collection=True,
            enable_preprocessing=True,
        )

        # 청크 저장
        success_count, error_count = vector_store_with_prep.add_document_chunks_batch(
            sample_chunks
        )

        print(
            f"  ✅ 전처리 적용하여 저장 완료: {success_count}개 성공, {error_count}개 실패"
        )

        # 검색 테스트
        search_results = vector_store_with_prep.search_similar(
            query="machine learning hardware monitoring",
            n_results=2,
            include_distances=True,
        )

        print(f"  📊 검색 결과: {len(search_results.get('results', []))}개")
        for i, result in enumerate(search_results.get("results", [])[:2]):
            print(f"    {i+1}. 거리: {result.get('distance', 'N/A'):.4f}")
            print(f"       내용: {result['content'][:80]}...")

            # 전처리 메타데이터 확인
            if result["metadata"].get("preprocessing_applied"):
                print(
                    f"       전처리 적용됨 - 압축비율: {result['metadata'].get('compression_ratio', 0):.3f}"
                )

    except Exception as e:
        print(f"  ❌ 전처리 활성화 테스트 실패: {e}")

    # 5. 성능 비교
    print("\n5️⃣ 성능 비교")
    try:
        # 통계 정보 비교
        stats_no_prep = vector_store_no_prep.get_stats()
        stats_with_prep = vector_store_with_prep.get_stats()

        print("  📊 전처리 없음:")
        print(f"    - 총 청크 수: {stats_no_prep.get('total_chunks', 0)}")

        print("  📊 전처리 적용:")
        print(f"    - 총 청크 수: {stats_with_prep.get('total_chunks', 0)}")

        # 임베딩 매니저 통계
        embedding_stats = embedding_manager.get_stats()
        print(f"  📊 총 토큰 사용량: {embedding_stats.get('total_tokens_used', 0)}")
        print(f"  📊 총 API 호출 수: {embedding_stats.get('total_requests', 0)}")

    except Exception as e:
        print(f"  ❌ 성능 비교 실패: {e}")


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    test_integrated_preprocessing()

    print("\n" + "=" * 80)
    print("🎉 전처리 통합 테스트 완료!")
    print("=" * 80)
