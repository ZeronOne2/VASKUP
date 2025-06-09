#!/usr/bin/env python3
"""Vector Store 테스트 스크립트"""

import json
import logging
import os
import sys
from typing import List

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.vector_store.embedding_manager import EmbeddingManager
from src.vector_store.patent_vector_store import PatentVectorStore
from src.patent_search.patent_parser import DocumentChunk


def test_vector_store():
    """Vector Store 통합 테스트"""
    print("=" * 80)
    print("🚀 Vector Store 통합 테스트 시작")
    print("=" * 80)

    # 1. 테스트 데이터 로드
    print("\n1️⃣ 테스트 데이터 로드")
    try:
        # 이전에 생성한 청킹 결과 로드
        with open("description_chunking_test_result.json", "r", encoding="utf-8") as f:
            test_data = json.load(f)

        # 실제 데이터 구조에 맞게 수정
        patent = test_data["patents"][0]  # patents 배열의 첫 번째 특허

        # 각 섹션의 chunks를 모두 수집하여 DocumentChunk 객체로 변환
        chunks_data = []

        # Title chunk 추가 (단일 문자열)
        if patent.get("title"):
            title_chunk = DocumentChunk(
                patent_number=patent["patent_number"],
                section="title",
                chunk_index=0,
                content=patent["title"],
                metadata={
                    "patent_number": patent["patent_number"],
                    "section": "title",
                    "chunk_index": 0,
                },
            )
            chunks_data.append(title_chunk)

        # Abstract chunk 추가 (단일 문자열)
        if patent.get("abstract"):
            abstract_chunk = DocumentChunk(
                patent_number=patent["patent_number"],
                section="abstract",
                chunk_index=0,
                content=patent["abstract"],
                metadata={
                    "patent_number": patent["patent_number"],
                    "section": "abstract",
                    "chunk_index": 0,
                },
            )
            chunks_data.append(abstract_chunk)

        # Claims chunks 추가 (문자열 배열)
        if patent.get("claims"):
            for i, claim in enumerate(patent["claims"]):
                claim_chunk = DocumentChunk(
                    patent_number=patent["patent_number"],
                    section="claims",
                    chunk_index=i,
                    content=claim,
                    metadata={
                        "patent_number": patent["patent_number"],
                        "section": "claims",
                        "chunk_index": i,
                    },
                )
                chunks_data.append(claim_chunk)

        # Description chunks 추가 (문자열 배열)
        if patent.get("description_chunks"):
            for i, chunk_text in enumerate(patent["description_chunks"]):
                desc_chunk = DocumentChunk(
                    patent_number=patent["patent_number"],
                    section="description",
                    chunk_index=i,
                    content=chunk_text,
                    metadata={
                        "patent_number": patent["patent_number"],
                        "section": "description",
                        "chunk_index": i,
                    },
                )
                chunks_data.append(desc_chunk)

        print(f"  ✅ 특허 로드: {patent['patent_number']}")
        print(f"  ✅ 총 청크 수: {len(chunks_data)}개")
        print(f"     - Title: {1 if patent.get('title') else 0}")
        print(f"     - Abstract: {1 if patent.get('abstract') else 0}")
        print(f"     - Claims: {len(patent.get('claims', []))}")
        print(f"     - Description: {len(patent.get('description_chunks', []))}")

    except Exception as e:
        print(f"  ❌ 테스트 데이터 로드 실패: {e}")
        return

    # 2. Embedding Manager 초기화
    print("\n2️⃣ Embedding Manager 초기화")
    try:
        embedding_manager = EmbeddingManager()
        print("  ✅ EmbeddingManager 초기화 완료")
        print(f"  📊 모델: {embedding_manager.model}")
        print(f"  📊 차원: {embedding_manager.dimensions}")
    except Exception as e:
        print(f"  ❌ EmbeddingManager 초기화 실패: {e}")
        return

    # 3. Vector Store 초기화
    print("\n3️⃣ Vector Store 초기화")
    try:
        vector_store = PatentVectorStore(
            embedding_manager=embedding_manager,
            collection_name="test_patents",
            persist_directory="./data/chroma_test",
            reset_collection=True,  # 테스트를 위해 기존 데이터 리셋
        )
        print("  ✅ PatentVectorStore 초기화 완료")
        print(f"  📊 컬렉션: {vector_store.collection_name}")
        print(f"  📊 저장 경로: {vector_store.persist_directory}")
    except Exception as e:
        print(f"  ❌ PatentVectorStore 초기화 실패: {e}")
        return

    # 4. 임베딩 생성 테스트 (샘플 5개)
    print("\n4️⃣ 임베딩 생성 테스트")
    try:
        sample_chunks = chunks_data[:5]  # 처음 5개 청크만 테스트

        for i, chunk in enumerate(sample_chunks):
            print(f"  테스트 {i+1}: {chunk.metadata['section']} 섹션")
            print(f"    내용 길이: {len(chunk.content)} 문자")
            print(f"    청크 ID: {chunk.chunk_id}")

        # 임베딩 생성
        embedding_results = embedding_manager.create_embeddings_batch(
            [chunk.content for chunk in sample_chunks]
        )
        embeddings = [result.embedding for result in embedding_results]

        print(f"  ✅ 임베딩 생성 완료: {len(embeddings)}개")
        print(f"  📊 임베딩 차원: {len(embeddings[0])}")
        print(f"  📊 토큰 사용량: {embedding_manager.total_tokens_used}")

    except Exception as e:
        print(f"  ❌ 임베딩 생성 실패: {e}")
        return

    # 5. Vector Store에 청크 저장
    print("\n5️⃣ Vector Store에 청크 저장")
    try:
        # 모든 청크를 배치로 저장
        success_count, error_count = vector_store.add_document_chunks_batch(
            chunks_data[:10]
        )  # 처음 10개만 테스트

        print(f"  ✅ 청크 저장 완료: {success_count + error_count}개")
        print(f"  📊 성공: {success_count}")
        print(f"  📊 실패: {error_count}")

    except Exception as e:
        print(f"  ❌ 청크 저장 실패: {e}")
        return

    # 6. 유사도 검색 테스트
    print("\n6️⃣ 유사도 검색 테스트")
    try:
        test_queries = [
            "machine learning storage system",
            "hardware component monitoring",
            "data storage and retrieval",
            "flash memory technology",
            "storage cluster architecture",
        ]

        for i, query in enumerate(test_queries):
            print(f"\n  검색 {i+1}: '{query}'")

            search_results = vector_store.search_similar(
                query=query, n_results=3, include_distances=True
            )
            results = search_results.get("results", [])

            if results:
                print(f"    찾은 결과: {len(results)}개")
                for j, result in enumerate(results):
                    distance = result.get("distance", "N/A")
                    content_preview = (
                        result["content"][:100] + "..."
                        if len(result["content"]) > 100
                        else result["content"]
                    )
                    section = result["metadata"].get("section", "unknown")
                    print(
                        f"      {j+1}. 거리: {distance:.4f} | {section} | {content_preview}"
                    )
            else:
                print("    검색 결과 없음")

    except Exception as e:
        print(f"  ❌ 검색 테스트 실패: {e}")
        return

    # 7. 통계 정보 출력
    print("\n7️⃣ 통계 정보")
    try:
        stats = vector_store.get_stats()
        print(f"  📊 총 문서 수: {stats.get('total_chunks', 0)}")
        print(f"  📊 컬렉션 정보: {stats.get('collection_info', {})}")

        embedding_stats = embedding_manager.get_stats()
        print(f"  📊 총 토큰 사용량: {embedding_stats.get('total_tokens', 0)}")
        print(f"  📊 총 API 호출 수: {embedding_stats.get('total_calls', 0)}")
        if embedding_stats.get("total_calls", 0) > 0:
            print(
                f"  📊 평균 토큰/호출: {embedding_stats.get('total_tokens', 0) / embedding_stats.get('total_calls', 1):.1f}"
            )

    except Exception as e:
        print(f"  ❌ 통계 정보 조회 실패: {e}")

    print("\n" + "=" * 80)
    print("🎉 Vector Store 통합 테스트 완료!")
    print("=" * 80)


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    test_vector_store()
