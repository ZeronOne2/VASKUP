#!/usr/bin/env python3
"""성능 모니터링 테스트"""

import os
import sys
import time
import logging
from typing import List

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.patent_search.patent_parser import Patent, PatentParser, DocumentChunk
from src.vector_store.patent_vector_store import PatentVectorStore
from src.vector_store.performance_optimizer import (
    PerformanceMonitor,
    performance_monitor,
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def test_performance_monitoring():
    """성능 모니터링 테스트"""
    print("=" * 50)
    print("성능 모니터링 테스트")
    print("=" * 50)

    # 1. 독립적인 성능 모니터 테스트
    print("\n1. 독립적인 성능 모니터 테스트")
    monitor = PerformanceMonitor()

    # 샘플 작업 기록
    monitor.record_operation("sample_operation", 0.5, True, {"data_size": 100})
    monitor.record_operation("sample_operation", 0.7, True, {"data_size": 150})
    monitor.record_operation("slow_operation", 2.5, True, {"data_size": 200})
    monitor.record_operation("failed_operation", 1.0, False, {"error": "timeout"})

    # 통계 출력
    stats = monitor.get_overall_stats()
    print(f"전체 작업 수: {stats['total_operations']}")
    print(f"성공률: {stats['success_rate']:.1%}")
    print(f"평균 응답 시간: {stats['avg_duration']:.3f}초")

    # 병목 지점 식별
    bottlenecks = monitor.identify_bottlenecks()
    print(f"병목 지점: {bottlenecks}")

    # 권장사항 생성
    recommendations = monitor.generate_recommendations()
    print("\n권장사항:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")

    # 2. 데코레이터 테스트
    print("\n2. 성능 데코레이터 테스트")

    @performance_monitor(monitor, "test_function")
    def slow_function(duration: float):
        time.sleep(duration)
        return f"완료: {duration}초 대기"

    result = slow_function(0.1)
    print(f"함수 결과: {result}")

    # 데코레이터 사용 후 통계 확인
    test_stats = monitor.get_operation_stats("test_function")
    print(f"test_function 통계: {test_stats}")

    # 3. Vector Store와 통합 테스트
    print("\n3. Vector Store 성능 모니터링 테스트")

    # 테스트 데이터 로드 또는 생성
    test_file = "test_data.json"
    if not os.path.exists(test_file):
        print(f"⚠️ 테스트 데이터 파일이 없어서 더미 데이터를 생성합니다")

        # 더미 특허 데이터 생성
        from src.patent_search.patent_parser import Patent

        patent = Patent(
            patent_number="TEST123456",
            patent_id="patent/TEST123456/en",
            title="Test Patent for Performance Monitoring",
            description="This is a test patent description for performance monitoring. "
            * 50,
            claims=["1. A test claim for monitoring performance. " * 20],
            abstract="Test abstract for performance analysis.",
            publication_date="2023-01-01",
            application_date="2022-01-01",
            inventor=["Test Inventor"],
            assignee=["Test Company"],
            google_patents_url="https://test.url",
        )
    else:
        parser = PatentParser()
        patents = parser.load_patents_from_json(test_file)

        if not patents:
            print("❌ 특허 데이터 로드 실패")
            return

        patent = patents[0]

    print(f"테스트 특허: {patent.patent_number}")

    # 청킹
    from src.patent_search.patent_parser import PatentChunker

    chunker = PatentChunker()
    chunks = chunker.chunk_patent(patent)
    print(f"청크 수: {len(chunks)}")

    # Vector Store (성능 모니터링 활성화)
    vector_store = PatentVectorStore(
        persist_directory="./test_chroma_performance",
        collection_name="performance_test",
        reset_collection=True,
        enable_performance_monitoring=True,
        enable_preprocessing=True,
    )

    # 배치 추가 성능 테스트
    print("\n배치 추가 성능 테스트...")
    success_count, error_count = vector_store.add_document_chunks_batch(chunks)
    print(f"추가 결과: {success_count}개 성공, {error_count}개 실패")

    # 검색 성능 테스트
    print("\n검색 성능 테스트...")
    search_results = vector_store.search_similar(
        "artificial intelligence machine learning", n_results=5
    )
    print(f"검색 결과: {len(search_results.get('documents', []))}개")

    # Vector Store 성능 통계 출력
    if vector_store.performance_monitor:
        vs_stats = vector_store.performance_monitor.get_overall_stats()
        print(f"\nVector Store 성능 통계:")
        print(f"  전체 작업: {vs_stats['total_operations']}")
        print(f"  성공률: {vs_stats['success_rate']:.1%}")
        print(f"  평균 응답 시간: {vs_stats['avg_duration']:.3f}초")

        print("\n작업별 성능:")
        for op_name, op_stats in vs_stats["operations_by_type"].items():
            print(
                f"  {op_name}: {op_stats['count']}회, 평균 {op_stats['avg_duration']:.3f}초"
            )

        # 권장사항
        vs_recommendations = vector_store.performance_monitor.generate_recommendations()
        print("\nVector Store 최적화 권장사항:")
        for i, rec in enumerate(vs_recommendations, 1):
            print(f"{i}. {rec}")

    # 정리
    vector_store.close()

    print("\n✅ 성능 모니터링 테스트 완료")


if __name__ == "__main__":
    test_performance_monitoring()
