#!/usr/bin/env python3
"""텍스트 전처리 테스트 스크립트"""

import json
import logging
import os
import sys

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.vector_store.text_preprocessor import (
    PatentTextPreprocessor,
    PatentChunkPreprocessor,
)
from src.patent_search.patent_parser import DocumentChunk


def test_text_preprocessing():
    """텍스트 전처리 기능 테스트"""
    print("=" * 80)
    print("🧹 텍스트 전처리 테스트 시작")
    print("=" * 80)

    # 1. 전처리기 초기화
    print("\n1️⃣ 전처리기 초기화")
    try:
        preprocessor = PatentTextPreprocessor(
            remove_html=True,
            normalize_whitespace=True,
            remove_special_chars=False,
            min_length=5,
            max_length=5000,
        )
        print("  ✅ PatentTextPreprocessor 초기화 완료")
    except Exception as e:
        print(f"  ❌ 전처리기 초기화 실패: {e}")
        return

    # 2. 샘플 텍스트 전처리 테스트
    print("\n2️⃣ 샘플 텍스트 전처리 테스트")

    sample_texts = [
        # HTML 태그가 포함된 텍스트
        "This is a <b>patent</b> description with <span>HTML tags</span> and multiple    spaces.",
        # 도면 참조가 포함된 텍스트
        "The system shown in FIG. 1A includes components (10) and (20). See also FIG. 2B for details.",
        # 청구항 형태의 텍스트
        "1. A method comprising: identifying, by a monitoring system (100), hardware components...",
        # 특수 문자가 많은 텍스트
        "The storage system™ uses advanced algorithms® for data processing... with efficiency!!!",
    ]

    for i, text in enumerate(sample_texts):
        print(f"\n  테스트 {i+1}: {text[:50]}...")
        try:
            result = preprocessor.preprocess_text(text)
            print(f"    원본 길이: {result.statistics['original_length']}")
            print(f"    처리 후 길이: {result.statistics['processed_length']}")
            print(f"    압축 비율: {result.statistics['compression_ratio']:.3f}")
            print(f"    제거된 요소 수: {result.statistics['removed_count']}")
            print(f"    처리 시간: {result.processing_time:.4f}초")
            print(f"    결과: {result.processed_text[:100]}...")

            if result.removed_elements:
                print(f"    제거된 요소들: {result.removed_elements[:3]}...")

        except Exception as e:
            print(f"    ❌ 전처리 실패: {e}")


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    test_text_preprocessing()

    print("\n" + "=" * 80)
    print("🎉 전처리 테스트 완료!")
    print("=" * 80)
