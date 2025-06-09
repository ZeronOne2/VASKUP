"""
SerpAPI 클라이언트 기본 테스트 (pytest 없이)
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.patent_search.serp_client import (
        SerpPatentClient,
        PatentSearchError,
        RateLimiter,
        PatentCache,
    )

    print("✅ SerpAPI 모듈 import 성공")
except ImportError as e:
    print(f"❌ SerpAPI 모듈 import 실패: {e}")
    sys.exit(1)


def test_patent_number_conversion():
    """특허 번호 변환 테스트"""
    print("\n🧪 특허 번호 변환 테스트...")

    try:
        client = SerpPatentClient(api_key="test_key")

        # US 특허 테스트
        us_result = client.convert_patent_number_to_id("US11734097B1")
        expected = "patent/US11734097B1/en"
        assert us_result == expected, f"Expected {expected}, got {us_result}"

        # KR 특허 테스트
        kr_result = client.convert_patent_number_to_id("KR10-2021-0123456")
        expected = "patent/KR10-2021-0123456/en"
        assert kr_result == expected, f"Expected {expected}, got {kr_result}"

        # 언어 코드 테스트
        ko_result = client.convert_patent_number_to_id("US11734097B1", "ko")
        expected = "patent/US11734097B1/ko"
        assert ko_result == expected, f"Expected {expected}, got {ko_result}"

        print("✅ 특허 번호 변환 테스트 통과")
        return True

    except Exception as e:
        print(f"❌ 특허 번호 변환 테스트 실패: {e}")
        return False


def test_rate_limiter():
    """속도 제한기 테스트"""
    print("\n🧪 속도 제한기 테스트...")

    try:
        limiter = RateLimiter(calls_per_minute=60)

        # 초기화 확인
        assert limiter.calls_per_minute == 60
        assert len(limiter.calls) == 0

        # 호출 테스트
        import time

        start_time = time.time()
        limiter.wait_if_needed()
        end_time = time.time()

        # 첫 번째 호출은 즉시 처리되어야 함
        assert end_time - start_time < 0.1
        assert len(limiter.calls) == 1

        print("✅ 속도 제한기 테스트 통과")
        return True

    except Exception as e:
        print(f"❌ 속도 제한기 테스트 실패: {e}")
        return False


def test_patent_cache():
    """특허 캐시 테스트"""
    print("\n🧪 특허 캐시 테스트...")

    try:
        cache = PatentCache(cache_dir="test_cache")

        # 테스트 데이터
        patent_id = "patent/US11734097B1/en"
        test_data = {
            "title": "Test Patent",
            "abstract": "Test abstract",
            "inventor": ["Test Inventor"],
        }

        # 캐시에 저장
        cache.set(patent_id, test_data)

        # 캐시에서 조회
        retrieved_data = cache.get(patent_id)
        assert retrieved_data == test_data

        # 존재하지 않는 데이터 조회
        non_existent = cache.get("patent/NONEXISTENT/en")
        assert non_existent is None

        print("✅ 특허 캐시 테스트 통과")

        # 테스트 캐시 정리
        import shutil

        if Path("test_cache").exists():
            shutil.rmtree("test_cache")

        return True

    except Exception as e:
        print(f"❌ 특허 캐시 테스트 실패: {e}")
        return False


def test_client_initialization():
    """클라이언트 초기화 테스트"""
    print("\n🧪 클라이언트 초기화 테스트...")

    try:
        # API 키로 초기화
        client = SerpPatentClient(api_key="test_key")
        assert client.api_key == "test_key"

        # API 키 없이 초기화 - 실패해야 함
        original_key = os.environ.get("SERPAPI_API_KEY")
        if "SERPAPI_API_KEY" in os.environ:
            del os.environ["SERPAPI_API_KEY"]

        try:
            SerpPatentClient()
            print("❌ API 키 없이 초기화가 성공했습니다 (실패해야 함)")
            return False
        except PatentSearchError:
            pass  # 예상된 오류

        # 환경 변수 복원
        if original_key:
            os.environ["SERPAPI_API_KEY"] = original_key

        print("✅ 클라이언트 초기화 테스트 통과")
        return True

    except Exception as e:
        print(f"❌ 클라이언트 초기화 테스트 실패: {e}")
        return False


def test_extract_patent_info():
    """특허 정보 추출 테스트"""
    print("\n🧪 특허 정보 추출 테스트...")

    try:
        client = SerpPatentClient(api_key="test_key")

        # 모의 API 응답 데이터
        mock_data = {
            "search_parameters": {"id": "patent/US11734097B1/en"},
            "title": "Test Patent Title",
            "abstract": "This is a test patent abstract.",
            "inventors": [{"name": "John Doe"}, {"name": "Jane Smith"}],
            "assignees": [{"name": "Test Company Inc."}],
            "publication_date": "2023-01-01",
            "application_date": "2022-01-01",
            "claims": ["First claim of the patent", "Second claim of the patent"],
            "classifications": [
                {"class": "G06F", "description": "Electric digital data processing"}
            ],
            "pdf_link": "https://example.com/patent.pdf",
        }

        # 정보 추출
        extracted = client._extract_patent_info(mock_data)

        # 결과 검증
        assert extracted["title"] == "Test Patent Title"
        assert extracted["abstract"] == "This is a test patent abstract."
        assert "John Doe" in extracted["inventor"]
        assert "Jane Smith" in extracted["inventor"]
        assert "Test Company Inc." in extracted["assignee"]
        assert extracted["publication_date"] == "2023-01-01"
        assert extracted["application_date"] == "2022-01-01"
        assert len(extracted["claims"]) == 2
        assert extracted["description_link"] == "https://example.com/patent.pdf"

        print("✅ 특허 정보 추출 테스트 통과")
        return True

    except Exception as e:
        print(f"❌ 특허 정보 추출 테스트 실패: {e}")
        return False


def main():
    """메인 테스트 실행 함수"""
    print("🚀 SerpAPI 클라이언트 기본 테스트 시작")
    print("=" * 50)

    tests = [
        test_patent_number_conversion,
        test_rate_limiter,
        test_patent_cache,
        test_client_initialization,
        test_extract_patent_info,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ 테스트 실행 중 예외 발생: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"🎯 테스트 결과: {passed}개 통과, {failed}개 실패")

    if failed == 0:
        print("🎉 모든 테스트가 통과했습니다!")
        return True
    else:
        print("⚠️ 일부 테스트가 실패했습니다.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
