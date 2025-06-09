"""
SerpAPI 클라이언트 단위 테스트
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from src.patent_search.serp_client import (
    SerpPatentClient,
    PatentSearchError,
    RateLimiter,
    PatentCache,
)


class TestPatentNumberConversion:
    """특허 번호 변환 테스트"""

    def test_convert_us_patent(self):
        """US 특허 번호 변환 테스트"""
        client = SerpPatentClient(api_key="test_key")

        result = client.convert_patent_number_to_id("US11734097B1")
        assert result == "patent/US11734097B1/en"

        result = client.convert_patent_number_to_id("US11734097B1", "ko")
        assert result == "patent/US11734097B1/ko"

    def test_convert_kr_patent(self):
        """KR 특허 번호 변환 테스트"""
        client = SerpPatentClient(api_key="test_key")

        result = client.convert_patent_number_to_id("KR10-2021-0123456")
        assert result == "patent/KR10-2021-0123456/en"

    def test_already_converted_patent(self):
        """이미 변환된 특허 ID 테스트"""
        client = SerpPatentClient(api_key="test_key")

        patent_id = "patent/US11734097B1/en"
        result = client.convert_patent_number_to_id(patent_id)
        assert result == patent_id


class TestRateLimiter:
    """속도 제한 테스트"""

    def test_rate_limiter_initialization(self):
        """속도 제한기 초기화 테스트"""
        limiter = RateLimiter(calls_per_minute=60)
        assert limiter.calls_per_minute == 60
        assert len(limiter.calls) == 0

    def test_rate_limiter_under_limit(self):
        """제한 이하 호출 테스트"""
        limiter = RateLimiter(calls_per_minute=100)

        # 제한 이하로 호출 시 즉시 처리되어야 함
        import time

        start_time = time.time()
        limiter.wait_if_needed()
        end_time = time.time()

        # 거의 즉시 처리되어야 함 (0.1초 이내)
        assert end_time - start_time < 0.1


class TestPatentCache:
    """특허 캐시 테스트"""

    def setup_method(self):
        """테스트 메서드 설정"""
        self.test_cache_dir = Path("test_cache")
        self.cache = PatentCache(cache_dir=str(self.test_cache_dir))

    def teardown_method(self):
        """테스트 메서드 정리"""
        # 테스트 캐시 파일 정리
        if self.test_cache_dir.exists():
            import shutil

            shutil.rmtree(self.test_cache_dir)

    def test_cache_get_set(self):
        """캐시 저장/조회 테스트"""
        patent_id = "patent/US11734097B1/en"
        test_data = {"title": "Test Patent", "abstract": "Test abstract"}

        # 캐시에 저장
        self.cache.set(patent_id, test_data)

        # 캐시에서 조회
        retrieved_data = self.cache.get(patent_id)
        assert retrieved_data == test_data

    def test_cache_miss(self):
        """캐시 미스 테스트"""
        non_existent_id = "patent/NONEXISTENT/en"
        result = self.cache.get(non_existent_id)
        assert result is None


class TestSerpPatentClient:
    """SerpAPI 클라이언트 테스트"""

    def test_client_initialization_with_api_key(self):
        """API 키로 클라이언트 초기화 테스트"""
        client = SerpPatentClient(api_key="test_key")
        assert client.api_key == "test_key"

    def test_client_initialization_without_api_key(self):
        """API 키 없이 클라이언트 초기화 테스트 - 실패해야 함"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(PatentSearchError):
                SerpPatentClient()

    @patch("src.patent_search.serp_client.requests.get")
    def test_successful_patent_search(self, mock_get):
        """성공적인 특허 검색 테스트"""
        # Mock API 응답
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "title": "Test Patent Title",
            "abstract": "Test patent abstract",
            "inventors": [{"name": "John Doe"}],
            "assignees": [{"name": "Test Company"}],
            "publication_date": "2023-01-01",
            "application_date": "2022-01-01",
        }
        mock_get.return_value = mock_response

        client = SerpPatentClient(api_key="test_key")
        result = client.search_patent("US11734097B1")

        # 결과 검증
        assert result["title"] == "Test Patent Title"
        assert result["abstract"] == "Test patent abstract"
        assert "John Doe" in result["inventor"]
        assert "Test Company" in result["assignee"]

    @patch("src.patent_search.serp_client.requests.get")
    def test_api_error_handling(self, mock_get):
        """API 오류 처리 테스트"""
        # Mock API 에러 응답
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"error": "Invalid API key"}
        mock_get.return_value = mock_response

        client = SerpPatentClient(api_key="test_key")

        with pytest.raises(PatentSearchError):
            client.search_patent("US11734097B1")


def run_basic_tests():
    """기본 테스트 실행 함수 (pytest 없이)"""
    print("🧪 SerpAPI 클라이언트 기본 테스트 실행")

    # 특허 번호 변환 테스트
    print("\n1. 특허 번호 변환 테스트...")
    try:
        client = SerpPatentClient(api_key="test_key")

        us_result = client.convert_patent_number_to_id("US11734097B1")
        kr_result = client.convert_patent_number_to_id("KR10-2021-0123456")

        assert us_result == "patent/US11734097B1/en"
        assert kr_result == "patent/KR10-2021-0123456/en"

        print("✅ 특허 번호 변환 테스트 통과")
    except Exception as e:
        print(f"❌ 특허 번호 변환 테스트 실패: {e}")

    # 캐시 테스트
    print("\n2. 캐시 시스템 테스트...")
    try:
        cache = PatentCache(cache_dir="test_cache")

        test_data = {"title": "Test Patent", "abstract": "Test abstract"}
        patent_id = "patent/TEST123/en"

        cache.set(patent_id, test_data)
        retrieved = cache.get(patent_id)

        assert retrieved == test_data
        print("✅ 캐시 시스템 테스트 통과")

        # 테스트 캐시 정리
        import shutil

        if Path("test_cache").exists():
            shutil.rmtree("test_cache")

    except Exception as e:
        print(f"❌ 캐시 시스템 테스트 실패: {e}")

    print("\n🎉 기본 테스트 완료!")


if __name__ == "__main__":
    run_basic_tests()
