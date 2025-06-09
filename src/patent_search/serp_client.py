"""
SerpAPI Google Patents API 클라이언트 모듈

이 모듈은 SerpAPI의 Google Patents 엔진을 사용하여 특허 정보를 검색하고
추출하는 기능을 제공합니다.
"""

import os
import re
import time
import json
import hashlib
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
import requests
from pathlib import Path


class PatentSearchError(Exception):
    """특허 검색 관련 커스텀 예외 클래스"""

    pass


class RateLimiter:
    """API 호출 속도 제한을 관리하는 클래스"""

    def __init__(self, calls_per_minute: int = 100):
        self.calls_per_minute = calls_per_minute
        self.calls = []

    def wait_if_needed(self):
        """필요한 경우 대기하여 속도 제한을 준수"""
        now = datetime.now()
        # 1분 이내의 호출만 유지
        self.calls = [call for call in self.calls if now - call < timedelta(minutes=1)]

        if len(self.calls) >= self.calls_per_minute:
            # 가장 오래된 호출로부터 1분 후까지 대기
            wait_until = self.calls[0] + timedelta(minutes=1)
            wait_seconds = (wait_until - now).total_seconds()
            if wait_seconds > 0:
                time.sleep(wait_seconds)

        self.calls.append(now)


class PatentCache:
    """특허 검색 결과를 캐싱하는 클래스"""

    def __init__(self, cache_dir: str = "src/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "patent_cache.json"
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        """캐시 파일에서 데이터를 로드"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}

    def _save_cache(self):
        """캐시 데이터를 파일에 저장"""
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _get_cache_key(self, patent_id: str) -> str:
        """특허 ID에 대한 캐시 키 생성"""
        return hashlib.md5(patent_id.encode()).hexdigest()

    def get(self, patent_id: str) -> Optional[Dict[str, Any]]:
        """캐시에서 특허 정보 조회"""
        key = self._get_cache_key(patent_id)
        if key in self.cache:
            # 캐시 만료 확인 (7일)
            cached_time = datetime.fromisoformat(self.cache[key]["cached_at"])
            if datetime.now() - cached_time < timedelta(days=7):
                return self.cache[key]["data"]
            else:
                # 만료된 캐시 삭제
                del self.cache[key]
                self._save_cache()
        return None

    def set(self, patent_id: str, data: Dict[str, Any]):
        """특허 정보를 캐시에 저장"""
        key = self._get_cache_key(patent_id)
        self.cache[key] = {"data": data, "cached_at": datetime.now().isoformat()}
        self._save_cache()


class SerpPatentClient:
    """SerpAPI Google Patents 클라이언트"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY")
        if not self.api_key:
            raise PatentSearchError("SERPAPI_API_KEY가 설정되지 않았습니다.")

        self.base_url = "https://serpapi.com/search"
        self.rate_limiter = RateLimiter(calls_per_minute=100)
        self.cache = PatentCache()

    def convert_patent_number_to_id(
        self, patent_number: str, country_code: str = "en"
    ) -> str:
        """
        특허 번호를 SerpAPI 형식의 ID로 변환

        Args:
            patent_number: 특허 번호 (예: US11734097B1, KR10-2021-0123456)
            country_code: 언어 코드 (기본값: en)

        Returns:
            변환된 특허 ID (예: patent/US11734097B1/en)
        """
        # 특허 번호 정규화 (공백 제거만, 대소문자는 유지)
        patent_number = patent_number.strip()

        # 이미 올바른 형식인지 확인
        if patent_number.lower().startswith("patent/"):
            return patent_number

        # SerpAPI 형식으로 변환
        return f"patent/{patent_number}/{country_code}"

    def search_patent(
        self, patent_number: str, country_code: str = "en"
    ) -> Dict[str, Any]:
        """
        특허 번호로 특허 정보를 검색

        Args:
            patent_number: 검색할 특허 번호
            country_code: 언어 코드

        Returns:
            특허 정보 딕셔너리
        """
        # 캐시 확인
        patent_id = self.convert_patent_number_to_id(patent_number, country_code)
        cached_result = self.cache.get(patent_id)
        if cached_result:
            return cached_result

        # API 호출 속도 제한
        self.rate_limiter.wait_if_needed()

        # API 파라미터 설정 (SerpAPI는 patent_id 파라미터 사용)
        params = {
            "engine": "google_patents_details",
            "patent_id": patent_id,
            "api_key": self.api_key,
        }

        try:
            # API 호출
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # 에러 확인
            if "error" in data:
                raise PatentSearchError(f"SerpAPI 오류: {data['error']}")

            # 결과 추출 및 정규화
            patent_info = self._extract_patent_info(data)

            # 캐시에 저장
            self.cache.set(patent_id, patent_info)

            return patent_info

        except requests.RequestException as e:
            raise PatentSearchError(f"API 요청 실패: {str(e)}")
        except json.JSONDecodeError as e:
            raise PatentSearchError(f"응답 파싱 실패: {str(e)}")

    def _extract_patent_info(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        SerpAPI 응답에서 필요한 특허 정보를 추출

        Args:
            raw_data: SerpAPI 원본 응답 데이터

        Returns:
            정규화된 특허 정보
        """
        # 기본 정보 추출
        patent_info = {
            "patent_id": raw_data.get("search_parameters", {}).get("id", ""),
            "title": raw_data.get("title", "제목 없음"),
            "abstract": raw_data.get("abstract", ""),
            "inventor": [],
            "assignee": [],
            "publication_date": raw_data.get("publication_date", ""),
            "application_date": raw_data.get("filing_date", ""),
            "claims": [],
            "description_link": raw_data.get("description_link", ""),
            "google_patents_url": raw_data.get("search_metadata", {}).get(
                "google_patents_details_url", ""
            ),
            "status": raw_data.get("status", ""),
            "classifications": [],
            "citations": {"cited_by": [], "cites": []},
            "search_timestamp": datetime.now().isoformat(),
            "raw_data": raw_data,  # 디버깅 및 추가 분석용
        }

        # 발명자 정보 추출
        inventors = raw_data.get("inventors", [])
        if isinstance(inventors, list):
            patent_info["inventor"] = []
            for inv in inventors:
                if isinstance(inv, dict):
                    patent_info["inventor"].append(inv.get("name", ""))
                elif isinstance(inv, str):
                    patent_info["inventor"].append(inv)

        # 양수인 정보 추출
        assignees = raw_data.get("assignees", [])
        if isinstance(assignees, list):
            patent_info["assignee"] = []
            for ass in assignees:
                if isinstance(ass, dict):
                    patent_info["assignee"].append(ass.get("name", ""))
                elif isinstance(ass, str):
                    patent_info["assignee"].append(ass)

        # 청구항 추출
        claims = raw_data.get("claims", [])
        if isinstance(claims, list):
            patent_info["claims"] = claims[:5]  # 처음 5개 청구항만

        # 분류 정보 추출
        classifications = raw_data.get("classifications", [])
        if isinstance(classifications, list):
            patent_info["classifications"] = classifications

        # 인용 정보 추출
        if "citations" in raw_data:
            citations = raw_data["citations"]
            if "cited_by" in citations:
                patent_info["citations"]["cited_by"] = citations["cited_by"][:10]
            if "cites" in citations:
                patent_info["citations"]["cites"] = citations["cites"][:10]

        return patent_info

    def search_patents(
        self, query: str, num_results: int = 10, country_code: str = "en"
    ) -> Dict[str, Any]:
        """
        키워드로 특허를 검색

        Args:
            query: 검색 키워드
            num_results: 반환할 결과 수
            country_code: 언어 코드

        Returns:
            검색 결과 딕셔너리
        """
        # API 호출 속도 제한
        self.rate_limiter.wait_if_needed()

        # API 파라미터 설정 (Google Patents 검색 엔진 사용)
        params = {
            "engine": "google_patents",
            "q": query,
            "num": num_results,
            "api_key": self.api_key,
        }

        try:
            # API 호출
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # 에러 확인
            if "error" in data:
                raise PatentSearchError(f"SerpAPI 오류: {data['error']}")

            return data

        except requests.RequestException as e:
            raise PatentSearchError(f"API 요청 실패: {str(e)}")
        except json.JSONDecodeError as e:
            raise PatentSearchError(f"응답 파싱 실패: {str(e)}")

    def batch_search_patents(
        self,
        patent_numbers: List[str],
        country_code: str = "en",
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        여러 특허 번호를 배치로 검색

        Args:
            patent_numbers: 검색할 특허 번호 목록
            country_code: 언어 코드
            progress_callback: 진행률 콜백 함수

        Returns:
            특허 번호별 검색 결과 딕셔너리
        """
        results = {}
        errors = {}

        total = len(patent_numbers)
        for i, patent_number in enumerate(patent_numbers):
            try:
                results[patent_number] = self.search_patent(patent_number, country_code)

                if progress_callback:
                    progress_callback(i + 1, total, patent_number, success=True)

            except PatentSearchError as e:
                errors[patent_number] = str(e)

                if progress_callback:
                    progress_callback(
                        i + 1, total, patent_number, success=False, error=str(e)
                    )

        return {
            "results": results,
            "errors": errors,
            "summary": {
                "total": total,
                "success": len(results),
                "failed": len(errors),
                "success_rate": len(results) / total * 100 if total > 0 else 0,
            },
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 정보 반환"""
        cache_size = len(self.cache.cache)
        cache_file_size = 0
        if self.cache.cache_file.exists():
            cache_file_size = self.cache.cache_file.stat().st_size

        return {
            "cached_patents": cache_size,
            "cache_file_size_mb": cache_file_size / (1024 * 1024),
            "cache_location": str(self.cache.cache_file),
        }


# 편의 함수들
def search_single_patent(
    patent_number: str, api_key: Optional[str] = None
) -> Dict[str, Any]:
    """단일 특허 검색 편의 함수"""
    client = SerpPatentClient(api_key)
    return client.search_patent(patent_number)


def search_multiple_patents(
    patent_numbers: List[str],
    api_key: Optional[str] = None,
    progress_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """다중 특허 검색 편의 함수"""
    client = SerpPatentClient(api_key)
    return client.batch_search_patents(
        patent_numbers, progress_callback=progress_callback
    )
