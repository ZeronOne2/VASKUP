"""
Google Patents 페이지에서 특허 설명을 스크래핑하는 모듈

이 모듈은 SerpAPI에서 제공하는 특허 링크로부터 상세한 특허 설명을
추출하고 정제하는 기능을 제공합니다.
"""

import re
import time
import logging
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Union, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse
import threading
from dataclasses import dataclass


# 스크래핑 오류 클래스
class ScrapingError(Exception):
    """스크래핑 관련 오류"""

    pass


@dataclass
class ScrapingResult:
    """스크래핑 결과 데이터 클래스"""

    patent_id: str
    url: str
    title: str = ""
    abstract: str = ""
    description: str = ""
    claims: List[str] = None
    success: bool = False
    error_message: str = ""
    processing_time: float = 0.0

    def __post_init__(self):
        if self.claims is None:
            self.claims = []


class RateLimiter:
    """요청 속도 제한기"""

    def __init__(self, requests_per_second: float = 1.0):
        """
        Args:
            requests_per_second: 초당 허용 요청 수
        """
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0
        self.lock = threading.Lock()

    def wait_if_needed(self):
        """필요시 대기"""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time

            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                time.sleep(wait_time)

            self.last_request_time = time.time()


class GooglePatentsDescriptionScraper:
    """Google Patents 특허 설명 스크래퍼"""

    def __init__(
        self, requests_per_second: float = 2.0, timeout: int = 30, max_retries: int = 3
    ):
        """
        Args:
            requests_per_second: 초당 요청 제한 (기본 2.0)
            timeout: 요청 타임아웃 (초)
            max_retries: 최대 재시도 횟수
        """
        self.rate_limiter = RateLimiter(requests_per_second)
        self.timeout = timeout
        self.max_retries = max_retries

        # User-Agent 헤더
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        # 로깅 설정
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _fetch_page_content(self, url: str) -> str:
        """웹 페이지 HTML 내용 가져오기"""
        self.rate_limiter.wait_if_needed()

        for attempt in range(self.max_retries):
            try:
                self.logger.debug(
                    f"페이지 요청 시도 {attempt + 1}/{self.max_retries}: {url}"
                )

                response = requests.get(
                    url,
                    headers=self.headers,
                    timeout=self.timeout,
                    allow_redirects=True,
                )
                response.raise_for_status()

                # 한국어 페이지인 경우 영어 버전으로 리다이렉트
                if (
                    "patents.google.com/patent/" in response.url
                    and "?hl=" not in response.url
                ):
                    english_url = f"{response.url}?hl=en"
                    self.logger.info(f"영어 버전으로 리다이렉트: {english_url}")
                    return self._fetch_page_content(english_url)

                return response.text

            except requests.RequestException as e:
                self.logger.warning(
                    f"요청 실패 (시도 {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt == self.max_retries - 1:
                    raise ScrapingError(f"페이지 요청 실패: {e}")
                time.sleep(2**attempt)  # 지수 백오프

    def _extract_patent_id_from_url(self, url: str) -> str:
        """URL에서 특허 ID 추출"""
        # Google Patents URL 패턴: https://patents.google.com/patent/US11734097B1/
        pattern = r"/patent/([^/\?]+)"
        match = re.search(pattern, url)
        return match.group(1) if match else "unknown"

    def _clean_text(self, text: str) -> str:
        """텍스트 정제"""
        if not text:
            return ""

        # HTML 엔티티 제거
        text = BeautifulSoup(text, "html.parser").get_text()

        # 과도한 공백 제거
        text = re.sub(r"\s+", " ", text)

        # 앞뒤 공백 제거
        text = text.strip()

        return text

    def _extract_patent_info(
        self, html_content: str, url: str
    ) -> Dict[str, Union[str, List[str]]]:
        """HTML에서 특허 정보 추출"""
        soup = BeautifulSoup(html_content, "html.parser")

        patent_info = {"title": "", "abstract": "", "description": "", "claims": []}

        try:
            # 제목 추출
            title_element = (
                soup.find("h1", {"data-auto-id": "patent-title"})
                or soup.find("meta", {"property": "og:title"})
                or soup.find("title")
            )

            if title_element:
                if title_element.name == "meta":
                    patent_info["title"] = title_element.get("content", "")
                else:
                    patent_info["title"] = title_element.get_text(strip=True)

            self.logger.debug(f"제목 추출: {patent_info['title'][:50]}...")

            # 초록(Abstract) 추출
            abstract_section = (
                soup.find("section", {"itemprop": "abstract"})
                or soup.find("div", class_=lambda x: x and "abstract" in x.lower())
                or soup.find("meta", {"name": "description"})
            )

            if abstract_section:
                if abstract_section.name == "meta":
                    patent_info["abstract"] = abstract_section.get("content", "")
                else:
                    # Google 번역 텍스트 제거
                    for span in abstract_section.find_all(
                        "span", class_="google-src-text"
                    ):
                        span.decompose()
                    patent_info["abstract"] = self._clean_text(
                        abstract_section.get_text()
                    )

            self.logger.debug(f"초록 추출: {len(patent_info['abstract'])} 문자")

            # 상세 설명(Description) 추출
            description_section = soup.find(
                "section", {"itemprop": "description"}
            ) or soup.find("div", class_=lambda x: x and "description" in x.lower())

            if description_section:
                # Google 번역 텍스트 제거
                for span in description_section.find_all(
                    "span", class_="google-src-text"
                ):
                    span.decompose()

                # 불필요한 섹션 제거 (도면 설명 등)
                for unwanted in description_section.find_all(["figure", "img"]):
                    unwanted.decompose()

                patent_info["description"] = self._clean_text(
                    description_section.get_text()
                )

            self.logger.debug(f"설명 추출: {len(patent_info['description'])} 문자")

            # 청구항(Claims) 추출
            claims_section = soup.find("section", {"itemprop": "claims"}) or soup.find(
                "div", class_=lambda x: x and "claims" in x.lower()
            )

            if claims_section:
                # Google 번역 텍스트 제거
                for span in claims_section.find_all("span", class_="google-src-text"):
                    span.decompose()

                # 개별 청구항 추출
                claims = []
                claim_elements = claims_section.find_all(
                    ["div", "p"],
                    class_=lambda x: x
                    and ("claim" in x.lower() or "patent-claim" in x.lower()),
                )

                if not claim_elements:
                    # 일반적인 청구항 패턴 찾기
                    claim_text = claims_section.get_text()
                    claim_matches = re.findall(
                        r"(\d+\.\s*[^0-9]+?)(?=\d+\.|$)", claim_text, re.DOTALL
                    )
                    claims = [
                        self._clean_text(claim) for claim in claim_matches[:10]
                    ]  # 최대 10개
                else:
                    claims = [
                        self._clean_text(elem.get_text())
                        for elem in claim_elements[:10]
                    ]

                patent_info["claims"] = claims

            self.logger.debug(f"청구항 추출: {len(patent_info['claims'])}개")

        except Exception as e:
            self.logger.error(f"정보 추출 중 오류: {e}")
            raise ScrapingError(f"특허 정보 추출 실패: {e}")

        return patent_info

    def scrape_patent_description(self, url: str) -> ScrapingResult:
        """단일 특허 설명 스크래핑"""
        start_time = time.time()
        patent_id = self._extract_patent_id_from_url(url)

        result = ScrapingResult(patent_id=patent_id, url=url)

        try:
            self.logger.info(f"특허 스크래핑 시작: {patent_id} ({url})")

            # HTML 내용 가져오기
            html_content = self._fetch_page_content(url)

            # 특허 정보 추출
            patent_info = self._extract_patent_info(html_content, url)

            # 결과 설정
            result.title = patent_info["title"]
            result.abstract = patent_info["abstract"]
            result.description = patent_info["description"]
            result.claims = patent_info["claims"]
            result.success = True

            self.logger.info(f"특허 스크래핑 완료: {patent_id}")

        except Exception as e:
            result.error_message = str(e)
            result.success = False
            self.logger.error(f"특허 스크래핑 실패 {patent_id}: {e}")

        result.processing_time = time.time() - start_time
        return result

    def batch_scrape_patents(
        self,
        urls: List[str],
        max_workers: int = 3,
        progress_callback: Optional[Callable] = None,
    ) -> List[ScrapingResult]:
        """다중 특허 병렬 스크래핑"""
        results = []

        self.logger.info(
            f"배치 스크래핑 시작: {len(urls)}개 특허, {max_workers}개 워커"
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 작업 제출
            future_to_url = {
                executor.submit(self.scrape_patent_description, url): url
                for url in urls
            }

            # 결과 수집
            completed = 0
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # 실패한 경우 빈 결과 생성
                    patent_id = self._extract_patent_id_from_url(url)
                    result = ScrapingResult(
                        patent_id=patent_id,
                        url=url,
                        success=False,
                        error_message=str(e),
                    )
                    results.append(result)

                completed += 1

                # 진행률 콜백 호출
                if progress_callback:
                    progress_callback(completed, len(urls), result)

        # 성공률 계산
        successful = sum(1 for r in results if r.success)
        success_rate = successful / len(results) * 100 if results else 0

        self.logger.info(
            f"배치 스크래핑 완료: {successful}/{len(results)} 성공 ({success_rate:.1f}%)"
        )

        return results

    def get_scraping_stats(
        self, results: List[ScrapingResult]
    ) -> Dict[str, Union[int, float]]:
        """스크래핑 통계 정보 반환"""
        if not results:
            return {}

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        total_processing_time = sum(r.processing_time for r in results)
        avg_processing_time = total_processing_time / len(results)

        total_description_length = sum(len(r.description) for r in successful)
        avg_description_length = (
            total_description_length / len(successful) if successful else 0
        )

        return {
            "total_patents": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(results) * 100,
            "total_processing_time": total_processing_time,
            "avg_processing_time": avg_processing_time,
            "avg_description_length": avg_description_length,
            "total_description_length": total_description_length,
        }


# 편의 함수들
def scrape_single_patent_description(url: str) -> ScrapingResult:
    """단일 특허 설명 스크래핑 편의 함수"""
    scraper = GooglePatentsDescriptionScraper()
    return scraper.scrape_patent_description(url)


def scrape_multiple_patent_descriptions(
    urls: List[str], max_workers: int = 3, progress_callback: Optional[Callable] = None
) -> List[ScrapingResult]:
    """다중 특허 설명 스크래핑 편의 함수"""
    scraper = GooglePatentsDescriptionScraper()
    return scraper.batch_scrape_patents(urls, max_workers, progress_callback)
