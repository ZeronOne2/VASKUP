"""
특허 데이터 처리 및 청킹 모듈

이 모듈은 SerpAPI 검색 결과와 웹 스크래핑 결과를 처리하여
구조화된 Patent 객체로 변환하고, 벡터스토어 저장을 위한
텍스트 청킹을 수행합니다.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib
import re

# LangChain 청킹 도구
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 로거 설정
logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """벡터스토어 저장용 문서 청크"""

    patent_number: str
    section: str  # "title", "abstract", "claims", "description"
    chunk_index: int  # description의 경우 청크 순서
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        """청크의 고유 식별자 생성"""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        return f"{self.patent_number}_{self.section}_{self.chunk_index}_{content_hash}"


@dataclass
class Patent:
    """특허 정보를 구조화하는 메인 클래스 (PRD 정의 기반)"""

    # 필수 필드
    patent_number: str  # 원본 특허번호 (예: US11734097B1)
    patent_id: str  # SERP API 형식 (예: patent/US11734097B1/en)

    # 기본 정보
    title: str = ""
    abstract: str = ""
    claims: List[str] = field(default_factory=list)
    description: str = ""  # 웹 스크래핑으로 수집된 영어 번역본
    description_link: str = ""  # SERP API에서 제공되는 description 링크

    # 날짜 정보
    filing_date: str = ""
    publication_date: str = ""
    application_date: str = ""  # 기존 코드 호환성

    # 상태 및 메타데이터
    status: str = ""
    inventor: List[str] = field(default_factory=list)
    assignee: List[str] = field(default_factory=list)
    classifications: List[Dict] = field(default_factory=list)
    citations: Dict = field(default_factory=dict)

    # 추가 메타데이터
    serp_metadata: Dict[str, Any] = field(default_factory=dict)
    search_timestamp: str = ""
    google_patents_url: str = ""

    # 웹 스크래핑 결과
    scraped_title: str = ""
    scraped_abstract: str = ""
    scraped_description: str = ""
    scraped_claims: List[str] = field(default_factory=list)
    scraping_success: bool = False

    # 청킹된 데이터
    description_chunks: List[str] = field(default_factory=list)

    def __post_init__(self):
        """초기화 후 데이터 정제"""
        # patent_id 형식 보장
        if not self.patent_id and self.patent_number:
            self.patent_id = format_patent_id(self.patent_number)

        # 날짜 필드 통합 (기존 코드 호환성)
        if not self.filing_date and self.application_date:
            self.filing_date = self.application_date
        elif not self.application_date and self.filing_date:
            self.application_date = self.filing_date

    @property
    def is_complete(self) -> bool:
        """특허 데이터가 완전한지 확인"""
        return bool(
            self.patent_number
            and self.title
            and self.abstract
            and (self.description or self.scraped_description)
        )

    @property
    def effective_description(self) -> str:
        """스크래핑된 설명 또는 기본 설명 반환"""
        return (
            self.scraped_description if self.scraped_description else self.description
        )

    @property
    def effective_claims(self) -> List[str]:
        """스크래핑된 청구항 또는 기본 청구항 반환"""
        return self.scraped_claims if self.scraped_claims else self.claims

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "patent_number": self.patent_number,
            "patent_id": self.patent_id,
            "title": self.title,
            "abstract": self.abstract,
            "claims": self.claims,
            "description": self.description,
            "description_link": self.description_link,
            "filing_date": self.filing_date,
            "publication_date": self.publication_date,
            "application_date": self.application_date,
            "status": self.status,
            "inventor": self.inventor,
            "assignee": self.assignee,
            "classifications": self.classifications,
            "citations": self.citations,
            "serp_metadata": self.serp_metadata,
            "search_timestamp": self.search_timestamp,
            "google_patents_url": self.google_patents_url,
            "scraped_title": self.scraped_title,
            "scraped_abstract": self.scraped_abstract,
            "scraped_description": self.scraped_description,
            "scraped_claims": self.scraped_claims,
            "scraping_success": self.scraping_success,
            "description_chunks": self.description_chunks,
        }


def format_patent_id(patent_number: str) -> str:
    """특허번호를 SERP API 형식으로 변환 (PRD 정의)"""
    if not patent_number:
        return ""

    # 이미 변환된 형식인지 확인
    if patent_number.startswith("patent/"):
        return patent_number

    # 특허번호 정제 (공백, 특수문자 제거)
    clean_number = re.sub(r"[^\w]", "", patent_number.strip().upper())
    return f"patent/{clean_number}/en"


def parse_description(html_content: str) -> str:
    """
    HTML에서 영어 번역본 추출, 원본 텍스트 제외 (PRD 정의)

    Args:
        html_content: HTML 형식의 설명 텍스트

    Returns:
        정제된 영어 번역본 텍스트
    """
    if not html_content:
        return ""

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_content, "html.parser")

        # <span class="google-src-text"> 원본 텍스트 제거
        for span in soup.find_all("span", class_="google-src-text"):
            span.decompose()

        # 텍스트 추출 및 정제
        text = soup.get_text(separator=" ", strip=True)

        # 연속 공백 제거
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    except Exception as e:
        logger.warning(f"HTML 파싱 중 오류 발생: {e}")
        # BeautifulSoup를 사용할 수 없는 경우 간단한 정제
        text = re.sub(r"<[^>]+>", "", html_content)
        return re.sub(r"\s+", " ", text).strip()


class PatentParser:
    """특허 데이터 파싱 및 처리를 담당하는 클래스"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PatentParser")

    def parse_serp_result(
        self, patent_number: str, serp_data: Dict[str, Any]
    ) -> Patent:
        """
        SerpAPI 검색 결과를 Patent 객체로 변환

        Args:
            patent_number: 특허번호
            serp_data: SerpAPI 응답 데이터

        Returns:
            구조화된 Patent 객체
        """
        try:
            patent = Patent(
                patent_number=patent_number,
                patent_id=format_patent_id(patent_number),
                title=serp_data.get("title", ""),
                abstract=serp_data.get("abstract", ""),
                claims=serp_data.get("claims", []),
                description=serp_data.get("description", ""),
                description_link=serp_data.get("description_link", ""),
                filing_date=serp_data.get("filing_date", ""),
                publication_date=serp_data.get("publication_date", ""),
                application_date=serp_data.get("application_date", ""),
                status=serp_data.get("status", ""),
                inventor=serp_data.get("inventor", []),
                assignee=serp_data.get("assignee", []),
                classifications=serp_data.get("classifications", []),
                citations=serp_data.get("citations", {}),
                search_timestamp=serp_data.get("search_timestamp", ""),
                google_patents_url=serp_data.get("google_patents_url", ""),
                serp_metadata=serp_data.get("raw_data", {}),
            )

            self.logger.debug(f"SerpAPI 데이터 파싱 완료: {patent_number}")
            return patent

        except Exception as e:
            self.logger.error(f"SerpAPI 데이터 파싱 실패 ({patent_number}): {e}")
            # 최소 정보로 Patent 객체 생성
            return Patent(
                patent_number=patent_number, patent_id=format_patent_id(patent_number)
            )

    def integrate_scraping_result(self, patent: Patent, scraping_result: Any) -> Patent:
        """
        웹 스크래핑 결과를 Patent 객체에 통합

        Args:
            patent: 기존 Patent 객체
            scraping_result: 웹 스크래핑 결과 객체

        Returns:
            스크래핑 결과가 통합된 Patent 객체
        """
        try:
            if scraping_result and scraping_result.success:
                patent.scraped_title = scraping_result.title or ""
                patent.scraped_abstract = scraping_result.abstract or ""
                patent.scraped_description = scraping_result.description or ""
                patent.scraped_claims = scraping_result.claims or []
                patent.scraping_success = True

                # HTML 형식의 설명이 있으면 파싱
                if patent.scraped_description:
                    patent.scraped_description = parse_description(
                        patent.scraped_description
                    )

                self.logger.debug(f"웹 스크래핑 결과 통합 완료: {patent.patent_number}")
            else:
                patent.scraping_success = False
                self.logger.warning(f"웹 스크래핑 실패: {patent.patent_number}")

            return patent

        except Exception as e:
            self.logger.error(f"스크래핑 결과 통합 실패 ({patent.patent_number}): {e}")
            patent.scraping_success = False
            return patent


class PatentChunker:
    """특허 문서 청킹을 담당하는 클래스"""

    def __init__(
        self,
        chunk_size: int = 1200,  # 토큰 단위 (PRD: 1000-1500)
        chunk_overlap: int = 200,  # 오버랩 크기
        length_function: callable = len,  # 길이 측정 함수
    ):
        """
        청킹 설정 초기화

        Args:
            chunk_size: 청크 크기 (문자 단위)
            chunk_overlap: 청크 간 오버랩 크기
            length_function: 길이 측정 함수 (토큰 카운터로 교체 가능)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function

        # LangChain 텍스트 분할기 초기화
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self.logger = logging.getLogger(f"{__name__}.PatentChunker")

    def chunk_patent(self, patent: Patent) -> List[DocumentChunk]:
        """
        특허 문서를 섹션별로 청킹

        Args:
            patent: 청킹할 Patent 객체

        Returns:
            문서 청크 리스트
        """
        chunks = []

        try:
            # 1. 제목 청크 (단일 청크)
            if patent.title:
                chunks.append(
                    DocumentChunk(
                        patent_number=patent.patent_number,
                        section="title",
                        chunk_index=0,
                        content=patent.title,
                        metadata={
                            "section_type": "title",
                            "patent_id": patent.patent_id,
                            "filing_date": patent.filing_date,
                            "publication_date": patent.publication_date,
                        },
                    )
                )

            # 2. 초록 청크 (단일 청크 또는 필요시 분할)
            if patent.abstract:
                abstract_chunks = self._chunk_section(
                    patent.abstract, patent.patent_number, "abstract", patent
                )
                chunks.extend(abstract_chunks)

            # 3. 청구항 청킹 (개별 청구항 또는 그룹별)
            effective_claims = patent.effective_claims
            if effective_claims:
                claims_chunks = self._chunk_claims(
                    effective_claims, patent.patent_number, patent
                )
                chunks.extend(claims_chunks)

            # 4. 상세 설명 청킹 (가장 중요한 부분)
            effective_description = patent.effective_description
            if effective_description:
                description_chunks = self._chunk_section(
                    effective_description, patent.patent_number, "description", patent
                )
                chunks.extend(description_chunks)

                # 청킹된 설명을 특허 객체에 저장
                patent.description_chunks = [
                    chunk.content for chunk in description_chunks
                ]

            self.logger.info(
                f"특허 청킹 완료: {patent.patent_number} ({len(chunks)}개 청크)"
            )

        except Exception as e:
            self.logger.error(f"특허 청킹 실패 ({patent.patent_number}): {e}")

        return chunks

    def _chunk_section(
        self, content: str, patent_number: str, section: str, patent: Patent
    ) -> List[DocumentChunk]:
        """
        특정 섹션을 청킹

        Args:
            content: 청킹할 텍스트
            patent_number: 특허번호
            section: 섹션 이름
            patent: Patent 객체 (메타데이터용)

        Returns:
            해당 섹션의 청크 리스트
        """
        if not content or not content.strip():
            return []

        # 짧은 텍스트는 분할하지 않음
        if len(content) <= self.chunk_size:
            return [
                DocumentChunk(
                    patent_number=patent_number,
                    section=section,
                    chunk_index=0,
                    content=content,
                    metadata=self._get_base_metadata(patent, section),
                )
            ]

        # 텍스트 분할
        text_chunks = self.text_splitter.split_text(content)

        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunks.append(
                DocumentChunk(
                    patent_number=patent_number,
                    section=section,
                    chunk_index=i,
                    content=chunk_text,
                    metadata={
                        **self._get_base_metadata(patent, section),
                        "chunk_total": len(text_chunks),
                        "chunk_position": i + 1,
                    },
                )
            )

        return chunks

    def _chunk_claims(
        self, claims: List[str], patent_number: str, patent: Patent
    ) -> List[DocumentChunk]:
        """
        청구항을 청킹 (개별 청구항 또는 그룹별)

        Args:
            claims: 청구항 리스트
            patent_number: 특허번호
            patent: Patent 객체 (메타데이터용)

        Returns:
            청구항 청크 리스트
        """
        chunks = []

        # 모든 청구항을 하나의 텍스트로 결합
        combined_claims = "\n\n".join(
            f"Claim {i+1}: {claim}" for i, claim in enumerate(claims)
        )

        # 크기에 따라 분할 여부 결정
        if len(combined_claims) <= self.chunk_size:
            # 단일 청크로 저장
            chunks.append(
                DocumentChunk(
                    patent_number=patent_number,
                    section="claims",
                    chunk_index=0,
                    content=combined_claims,
                    metadata={
                        **self._get_base_metadata(patent, "claims"),
                        "claims_count": len(claims),
                    },
                )
            )
        else:
            # 분할 저장
            claim_chunks = self.text_splitter.split_text(combined_claims)
            for i, chunk_text in enumerate(claim_chunks):
                chunks.append(
                    DocumentChunk(
                        patent_number=patent_number,
                        section="claims",
                        chunk_index=i,
                        content=chunk_text,
                        metadata={
                            **self._get_base_metadata(patent, "claims"),
                            "claims_count": len(claims),
                            "chunk_total": len(claim_chunks),
                            "chunk_position": i + 1,
                        },
                    )
                )

        return chunks

    def _get_base_metadata(self, patent: Patent, section: str) -> Dict[str, Any]:
        """청크 기본 메타데이터 생성"""
        return {
            "section_type": section,
            "patent_id": patent.patent_id,
            "patent_title": patent.title,
            "filing_date": patent.filing_date,
            "publication_date": patent.publication_date,
            "inventor": patent.inventor,
            "assignee": patent.assignee,
            "scraping_success": patent.scraping_success,
            "timestamp": datetime.now().isoformat(),
        }


class PatentDataProcessor:
    """특허 데이터 종합 처리 클래스"""

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 200):
        """
        Args:
            chunk_size: 청크 크기 (문자 단위)
            chunk_overlap: 청크 간 오버랩 크기
        """
        self.parser = PatentParser()
        self.chunker = PatentChunker(chunk_size, chunk_overlap)
        self.logger = logging.getLogger(f"{__name__}.PatentDataProcessor")

    def process_search_results(
        self,
        search_results: Dict[str, Any],
        scraping_results: Optional[List[Any]] = None,
    ) -> Tuple[List[Patent], List[DocumentChunk]]:
        """
        검색 결과와 스크래핑 결과를 종합 처리

        Args:
            search_results: SerpAPI 검색 결과
            scraping_results: 웹 스크래핑 결과 (선택적)

        Returns:
            (처리된 Patent 객체 리스트, 문서 청크 리스트)
        """
        patents = []
        all_chunks = []

        # 스크래핑 결과를 URL 기반으로 매핑
        scraping_map = {}
        if scraping_results:
            for result in scraping_results:
                if result and hasattr(result, "url"):
                    scraping_map[result.url] = result

        # 각 특허 처리
        for patent_number, serp_data in search_results.get("results", {}).items():
            try:
                # 1. SerpAPI 데이터 파싱
                patent = self.parser.parse_serp_result(patent_number, serp_data)

                # 2. 스크래핑 결과 통합
                if scraping_map:
                    # URL로 매칭하여 스크래핑 결과 찾기
                    description_link = serp_data.get("description_link", "")
                    google_url = serp_data.get("google_patents_url", "")

                    scraping_result = scraping_map.get(
                        description_link
                    ) or scraping_map.get(google_url)

                    if scraping_result:
                        patent = self.parser.integrate_scraping_result(
                            patent, scraping_result
                        )

                # 3. 청킹 수행
                chunks = self.chunker.chunk_patent(patent)

                patents.append(patent)
                all_chunks.extend(chunks)

                self.logger.debug(
                    f"특허 처리 완료: {patent_number} "
                    f"({len(chunks)}개 청크, 스크래핑: {patent.scraping_success})"
                )

            except Exception as e:
                self.logger.error(f"특허 처리 실패 ({patent_number}): {e}")
                continue

        self.logger.info(
            f"전체 처리 완료: {len(patents)}개 특허, {len(all_chunks)}개 청크"
        )

        return patents, all_chunks

    def combine_to_structured_format(
        self, patents: List[Patent], chunks: List[DocumentChunk]
    ) -> Dict[str, Any]:
        """
        처리된 데이터를 구조화된 형식으로 결합

        Args:
            patents: 처리된 특허 리스트
            chunks: 문서 청크 리스트

        Returns:
            구조화된 전체 데이터
        """
        return {
            "metadata": {
                "total_patents": len(patents),
                "total_chunks": len(chunks),
                "processing_timestamp": datetime.now().isoformat(),
                "chunk_distribution": self._get_chunk_distribution(chunks),
            },
            "patents": [patent.to_dict() for patent in patents],
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "patent_number": chunk.patent_number,
                    "section": chunk.section,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                }
                for chunk in chunks
            ],
            "patent_index": {
                patent.patent_number: i for i, patent in enumerate(patents)
            },
            "chunk_index": {chunk.chunk_id: i for i, chunk in enumerate(chunks)},
        }

    def _get_chunk_distribution(self, chunks: List[DocumentChunk]) -> Dict[str, int]:
        """청크 분포 통계 생성"""
        distribution = {}
        for chunk in chunks:
            section = chunk.section
            distribution[section] = distribution.get(section, 0) + 1
        return distribution
