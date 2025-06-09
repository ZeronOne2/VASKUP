#!/usr/bin/env python3
"""
텍스트 전처리 모듈

특허 문서의 텍스트를 임베딩에 최적화된 형태로 전처리합니다.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# 로거 설정
logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """전처리 결과 데이터 클래스"""

    original_text: str
    processed_text: str
    removed_elements: List[str]
    statistics: Dict[str, int]
    processing_time: float


class PatentTextPreprocessor:
    """
    특허 텍스트 전처리기

    주요 기능:
    - HTML 태그 제거
    - 특수 문자 정규화
    - 중복 공백 제거
    - 특허 특화 패턴 처리
    - 텍스트 품질 향상
    """

    def __init__(
        self,
        remove_html: bool = True,
        normalize_whitespace: bool = True,
        remove_special_chars: bool = False,
        min_length: int = 10,
        max_length: int = 10000,
    ):
        """
        전처리기 초기화

        Args:
            remove_html: HTML 태그 제거 여부
            normalize_whitespace: 공백 정규화 여부
            remove_special_chars: 특수 문자 제거 여부
            min_length: 최소 텍스트 길이
            max_length: 최대 텍스트 길이
        """
        self.remove_html = remove_html
        self.normalize_whitespace = normalize_whitespace
        self.remove_special_chars = remove_special_chars
        self.min_length = min_length
        self.max_length = max_length

        # 통계 추적
        self.total_processed = 0
        self.total_removed_chars = 0
        self.total_processing_time = 0.0

        # 정규식 패턴 컴파일
        self._compile_patterns()

        logger.info("특허 텍스트 전처리기 초기화 완료")

    def _compile_patterns(self):
        """정규식 패턴 컴파일"""
        # HTML 태그 패턴
        self.html_pattern = re.compile(r"<[^>]+>")

        # 공백 정규화 패턴
        self.whitespace_pattern = re.compile(r"\s+")

        # 특허 번호 패턴
        self.patent_number_pattern = re.compile(r"[A-Z]{2}\d{8}[A-Z]\d")

        # 도면 참조 패턴
        self.figure_ref_pattern = re.compile(r"FIG\.\s*\d+[A-Z]?", re.IGNORECASE)

        # 청구항 번호 패턴
        self.claim_number_pattern = re.compile(r"^\d+\.\s*")

        # 특수 문자 패턴 (선택적)
        self.special_chars_pattern = re.compile(r"[^\w\s\.\,\;\:\!\?\-\(\)]")

        # 중복 구두점 패턴
        self.duplicate_punct_pattern = re.compile(r"([\.]{2,}|[,]{2,}|[;]{2,})")

        # 괄호 내용 패턴 (참조 번호 등)
        self.reference_pattern = re.compile(r"\(\d+[a-z]?\)")

    def preprocess_text(self, text: str) -> PreprocessingResult:
        """
        단일 텍스트 전처리

        Args:
            text: 전처리할 텍스트

        Returns:
            PreprocessingResult: 전처리 결과
        """
        import time

        start_time = time.time()

        if not text or not isinstance(text, str):
            return PreprocessingResult(
                original_text=text or "",
                processed_text="",
                removed_elements=[],
                statistics={"original_length": 0, "processed_length": 0},
                processing_time=0.0,
            )

        original_text = text
        processed_text = text
        removed_elements = []

        # 1. HTML 태그 제거
        if self.remove_html:
            html_matches = self.html_pattern.findall(processed_text)
            if html_matches:
                removed_elements.extend(html_matches)
                processed_text = self.html_pattern.sub(" ", processed_text)

        # 2. 도면 참조 정규화
        figure_refs = self.figure_ref_pattern.findall(processed_text)
        if figure_refs:
            removed_elements.extend(figure_refs)
            processed_text = self.figure_ref_pattern.sub("[FIGURE]", processed_text)

        # 3. 참조 번호 제거 (선택적)
        ref_matches = self.reference_pattern.findall(processed_text)
        if ref_matches:
            removed_elements.extend(ref_matches)
            processed_text = self.reference_pattern.sub("", processed_text)

        # 4. 중복 구두점 정리
        processed_text = self.duplicate_punct_pattern.sub(r"\1", processed_text)

        # 5. 특수 문자 제거 (선택적)
        if self.remove_special_chars:
            special_chars = self.special_chars_pattern.findall(processed_text)
            if special_chars:
                removed_elements.extend(special_chars)
                processed_text = self.special_chars_pattern.sub(" ", processed_text)

        # 6. 공백 정규화
        if self.normalize_whitespace:
            processed_text = self.whitespace_pattern.sub(" ", processed_text)
            processed_text = processed_text.strip()

        # 7. 길이 검증
        if len(processed_text) < self.min_length:
            logger.warning(
                f"텍스트가 너무 짧습니다: {len(processed_text)} < {self.min_length}"
            )
        elif len(processed_text) > self.max_length:
            logger.warning(
                f"텍스트가 너무 깁니다: {len(processed_text)} > {self.max_length}"
            )
            processed_text = processed_text[: self.max_length]
            removed_elements.append(
                f"Truncated {len(original_text) - self.max_length} characters"
            )

        processing_time = time.time() - start_time

        # 통계 업데이트
        self.total_processed += 1
        self.total_removed_chars += len(original_text) - len(processed_text)
        self.total_processing_time += processing_time

        statistics = {
            "original_length": len(original_text),
            "processed_length": len(processed_text),
            "removed_count": len(removed_elements),
            "compression_ratio": (
                len(processed_text) / len(original_text) if original_text else 0
            ),
        }

        return PreprocessingResult(
            original_text=original_text,
            processed_text=processed_text,
            removed_elements=removed_elements,
            statistics=statistics,
            processing_time=processing_time,
        )

    def preprocess_batch(self, texts: List[str]) -> List[PreprocessingResult]:
        """
        여러 텍스트를 배치로 전처리

        Args:
            texts: 전처리할 텍스트 리스트

        Returns:
            List[PreprocessingResult]: 전처리 결과 리스트
        """
        logger.info(f"배치 전처리 시작: {len(texts)}개 텍스트")

        results = []
        for i, text in enumerate(texts):
            try:
                result = self.preprocess_text(text)
                results.append(result)

                if (i + 1) % 100 == 0:
                    logger.debug(f"전처리 진행: {i + 1}/{len(texts)}")

            except Exception as e:
                logger.error(f"텍스트 전처리 실패 (인덱스 {i}): {e}")
                # 실패한 경우 원본 텍스트 반환
                results.append(
                    PreprocessingResult(
                        original_text=text,
                        processed_text=text,
                        removed_elements=[],
                        statistics={
                            "original_length": len(text),
                            "processed_length": len(text),
                        },
                        processing_time=0.0,
                    )
                )

        logger.info(f"배치 전처리 완료: {len(results)}개 결과")
        return results

    def preprocess_patent_sections(
        self,
        title: str = "",
        abstract: str = "",
        claims: List[str] = None,
        description: str = "",
    ) -> Dict[str, any]:
        """
        특허의 각 섹션을 전처리

        Args:
            title: 특허 제목
            abstract: 특허 요약
            claims: 청구항 리스트
            description: 상세 설명

        Returns:
            Dict: 전처리된 섹션들
        """
        results = {}

        # 제목 전처리 (간단한 정리만)
        if title:
            title_result = self.preprocess_text(title)
            results["title"] = title_result.processed_text

        # 요약 전처리
        if abstract:
            abstract_result = self.preprocess_text(abstract)
            results["abstract"] = abstract_result.processed_text

        # 청구항 전처리
        if claims:
            processed_claims = []
            for i, claim in enumerate(claims):
                # 청구항 번호 제거
                claim_text = self.claim_number_pattern.sub("", claim)
                claim_result = self.preprocess_text(claim_text)
                processed_claims.append(claim_result.processed_text)
            results["claims"] = processed_claims

        # 상세 설명 전처리
        if description:
            desc_result = self.preprocess_text(description)
            results["description"] = desc_result.processed_text

        return results

    def get_statistics(self) -> Dict[str, any]:
        """전처리 통계 정보 반환"""
        return {
            "total_processed": self.total_processed,
            "total_removed_chars": self.total_removed_chars,
            "total_processing_time": self.total_processing_time,
            "avg_processing_time": (
                self.total_processing_time / self.total_processed
                if self.total_processed > 0
                else 0
            ),
            "avg_compression_ratio": (
                self.total_removed_chars / self.total_processed
                if self.total_processed > 0
                else 0
            ),
        }

    def reset_statistics(self):
        """통계 정보 초기화"""
        self.total_processed = 0
        self.total_removed_chars = 0
        self.total_processing_time = 0.0
        logger.info("전처리 통계 초기화 완료")


class PatentChunkPreprocessor:
    """
    특허 청크 전처리기

    DocumentChunk 객체들을 전처리하는 특화된 클래스
    """

    def __init__(self, text_preprocessor: Optional[PatentTextPreprocessor] = None):
        """
        청크 전처리기 초기화

        Args:
            text_preprocessor: 텍스트 전처리기 (None이면 기본값 사용)
        """
        self.text_preprocessor = text_preprocessor or PatentTextPreprocessor()
        logger.info("특허 청크 전처리기 초기화 완료")

    def preprocess_chunk(self, chunk) -> "DocumentChunk":
        """
        단일 DocumentChunk 전처리

        Args:
            chunk: DocumentChunk 객체

        Returns:
            DocumentChunk: 전처리된 청크
        """
        from ..patent_search.patent_parser import DocumentChunk

        # 텍스트 전처리
        result = self.text_preprocessor.preprocess_text(chunk.content)

        # 새로운 청크 생성 (전처리된 내용으로)
        processed_chunk = DocumentChunk(
            patent_number=chunk.patent_number,
            section=chunk.section,
            chunk_index=chunk.chunk_index,
            content=result.processed_text,
            metadata={
                **chunk.metadata,
                "preprocessing_applied": True,
                "original_length": result.statistics["original_length"],
                "processed_length": result.statistics["processed_length"],
                "compression_ratio": result.statistics["compression_ratio"],
            },
        )

        return processed_chunk

    def preprocess_chunks_batch(self, chunks: List) -> List:
        """
        여러 DocumentChunk를 배치로 전처리

        Args:
            chunks: DocumentChunk 리스트

        Returns:
            List[DocumentChunk]: 전처리된 청크 리스트
        """
        logger.info(f"청크 배치 전처리 시작: {len(chunks)}개")

        processed_chunks = []
        for chunk in chunks:
            try:
                processed_chunk = self.preprocess_chunk(chunk)
                processed_chunks.append(processed_chunk)
            except Exception as e:
                logger.error(f"청크 전처리 실패 {chunk.chunk_id}: {e}")
                # 실패한 경우 원본 청크 반환
                processed_chunks.append(chunk)

        logger.info(f"청크 배치 전처리 완료: {len(processed_chunks)}개")
        return processed_chunks
