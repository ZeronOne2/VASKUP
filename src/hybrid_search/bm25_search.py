"""
BM25 키워드 검색 엔진

rank-bm25 라이브러리를 사용하여 특허 문서에 특화된 키워드 기반 검색을 제공합니다.
"""

import logging
import pickle
import re
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from rank_bm25 import BM25Okapi
import numpy as np

logger = logging.getLogger(__name__)


class BM25SearchEngine:
    """
    BM25 알고리즘 기반 키워드 검색 엔진

    특허 문서에 특화된 전처리와 파라미터 튜닝을 통해
    정확한 키워드 매칭과 전문 용어 검색을 지원합니다.
    """

    def __init__(
        self, k1: float = 1.5, b: float = 0.75, cache_dir: Optional[str] = None
    ):
        """
        BM25SearchEngine 초기화

        Args:
            k1: 용어 빈도 포화 파라미터 (기본값: 1.5)
            b: 문서 길이 정규화 파라미터 (기본값: 0.75)
            cache_dir: 인덱스 캐시 디렉토리
        """
        self.k1 = k1
        self.b = b
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # BM25 모델과 문서 정보
        self.bm25_model: Optional[BM25Okapi] = None
        self.documents: List[Dict[str, Any]] = []
        self.tokenized_docs: List[List[str]] = []
        self.doc_metadata: List[Dict[str, Any]] = []

        # 한국어 특허 도메인 특화 설정
        self.stop_words = {
            "의",
            "가",
            "을",
            "를",
            "에",
            "에서",
            "와",
            "과",
            "로",
            "으로",
            "은",
            "는",
            "이",
            "그",
            "및",
            "또는",
            "등",
            "것",
            "수",
            "내",
            "상기",
            "본",
            "해당",
            "관련",
            "포함",
            "구성",
            "형성",
            "설치",
        }

    def _preprocess_text(self, text: str) -> List[str]:
        """
        특허 문서 텍스트 전처리 및 토큰화

        Args:
            text: 원본 텍스트

        Returns:
            전처리된 토큰 리스트
        """
        if not text:
            return []

        # 소문자 변환
        text = text.lower()

        # 특수문자 제거 (특허 번호와 기술 용어는 보존)
        text = re.sub(r"[^\w\s\-\.()]", " ", text)

        # 연속된 공백 제거
        text = re.sub(r"\s+", " ", text).strip()

        # 토큰화
        tokens = text.split()

        # 불용어 제거 및 최소 길이 필터링
        tokens = [
            token
            for token in tokens
            if len(token) >= 2 and token not in self.stop_words
        ]

        return tokens

    def _extract_technical_terms(self, tokens: List[str]) -> List[str]:
        """
        기술 용어와 특허 번호 추출

        Args:
            tokens: 토큰 리스트

        Returns:
            기술 용어가 강화된 토큰 리스트
        """
        enhanced_tokens = tokens.copy()

        # 특허 번호 패턴 강화
        patent_patterns = [
            r"\d{4}-\d+",  # 한국 특허번호
            r"US\d+",  # 미국 특허번호
            r"EP\d+",  # 유럽 특허번호
        ]

        text = " ".join(tokens)
        for pattern in patent_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            enhanced_tokens.extend(matches)

        # 기술 용어 패턴 (대문자로 시작하는 복합어)
        technical_terms = [
            token
            for token in tokens
            if len(token) > 3 and any(c.isupper() for c in token)
        ]
        enhanced_tokens.extend(technical_terms)

        return enhanced_tokens

    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        """
        문서 컬렉션으로부터 BM25 인덱스 구축

        Args:
            documents: 문서 딕셔너리 리스트 (content, metadata 포함)
        """
        logger.info(f"BM25 인덱스 구축 시작: {len(documents)}개 문서")

        self.documents = documents
        self.tokenized_docs = []
        self.doc_metadata = []

        for i, doc in enumerate(documents):
            # 문서 내용 추출
            content = doc.get("content", "")
            if isinstance(content, dict):
                # Vector Store 형식의 문서인 경우
                content = content.get("page_content", str(content))
            elif not isinstance(content, str):
                content = str(content)

            # 텍스트 전처리 및 토큰화
            tokens = self._preprocess_text(content)
            enhanced_tokens = self._extract_technical_terms(tokens)

            self.tokenized_docs.append(enhanced_tokens)

            # 메타데이터 저장
            metadata = doc.get("metadata", {})
            if isinstance(metadata, dict):
                self.doc_metadata.append(
                    {
                        "doc_index": i,
                        "patent_number": metadata.get("patent_number", f"doc_{i}"),
                        **metadata,
                    }
                )
            else:
                self.doc_metadata.append({"doc_index": i, "patent_number": f"doc_{i}"})

        # BM25 모델 생성
        self.bm25_model = BM25Okapi(self.tokenized_docs, k1=self.k1, b=self.b)

        logger.info(f"BM25 인덱스 구축 완료: {len(self.tokenized_docs)}개 문서 인덱싱")

        # 캐시 저장
        if self.cache_dir:
            self._save_cache()

    def search(
        self, query: str, n_results: int = 10, min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        BM25 키워드 검색 수행

        Args:
            query: 검색 질의
            n_results: 반환할 결과 수
            min_score: 최소 스코어 임계값

        Returns:
            검색 결과 리스트 (스코어 포함)
        """
        if not self.bm25_model:
            raise ValueError(
                "BM25 인덱스가 구축되지 않았습니다. build_index()를 먼저 호출하세요."
            )

        # 쿼리 전처리
        query_tokens = self._preprocess_text(query)
        enhanced_query_tokens = self._extract_technical_terms(query_tokens)

        if not enhanced_query_tokens:
            logger.warning("검색 쿼리에서 유효한 토큰을 찾을 수 없습니다.")
            return []

        # BM25 스코어 계산
        scores = self.bm25_model.get_scores(enhanced_query_tokens)

        # 결과 정렬 및 필터링
        scored_docs = [
            (i, score) for i, score in enumerate(scores) if score >= min_score
        ]
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 상위 n_results 개 반환
        results = []
        for doc_idx, score in scored_docs[:n_results]:
            result = {
                "content": self.documents[doc_idx],
                "score": float(score),
                "doc_index": doc_idx,
                "metadata": self.doc_metadata[doc_idx],
                "search_method": "bm25",
            }
            results.append(result)

        logger.debug(f"BM25 검색 완료: 쿼리='{query}', 결과={len(results)}개")
        return results

    def get_document_stats(self) -> Dict[str, Any]:
        """
        인덱싱된 문서 통계 정보 반환

        Returns:
            문서 통계 딕셔너리
        """
        if not self.bm25_model:
            return {}

        avg_doc_len = np.mean([len(doc) for doc in self.tokenized_docs])

        return {
            "total_documents": len(self.documents),
            "average_document_length": float(avg_doc_len),
            "vocabulary_size": len(set().union(*self.tokenized_docs)),
            "bm25_parameters": {"k1": self.k1, "b": self.b},
        }

    def _save_cache(self) -> None:
        """BM25 인덱스를 캐시에 저장"""
        if not self.cache_dir:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / "bm25_index.pkl"

        cache_data = {
            "bm25_model": self.bm25_model,
            "documents": self.documents,
            "tokenized_docs": self.tokenized_docs,
            "doc_metadata": self.doc_metadata,
            "k1": self.k1,
            "b": self.b,
        }

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f)
            logger.info(f"BM25 인덱스 캐시 저장: {cache_file}")
        except Exception as e:
            logger.error(f"BM25 인덱스 캐시 저장 실패: {e}")

    def load_cache(self) -> bool:
        """
        캐시된 BM25 인덱스 로드

        Returns:
            로드 성공 여부
        """
        if not self.cache_dir:
            return False

        cache_file = self.cache_dir / "bm25_index.pkl"
        if not cache_file.exists():
            return False

        try:
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)

            self.bm25_model = cache_data["bm25_model"]
            self.documents = cache_data["documents"]
            self.tokenized_docs = cache_data["tokenized_docs"]
            self.doc_metadata = cache_data["doc_metadata"]
            self.k1 = cache_data["k1"]
            self.b = cache_data["b"]

            logger.info(f"BM25 인덱스 캐시 로드 완료: {len(self.documents)}개 문서")
            return True

        except Exception as e:
            logger.error(f"BM25 인덱스 캐시 로드 실패: {e}")
            return False
