"""
Jina AI Reranker API 기반 문서 재랭킹 시스템

Jina AI의 고성능 reranker API를 사용하여
질문-문서 쌍의 관련성을 정밀하게 평가하고 재랭킹을 수행합니다.
"""

import logging
import os
from typing import List, Dict, Any, Optional
import requests
import time

logger = logging.getLogger(__name__)


class JinaReranker:
    """
    Jina AI Reranker API 기반 문서 재랭킹 시스템

    Jina AI의 강력한 reranker 모델을 사용하여
    검색 결과의 순서를 최적화합니다.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "jina-reranker-v2-base-multilingual",
        base_url: str = "https://api.jina.ai/v1/rerank",
        max_documents: int = 100,
        timeout: int = 30,
    ):
        """
        JinaReranker 초기화

        Args:
            api_key: Jina AI API 키 (환경변수 JINA_API_KEY에서도 가져올 수 있음)
            model: 사용할 Jina reranker 모델
            base_url: Jina AI reranker API 엔드포인트
            max_documents: 한 번에 처리할 최대 문서 수
            timeout: API 요청 타임아웃 (초)
        """
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Jina AI API 키가 필요합니다. "
                "api_key 매개변수로 전달하거나 JINA_API_KEY 환경변수를 설정하세요."
            )

        self.model = model
        self.base_url = base_url
        self.max_documents = max_documents
        self.timeout = timeout

        # 사용 가능한 모델 목록
        self.available_models = {
            "v2-multilingual": "jina-reranker-v2-base-multilingual",
            "colbert-v2": "jina-colbert-v2",
            "v1-english": "jina-reranker-v1-base-en",
            "v1-tiny": "jina-reranker-v1-tiny-en",
            "multimodal": "jina-reranker-m0",
        }

        # API 헤더 설정
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        logger.info(f"JinaReranker 초기화: 모델={self.model}")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        return_documents: bool = False,
        return_scores: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        문서 재랭킹 수행

        Args:
            query: 검색 질의
            documents: 재랭킹할 문서 리스트
            top_k: 반환할 상위 K개 문서 (None=전체)
            min_score: 최소 관련성 스코어 임계값
            return_documents: API 응답에 문서 내용 포함 여부
            return_scores: 스코어 정보 포함 여부

        Returns:
            재랭킹된 문서 리스트 (스코어 포함)
        """
        if not documents:
            return []

        logger.debug(f"문서 재랭킹 시작: 쿼리='{query}', 문서 수={len(documents)}")

        # 문서를 배치로 나누어 처리 (API 제한 고려)
        all_reranked_docs = []
        batch_size = min(self.max_documents, len(documents))

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            
            # 문서 텍스트 추출
            doc_texts = [self._extract_document_text(doc) for doc in batch_docs]

            # Jina API 호출
            try:
                rerank_results = self._call_jina_api(
                    query=query,
                    documents=doc_texts,
                    top_n=len(batch_docs),
                    return_documents=return_documents,
                )

                # 결과 처리
                batch_reranked = self._process_rerank_results(
                    batch_docs, rerank_results, return_scores, i
                )
                all_reranked_docs.extend(batch_reranked)

            except Exception as e:
                logger.error(f"배치 {i//batch_size + 1} 재랭킹 실패: {e}")
                # 폴백: 원본 순서 유지
                for j, doc in enumerate(batch_docs):
                    doc_copy = doc.copy()
                    if return_scores:
                        doc_copy["rerank_score"] = 0.5
                        doc_copy["rerank_position"] = i + j + 1
                        doc_copy["original_position"] = i + j + 1
                    all_reranked_docs.append(doc_copy)

        # 전체 결과를 스코어 기준으로 재정렬
        if return_scores:
            all_reranked_docs.sort(
                key=lambda x: x.get("rerank_score", 0), reverse=True
            )
            # 위치 재계산
            for i, doc in enumerate(all_reranked_docs):
                doc["rerank_position"] = i + 1

        # 최소 스코어 필터링
        if min_score is not None:
            all_reranked_docs = [
                doc
                for doc in all_reranked_docs
                if doc.get("rerank_score", 0) >= min_score
            ]

        # 상위 K개 선택
        if top_k:
            all_reranked_docs = all_reranked_docs[:top_k]

        logger.debug(
            f"문서 재랭킹 완료: 입력={len(documents)}, 최종={len(all_reranked_docs)}"
        )

        return all_reranked_docs

    def _call_jina_api(
        self,
        query: str,
        documents: List[str],
        top_n: int,
        return_documents: bool = False,
    ) -> Dict[str, Any]:
        """
        Jina AI reranker API 호출

        Args:
            query: 검색 질의
            documents: 문서 텍스트 리스트
            top_n: 반환할 상위 문서 수
            return_documents: 문서 내용 포함 여부

        Returns:
            API 응답 결과
        """
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": return_documents,
        }

        try:
            logger.debug(f"Jina API 호출: {len(documents)}개 문서")
            start_time = time.time()

            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout,
            )

            elapsed_time = time.time() - start_time
            logger.debug(f"API 응답 시간: {elapsed_time:.2f}초")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            logger.error(f"API 요청 타임아웃: {self.timeout}초")
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"API HTTP 오류: {e}")
            logger.error(f"응답 내용: {response.text}")
            raise
        except Exception as e:
            logger.error(f"API 호출 실패: {e}")
            raise

    def _process_rerank_results(
        self,
        original_docs: List[Dict[str, Any]],
        rerank_results: Dict[str, Any],
        return_scores: bool,
        batch_offset: int,
    ) -> List[Dict[str, Any]]:
        """
        재랭킹 결과를 처리하여 최종 형태로 변환

        Args:
            original_docs: 원본 문서 리스트
            rerank_results: Jina API 응답 결과
            return_scores: 스코어 정보 포함 여부
            batch_offset: 배치 오프셋

        Returns:
            처리된 문서 리스트
        """
        processed_docs = []

        results = rerank_results.get("results", [])

        for result in results:
            doc_index = result.get("index")
            score = result.get("relevance_score", 0.0)

            if doc_index < len(original_docs):
                doc_copy = original_docs[doc_index].copy()

                if return_scores:
                    doc_copy["rerank_score"] = float(score)
                    doc_copy["original_position"] = batch_offset + doc_index + 1

                processed_docs.append(doc_copy)

        return processed_docs

    def _extract_document_text(self, document: Dict[str, Any]) -> str:
        """
        문서에서 텍스트 추출

        Args:
            document: 문서 딕셔너리

        Returns:
            추출된 텍스트
        """
        # 다양한 텍스트 필드 시도
        text_fields = [
            "content",
            "text",
            "body",
            "description",
            "abstract",
            "title",
            "summary",
        ]

        for field in text_fields:
            if field in document and document[field]:
                text = str(document[field])
                if len(text.strip()) > 0:
                    return text

        # 메타데이터에서 텍스트 추출 시도
        if "metadata" in document:
            metadata = document["metadata"]
            for field in text_fields:
                if field in metadata and metadata[field]:
                    text = str(metadata[field])
                    if len(text.strip()) > 0:
                        return text

        # 전체 문서를 문자열로 변환 (마지막 수단)
        return str(document)

    def predict_score(self, query: str, document: str) -> float:
        """
        단일 질문-문서 쌍의 관련성 스코어 예측

        Args:
            query: 검색 질의
            document: 문서 텍스트

        Returns:
            관련성 스코어 (0-1)
        """
        try:
            result = self._call_jina_api(
                query=query, documents=[document], top_n=1, return_documents=False
            )

            results = result.get("results", [])
            if results:
                return float(results[0].get("relevance_score", 0.0))

        except Exception as e:
            logger.error(f"단일 스코어 예측 실패: {e}")

        return 0.0

    def get_model_info(self) -> Dict[str, Any]:
        """
        현재 사용 중인 모델 정보 반환

        Returns:
            모델 정보 딕셔너리
        """
        return {
            "model_name": self.model,
            "api_endpoint": self.base_url,
            "max_documents": self.max_documents,
            "timeout": self.timeout,
            "available_models": self.available_models,
        }

    def update_model(self, model_name: str) -> None:
        """
        사용할 모델 변경

        Args:
            model_name: 새로운 모델명
        """
        if model_name in self.available_models.values():
            self.model = model_name
            logger.info(f"모델 변경됨: {model_name}")
        else:
            logger.warning(f"알 수 없는 모델: {model_name}")

    @classmethod
    def get_available_models(cls) -> Dict[str, str]:
        """
        사용 가능한 모델 목록 반환

        Returns:
            모델 목록 딕셔너리
        """
        return {
            "v2-multilingual": "jina-reranker-v2-base-multilingual",
            "colbert-v2": "jina-colbert-v2",
            "v1-english": "jina-reranker-v1-base-en",
            "v1-tiny": "jina-reranker-v1-tiny-en",
            "multimodal": "jina-reranker-m0",
        }

    def benchmark_reranking(
        self, query: str, documents: List[Dict[str, Any]], original_scores: List[float]
    ) -> Dict[str, Any]:
        """
        재랭킹 성능 벤치마크

        Args:
            query: 검색 질의
            documents: 문서 리스트
            original_scores: 원본 스코어 리스트

        Returns:
            벤치마크 결과
        """
        start_time = time.time()

        # 재랭킹 수행
        reranked_docs = self.rerank(query, documents, return_scores=True)

        end_time = time.time()
        processing_time = end_time - start_time

        # 성능 지표 계산
        if reranked_docs:
            avg_rerank_score = sum(
                doc.get("rerank_score", 0) for doc in reranked_docs
            ) / len(reranked_docs)
            max_rerank_score = max(doc.get("rerank_score", 0) for doc in reranked_docs)
            min_rerank_score = min(doc.get("rerank_score", 0) for doc in reranked_docs)
        else:
            avg_rerank_score = max_rerank_score = min_rerank_score = 0.0

        return {
            "processing_time": processing_time,
            "total_documents": len(documents),
            "reranked_documents": len(reranked_docs),
            "avg_rerank_score": avg_rerank_score,
            "max_rerank_score": max_rerank_score,
            "min_rerank_score": min_rerank_score,
            "model_used": self.model,
        } 