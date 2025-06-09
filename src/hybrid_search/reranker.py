"""
Cross-encoder 기반 Reranker

sentence-transformers의 Cross-encoder 모델을 사용하여
질문-문서 쌍의 관련성을 정밀하게 평가하고 재랭킹을 수행합니다.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np

try:
    from sentence_transformers import CrossEncoder

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    CrossEncoder = None

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Cross-encoder 기반 문서 재랭킹 시스템

    질문-문서 쌍에 대한 정밀한 관련성 스코어를 계산하여
    검색 결과의 순서를 최적화합니다.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 32,
        cache_dir: Optional[str] = None,
    ):
        """
        CrossEncoderReranker 초기화

        Args:
            model_name: 사용할 Cross-encoder 모델명
            device: 사용할 디바이스 ('cpu', 'cuda', None=자동선택)
            max_length: 최대 토큰 길이
            batch_size: 배치 크기
            cache_dir: 모델 캐시 디렉토리
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers가 설치되지 않았습니다. "
                "'pip install sentence-transformers'를 실행하세요."
            )

        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.cache_dir = cache_dir

        # Cross-encoder 모델
        self.model: Optional[CrossEncoder] = None
        self._is_loaded = False

        # 한국어 특허 도메인 특화 모델 목록
        self.korean_optimized_models = {
            "multilingual": "cross-encoder/ms-marco-MiniLM-L-12-v2",
            "lightweight": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "performance": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
            "robust": "cross-encoder/ms-marco-electra-base",
        }

        logger.info(f"CrossEncoderReranker 초기화: 모델={model_name}")

    def load_model(self) -> None:
        """Cross-encoder 모델 로드"""
        if self._is_loaded:
            return

        try:
            logger.info(f"Cross-encoder 모델 로딩 시작: {self.model_name}")

            # 모델 로드 설정
            model_kwargs = {"max_length": self.max_length, "device": self.device}

            if self.cache_dir:
                model_kwargs["cache_folder"] = self.cache_dir

            self.model = CrossEncoder(self.model_name, **model_kwargs)

            self._is_loaded = True
            logger.info(f"Cross-encoder 모델 로딩 완료: {self.model_name}")

        except Exception as e:
            logger.error(f"Cross-encoder 모델 로딩 실패: {e}")
            # 기본 모델로 폴백
            if self.model_name != self.korean_optimized_models["lightweight"]:
                logger.info("기본 모델로 폴백 시도...")
                self.model_name = self.korean_optimized_models["lightweight"]
                self.load_model()
            else:
                raise

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        min_score: float = 0.0,
        return_scores: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        문서 재랭킹 수행

        Args:
            query: 검색 질의
            documents: 재랭킹할 문서 리스트
            top_k: 반환할 상위 K개 문서 (None=전체)
            min_score: 최소 관련성 스코어 임계값
            return_scores: 스코어 정보 포함 여부

        Returns:
            재랭킹된 문서 리스트 (스코어 포함)
        """
        if not self._is_loaded:
            self.load_model()

        if not documents:
            return []

        logger.debug(f"문서 재랭킹 시작: 쿼리='{query}', 문서 수={len(documents)}")

        # 질문-문서 쌍 생성
        query_doc_pairs = []
        for doc in documents:
            doc_text = self._extract_document_text(doc)
            query_doc_pairs.append([query, doc_text])

        # Cross-encoder로 관련성 스코어 계산
        try:
            scores = self.model.predict(
                query_doc_pairs,
                batch_size=self.batch_size,
                show_progress_bar=len(query_doc_pairs) > 100,
            )

            # numpy array를 리스트로 변환
            if isinstance(scores, np.ndarray):
                scores = scores.tolist()

        except Exception as e:
            logger.error(f"Cross-encoder 예측 실패: {e}")
            # 폴백: 원본 순서 유지
            scores = [0.5] * len(documents)

        # 문서와 스코어 결합
        doc_score_pairs = list(zip(documents, scores))

        # 최소 스코어 필터링
        if min_score > 0:
            doc_score_pairs = [
                (doc, score) for doc, score in doc_score_pairs if score >= min_score
            ]

        # 스코어 기준 정렬 (높은 순)
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # 상위 K개 선택
        if top_k:
            doc_score_pairs = doc_score_pairs[:top_k]

        # 결과 포맷팅
        reranked_docs = []
        for i, (doc, score) in enumerate(doc_score_pairs):
            reranked_doc = doc.copy()

            if return_scores:
                # 기존 스코어 정보 보존 및 새 스코어 추가
                if "rerank_score" not in reranked_doc:
                    reranked_doc["rerank_score"] = float(score)
                    reranked_doc["rerank_position"] = i + 1

                # 원본 스코어 정보도 보존
                if "original_position" not in reranked_doc:
                    original_pos = documents.index(doc) + 1
                    reranked_doc["original_position"] = original_pos

            reranked_docs.append(reranked_doc)

        logger.debug(
            f"문서 재랭킹 완료: 입력={len(documents)}, "
            f"필터링 후={len(doc_score_pairs)}, 최종={len(reranked_docs)}"
        )

        return reranked_docs

    def predict_score(self, query: str, document: str) -> float:
        """
        단일 질문-문서 쌍의 관련성 스코어 예측

        Args:
            query: 검색 질의
            document: 문서 텍스트

        Returns:
            관련성 스코어 (0~1)
        """
        if not self._is_loaded:
            self.load_model()

        try:
            score = self.model.predict([[query, document]])[0]
            return float(score)
        except Exception as e:
            logger.error(f"스코어 예측 실패: {e}")
            return 0.5

    def batch_predict_scores(
        self, query_doc_pairs: List[Tuple[str, str]]
    ) -> List[float]:
        """
        배치 관련성 스코어 예측

        Args:
            query_doc_pairs: (질문, 문서) 쌍 리스트

        Returns:
            관련성 스코어 리스트
        """
        if not self._is_loaded:
            self.load_model()

        if not query_doc_pairs:
            return []

        try:
            scores = self.model.predict(
                query_doc_pairs,
                batch_size=self.batch_size,
                show_progress_bar=len(query_doc_pairs) > 100,
            )

            if isinstance(scores, np.ndarray):
                return scores.tolist()
            return list(scores)

        except Exception as e:
            logger.error(f"배치 스코어 예측 실패: {e}")
            return [0.5] * len(query_doc_pairs)

    def _extract_document_text(self, document: Dict[str, Any]) -> str:
        """문서에서 텍스트 추출"""
        # 문서 형식에 따른 텍스트 추출
        if isinstance(document, dict):
            # content 필드 확인
            content = document.get("content", {})

            if isinstance(content, str):
                return content
            elif isinstance(content, dict):
                # Vector Store 형식
                text = content.get("page_content", "")
                if text:
                    return text

                # 기타 텍스트 필드들 시도
                for field in ["text", "description", "summary"]:
                    if field in content and content[field]:
                        return str(content[field])

            # 메타데이터에서 텍스트 추출 시도
            metadata = document.get("metadata", {})
            if isinstance(metadata, dict):
                for field in ["description", "summary", "abstract"]:
                    if field in metadata and metadata[field]:
                        return str(metadata[field])

            # 전체 내용을 문자열로 변환
            return str(content)

        return str(document)

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        info = {
            "model_name": self.model_name,
            "is_loaded": self._is_loaded,
            "device": self.device,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "available_models": self.korean_optimized_models,
        }

        if self._is_loaded and self.model:
            try:
                info["model_device"] = str(self.model.model.device)
                info["model_config"] = self.model.model.config.to_dict()
            except:
                pass

        return info

    def update_model(self, model_name: str) -> None:
        """
        사용 모델 변경

        Args:
            model_name: 새로운 모델명 또는 프리셋 키
        """
        # 프리셋 모델 확인
        if model_name in self.korean_optimized_models:
            model_name = self.korean_optimized_models[model_name]

        if model_name != self.model_name:
            logger.info(f"모델 변경: {self.model_name} -> {model_name}")
            self.model_name = model_name
            self._is_loaded = False
            self.model = None

    @classmethod
    def get_available_models(cls) -> Dict[str, str]:
        """사용 가능한 모델 목록 반환"""
        return {
            "multilingual": "cross-encoder/ms-marco-MiniLM-L-12-v2",
            "lightweight": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "performance": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
            "robust": "cross-encoder/ms-marco-electra-base",
            "distilbert": "cross-encoder/ms-marco-MiniLM-L-4-v2",
            "large": "cross-encoder/ms-marco-roberta-base",
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
        import time

        start_time = time.time()
        reranked_docs = self.rerank(query, documents, return_scores=True)
        end_time = time.time()

        # 순서 변화 분석
        original_order = list(range(len(documents)))
        reranked_order = [documents.index(doc) for doc in reranked_docs]

        # 상위 문서 변화 분석 (top-5)
        top_5_original = original_order[:5]
        top_5_reranked = reranked_order[:5]
        top_5_overlap = len(set(top_5_original) & set(top_5_reranked))

        return {
            "processing_time": end_time - start_time,
            "total_documents": len(documents),
            "reranked_documents": len(reranked_docs),
            "top_5_overlap": top_5_overlap,
            "top_5_overlap_ratio": top_5_overlap / 5,
            "average_rerank_score": np.mean(
                [doc.get("rerank_score", 0) for doc in reranked_docs]
            ),
            "score_distribution": {
                "min": min([doc.get("rerank_score", 0) for doc in reranked_docs]),
                "max": max([doc.get("rerank_score", 0) for doc in reranked_docs]),
                "mean": np.mean([doc.get("rerank_score", 0) for doc in reranked_docs]),
                "std": np.std([doc.get("rerank_score", 0) for doc in reranked_docs]),
            },
        }
