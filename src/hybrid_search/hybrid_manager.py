"""
하이브리드 검색 매니저

Vector Search와 BM25 키워드 검색을 결합하여
의미적 유사성과 정확한 키워드 매칭의 장점을 모두 활용합니다.
Jina AI Reranker를 통한 고도화된 재랭킹 기능도 제공합니다.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from .bm25_search import BM25SearchEngine

try:
    from .jina_reranker import JinaReranker

    JINA_RERANKER_AVAILABLE = True
except ImportError:
    JINA_RERANKER_AVAILABLE = False
    JinaReranker = None

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """통합 검색 결과 데이터 클래스"""

    content: Dict[str, Any]
    vector_score: float
    bm25_score: float
    combined_score: float
    doc_index: int
    metadata: Dict[str, Any]
    search_methods: List[str]
    rerank_score: Optional[float] = None
    original_position: Optional[int] = None
    rerank_position: Optional[int] = None


class HybridSearchManager:
    """
    하이브리드 검색 매니저

    Vector Search와 BM25 검색을 결합하여
    특허 문서에 대한 최적의 검색 성능을 제공합니다.
    Jina AI Reranker를 통한 고도화된 재랭킹도 지원합니다.
    """

    def __init__(
        self,
        vector_store,
        bm25_engine: Optional[BM25SearchEngine] = None,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        normalization_method: str = "min_max",
        cache_dir: Optional[str] = None,
        use_jina_reranker: bool = False,
        jina_api_key: Optional[str] = None,
        jina_model: str = "jina-reranker-v2-base-multilingual",
    ):
        """
        HybridSearchManager 초기화

        Args:
            vector_store: Vector Store 인스턴스
            bm25_engine: BM25 검색 엔진 (None이면 자동 생성)
            vector_weight: Vector Search 가중치 (기본값: 0.6)
            bm25_weight: BM25 Search 가중치 (기본값: 0.4)
            normalization_method: 스코어 정규화 방법 ('min_max', 'z_score')
            cache_dir: BM25 캐시 디렉토리
            use_jina_reranker: Jina AI Reranker 사용 여부
            jina_api_key: Jina AI API 키
            jina_model: 사용할 Jina reranker 모델
        """
        self.vector_store = vector_store
        self.bm25_engine = bm25_engine or BM25SearchEngine(cache_dir=cache_dir)

        # 가중치 설정 (합이 1이 되도록 정규화)
        total_weight = vector_weight + bm25_weight
        self.vector_weight = vector_weight / total_weight
        self.bm25_weight = bm25_weight / total_weight

        self.normalization_method = normalization_method
        self._is_initialized = False

        # Jina Reranker 설정
        self.use_jina_reranker = use_jina_reranker
        self.jina_reranker = None

        if self.use_jina_reranker:
            if not JINA_RERANKER_AVAILABLE:
                logger.warning(
                    "Jina Reranker를 사용할 수 없습니다. 기본 검색으로 전환됩니다."
                )
                self.use_jina_reranker = False
            else:
                try:
                    self.jina_reranker = JinaReranker(
                        api_key=jina_api_key, model=jina_model
                    )
                    logger.info(f"Jina Reranker 초기화 완료: {jina_model}")
                except Exception as e:
                    logger.error(f"Jina Reranker 초기화 실패: {e}")
                    self.use_jina_reranker = False

        logger.info(
            f"하이브리드 검색 매니저 초기화: "
            f"Vector({self.vector_weight:.2f}) + BM25({self.bm25_weight:.2f})"
            f"{' + Jina Reranker' if self.use_jina_reranker else ''}"
        )

    def initialize(self, force_rebuild: bool = False) -> None:
        """
        하이브리드 검색 시스템 초기화

        Args:
            force_rebuild: BM25 인덱스 강제 재구축 여부
        """
        logger.info("하이브리드 검색 시스템 초기화 시작...")

        # Vector Store에서 문서 추출
        documents = self._extract_documents_from_vector_store()

        if not documents:
            raise ValueError("Vector Store에서 문서를 찾을 수 없습니다.")

        # BM25 인덱스 구축
        if not force_rebuild and self.bm25_engine.load_cache():
            logger.info("BM25 캐시에서 인덱스 로드 완료")
        else:
            logger.info(f"BM25 인덱스 구축 시작: {len(documents)}개 문서")
            self.bm25_engine.build_index(documents)

        self._is_initialized = True
        logger.info("하이브리드 검색 시스템 초기화 완료")

    def search(
        self,
        query: str,
        n_results: int = 10,
        vector_n_results: int = 20,
        bm25_n_results: int = 20,
        min_combined_score: float = 0.0,
        boost_exact_matches: bool = True,
        enable_reranking: bool = True,
        rerank_top_k: int = 10,
    ) -> List[SearchResult]:
        """
        하이브리드 검색 수행

        Args:
            query: 검색 질의
            n_results: 최종 반환할 결과 수
            vector_n_results: Vector Search 중간 결과 수
            bm25_n_results: BM25 Search 중간 결과 수
            min_combined_score: 최소 결합 스코어 임계값
            boost_exact_matches: 정확한 매칭에 대한 부스팅 여부
            enable_reranking: Jina 재랭킹 활성화 여부
            rerank_top_k: 재랭킹할 상위 문서 수

        Returns:
            통합된 검색 결과 리스트 (재랭킹 포함)
        """
        if not self._is_initialized:
            raise ValueError(
                "하이브리드 검색 시스템이 초기화되지 않았습니다. initialize()를 먼저 호출하세요."
            )

        logger.debug(f"하이브리드 검색 시작: 쿼리='{query}'")

        # Vector Search 수행
        vector_results = self._perform_vector_search(query, vector_n_results)

        # BM25 Search 수행
        bm25_results = self._perform_bm25_search(query, bm25_n_results)

        # 결과 통합 및 스코어 결합
        combined_results = self._combine_search_results(
            vector_results, bm25_results, boost_exact_matches, query
        )

        # 최종 결과 필터링 및 정렬
        final_results = [
            result
            for result in combined_results
            if result.combined_score >= min_combined_score
        ]
        final_results.sort(key=lambda x: x.combined_score, reverse=True)

        # Jina Reranker 적용 (설정된 경우)
        if (
            self.use_jina_reranker
            and enable_reranking
            and self.jina_reranker
            and len(final_results) > 1
        ):

            logger.debug(f"Jina 재랭킹 시작: {len(final_results)}개 문서")

            # 원본 위치 기록
            for i, result in enumerate(final_results):
                result.original_position = i + 1

            # 재랭킹할 문서 선택
            rerank_candidates = final_results[:rerank_top_k]

            try:
                # SearchResult를 딕셔너리로 변환하여 Jina reranker에 전달
                docs_for_rerank = []
                for result in rerank_candidates:
                    doc_dict = {
                        "content": result.content,
                        "metadata": result.metadata,
                        "combined_score": result.combined_score,
                        "vector_score": result.vector_score,
                        "bm25_score": result.bm25_score,
                    }
                    docs_for_rerank.append(doc_dict)

                # Jina reranker 호출
                reranked_docs = self.jina_reranker.rerank(
                    query=query,
                    documents=docs_for_rerank,
                    top_k=len(rerank_candidates),
                    return_scores=True,
                )

                # 재랭킹 결과를 SearchResult로 다시 변환
                reranked_results = []
                for i, reranked_doc in enumerate(reranked_docs):
                    # 원본 SearchResult 찾기
                    original_result = None
                    for result in rerank_candidates:
                        if (
                            result.content == reranked_doc["content"]
                            and result.metadata == reranked_doc["metadata"]
                        ):
                            original_result = result
                            break

                    if original_result:
                        # 재랭킹 정보 추가
                        original_result.rerank_score = reranked_doc.get(
                            "rerank_score", 0.0
                        )
                        original_result.rerank_position = i + 1
                        reranked_results.append(original_result)

                # 재랭킹되지 않은 나머지 결과 추가
                remaining_results = final_results[rerank_top_k:]
                for i, result in enumerate(remaining_results):
                    result.original_position = rerank_top_k + i + 1
                    result.rerank_position = len(reranked_results) + i + 1

                final_results = reranked_results + remaining_results

                logger.debug(f"Jina 재랭킹 완료: {len(reranked_results)}개 문서 재정렬")

            except Exception as e:
                logger.error(f"Jina 재랭킹 실패: {e}")
                # 재랭킹 실패 시 원본 순서 유지
                for i, result in enumerate(final_results):
                    result.original_position = i + 1
                    result.rerank_position = i + 1
        else:
            # 재랭킹 미사용 시 위치 정보만 설정
            for i, result in enumerate(final_results):
                result.original_position = i + 1
                result.rerank_position = i + 1

        # 상위 n_results 반환
        final_results = final_results[:n_results]

        logger.debug(
            f"하이브리드 검색 완료: "
            f"Vector={len(vector_results)}, BM25={len(bm25_results)}, "
            f"Combined={len(combined_results)}, Final={len(final_results)}"
            f"{', Reranked' if self.use_jina_reranker and enable_reranking else ''}"
        )

        return final_results

    def _extract_documents_from_vector_store(self) -> List[Dict[str, Any]]:
        """Vector Store에서 문서 추출"""
        try:
            # Vector Store의 모든 문서 가져오기
            if hasattr(self.vector_store, "get_all_documents"):
                return self.vector_store.get_all_documents()
            elif hasattr(self.vector_store, "_collection"):
                # Chroma DB의 경우
                collection = self.vector_store._collection
                all_data = collection.get()

                documents = []
                for i, (doc_id, content, metadata) in enumerate(
                    zip(
                        all_data.get("ids", []),
                        all_data.get("documents", []),
                        all_data.get("metadatas", []),
                    )
                ):
                    documents.append(
                        {
                            "content": content,
                            "metadata": metadata or {},
                            "doc_id": doc_id,
                        }
                    )
                return documents
            else:
                logger.warning("Vector Store에서 문서 추출 방법을 찾을 수 없습니다.")
                return []

        except Exception as e:
            logger.error(f"Vector Store에서 문서 추출 실패: {e}")
            return []

    def _perform_vector_search(
        self, query: str, n_results: int
    ) -> List[Dict[str, Any]]:
        """Vector Search 수행"""
        try:
            search_results = self.vector_store.search_similar(
                query, n_results=n_results
            )

            # 결과 형식 통일
            vector_results = []
            results = search_results.get("results", [])

            for i, result in enumerate(results):
                vector_results.append(
                    {
                        "content": result,
                        "score": 1.0 - (i * 0.1),  # 거리 기반 스코어 추정
                        "doc_index": i,
                        "metadata": result.get("metadata", {}),
                        "search_method": "vector",
                    }
                )

            return vector_results

        except Exception as e:
            logger.error(f"Vector Search 실패: {e}")
            return []

    def _perform_bm25_search(self, query: str, n_results: int) -> List[Dict[str, Any]]:
        """BM25 Search 수행"""
        try:
            return self.bm25_engine.search(query, n_results=n_results)
        except Exception as e:
            logger.error(f"BM25 Search 실패: {e}")
            return []

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """스코어 정규화"""
        if not scores:
            return []

        scores_array = np.array(scores)

        if self.normalization_method == "min_max":
            min_score = scores_array.min()
            max_score = scores_array.max()
            if max_score > min_score:
                return ((scores_array - min_score) / (max_score - min_score)).tolist()
            else:
                return [1.0] * len(scores)

        elif self.normalization_method == "z_score":
            mean_score = scores_array.mean()
            std_score = scores_array.std()
            if std_score > 0:
                normalized = (scores_array - mean_score) / std_score
                # Z-score를 0-1 범위로 변환
                return np.clip((normalized + 3) / 6, 0, 1).tolist()
            else:
                return [0.5] * len(scores)

        else:
            return scores

    def _combine_search_results(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        boost_exact_matches: bool,
        query: str,
    ) -> List[SearchResult]:
        """검색 결과 통합 및 스코어 결합"""

        # 문서 ID별 결과 매핑
        doc_map = {}

        # Vector Search 결과 처리
        vector_scores = [r["score"] for r in vector_results]
        normalized_vector_scores = self._normalize_scores(vector_scores)

        for result, norm_score in zip(vector_results, normalized_vector_scores):
            doc_key = self._get_document_key(result)
            doc_map[doc_key] = {
                "content": result["content"],
                "vector_score": norm_score,
                "bm25_score": 0.0,
                "metadata": result["metadata"],
                "search_methods": ["vector"],
            }

        # BM25 Search 결과 처리
        bm25_scores = [r["score"] for r in bm25_results]
        normalized_bm25_scores = self._normalize_scores(bm25_scores)

        for result, norm_score in zip(bm25_results, normalized_bm25_scores):
            doc_key = self._get_document_key(result)

            if doc_key in doc_map:
                # 두 검색 방법 모두에서 발견된 문서
                doc_map[doc_key]["bm25_score"] = norm_score
                doc_map[doc_key]["search_methods"].append("bm25")
            else:
                # BM25에서만 발견된 문서
                doc_map[doc_key] = {
                    "content": result["content"],
                    "vector_score": 0.0,
                    "bm25_score": norm_score,
                    "metadata": result["metadata"],
                    "search_methods": ["bm25"],
                }

        # 결합 스코어 계산
        combined_results = []
        for i, (doc_key, doc_data) in enumerate(doc_map.items()):
            # 기본 결합 스코어
            combined_score = (
                doc_data["vector_score"] * self.vector_weight
                + doc_data["bm25_score"] * self.bm25_weight
            )

            # 정확한 매칭 부스팅
            if boost_exact_matches:
                exact_match_boost = self._calculate_exact_match_boost(
                    doc_data["content"], query
                )
                combined_score += exact_match_boost

            # 다중 검색 방법 부스팅
            if len(doc_data["search_methods"]) > 1:
                combined_score *= 1.1  # 10% 부스팅

            search_result = SearchResult(
                content=doc_data["content"],
                vector_score=doc_data["vector_score"],
                bm25_score=doc_data["bm25_score"],
                combined_score=combined_score,
                doc_index=i,
                metadata=doc_data["metadata"],
                search_methods=doc_data["search_methods"],
            )

            combined_results.append(search_result)

        return combined_results

    def _get_document_key(self, result: Dict[str, Any]) -> str:
        """문서 고유 키 생성"""
        content = result.get("content", {})
        metadata = result.get("metadata", {})

        # 특허 번호가 있으면 사용
        if "patent_number" in metadata:
            return metadata["patent_number"]

        # 문서 ID가 있으면 사용
        if "doc_id" in result:
            return result["doc_id"]

        # 내용 해시 사용
        content_str = str(content)
        return str(hash(content_str))

    def _calculate_exact_match_boost(
        self, content: Dict[str, Any], query: str
    ) -> float:
        """정확한 매칭에 대한 부스팅 스코어 계산"""
        content_str = str(content).lower()
        query_lower = query.lower()

        # 정확한 구문 매칭
        if query_lower in content_str:
            return 0.1

        # 개별 단어 매칭 비율
        query_words = query_lower.split()
        matched_words = sum(1 for word in query_words if word in content_str)
        word_match_ratio = matched_words / len(query_words) if query_words else 0

        return word_match_ratio * 0.05

    def get_search_stats(self) -> Dict[str, Any]:
        """검색 시스템 통계 정보 반환"""
        stats = {
            "is_initialized": self._is_initialized,
            "vector_weight": self.vector_weight,
            "bm25_weight": self.bm25_weight,
            "normalization_method": self.normalization_method,
            "use_jina_reranker": self.use_jina_reranker,
        }

        if self.bm25_engine:
            stats["bm25_stats"] = self.bm25_engine.get_document_stats()

        if self.jina_reranker:
            stats["jina_reranker_info"] = self.jina_reranker.get_model_info()

        return stats

    def update_weights(self, vector_weight: float, bm25_weight: float) -> None:
        """검색 가중치 업데이트"""
        total_weight = vector_weight + bm25_weight
        self.vector_weight = vector_weight / total_weight
        self.bm25_weight = bm25_weight / total_weight

        logger.info(
            f"검색 가중치 업데이트: "
            f"Vector({self.vector_weight:.2f}) + BM25({self.bm25_weight:.2f})"
        )

    def hybrid_search(self, query: str, k: int = 10, **kwargs) -> List[SearchResult]:
        """
        하이브리드 검색 (호환성을 위한 래퍼 메서드)

        Args:
            query: 검색 질의
            k: 반환할 결과 수
            **kwargs: 추가 매개변수

        Returns:
            검색 결과 리스트
        """
        logger.warning(
            "hybrid_search() 메서드는 deprecated입니다. search()를 사용하세요."
        )
        return self.search(query=query, n_results=k, **kwargs)
