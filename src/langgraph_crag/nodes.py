"""
LangGraph CRAG 시스템의 노드 함수들 (Vector Store 전용)

각 처리 단계를 담당하는 노드 함수들을 정의합니다.
"""

import os
import logging
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.schema import Document

from .graders import DocumentGrader
from .state import GraphState
from ..hybrid_search import HybridSearchManager

logger = logging.getLogger(__name__)


class CRAGNodes:
    """CRAG 파이프라인의 모든 노드 함수들을 포함하는 클래스 (Vector Store 전용)"""

    def __init__(
        self,
        vector_store,
        model_name: str = "gpt-4o-mini",
        enable_hybrid_search: bool = True,
        enable_reranking: bool = True,
        hybrid_vector_weight: float = 0.6,
        rerank_top_k: int = 10,
        jina_api_key: str = None,
        jina_model: str = "jina-reranker-v2-base-multilingual",
    ):
        """
        CRAGNodes 초기화

        Args:
            vector_store: 특허 벡터 스토어 인스턴스
            model_name: 사용할 LLM 모델명
            enable_hybrid_search: 하이브리드 검색 활성화 여부
            enable_reranking: Jina AI 재랭킹 활성화 여부
            hybrid_vector_weight: 하이브리드 검색에서 Vector Search 가중치
            rerank_top_k: 재랭킹할 상위 문서 수
            jina_api_key: Jina AI API 키
            jina_model: 사용할 Jina reranker 모델
        """
        self.vector_store = vector_store
        self.model_name = model_name
        self.enable_hybrid_search = enable_hybrid_search
        self.enable_reranking = enable_reranking
        self.rerank_top_k = rerank_top_k

        # 문서 그레이더 초기화
        self.document_grader = DocumentGrader(model_name=model_name)

        # LLM 초기화
        self.llm = ChatOpenAI(
            model=model_name, temperature=0, api_key=os.environ.get("OPENAI_API_KEY")
        )

        # 하이브리드 검색 매니저 초기화 (Jina reranker 포함)
        if self.enable_hybrid_search:
            try:
                self.hybrid_manager = HybridSearchManager(
                    vector_store=vector_store,
                    vector_weight=hybrid_vector_weight,
                    bm25_weight=1.0 - hybrid_vector_weight,
                    use_jina_reranker=enable_reranking,
                    jina_api_key=jina_api_key,
                    jina_model=jina_model,
                )
                # BM25 인덱스 초기화 (중요!)
                self.hybrid_manager.initialize()
                logger.info(
                    "하이브리드 검색 매니저 초기화 완료 (BM25 인덱스 + Jina Reranker 포함)"
                )
            except Exception as e:
                logger.warning(f"하이브리드 검색 초기화 실패: {e}")
                self.enable_hybrid_search = False
                self.hybrid_manager = None
        else:
            self.hybrid_manager = None

        # 프롬프트 템플릿들
        self._init_prompts()

        logger.info(
            f"CRAGNodes 초기화 완료 - "
            f"하이브리드검색: {self.enable_hybrid_search}, "
            f"Jina 재랭킹: {self.enable_reranking}"
        )

    def _init_prompts(self):
        """프롬프트 템플릿 초기화"""

        # 문서 관련성 평가 프롬프트
        self.retrieval_grader_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """당신은 검색된 문서가 사용자 질문과 관련이 있는지 평가하는 평가자입니다. 
특허 문서가 질문 해결에 유용한 정보를 포함하고 있으면 관련이 있다고 판단하세요. \n 
키워드 매칭에만 의존하지 말고, 의미적 관련성을 중시하여 평가하세요. \n
이진 점수 'yes' 또는 'no'만 제공하여 문서가 질문과 관련이 있는지 나타내세요.
검색된 문서: \n\n {document} \n\n
사용자 질문: {question} \n\n
이 문서가 사용자 질문과 관련이 있나요?
""",
                ),
                (
                    "human",
                    "검색된 문서: {document}\n\n사용자 질문: {question}\n\n이 문서가 사용자 질문과 관련이 있나요?",
                ),
            ]
        )

        # 질문 재작성 프롬프트
        self.query_rewriter_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """당신은 특허 도메인에 특화된 질문 재작성 전문가입니다. 
사용자의 원래 질문을 분석하여 더 나은 검색 결과를 얻을 수 있도록 질문을 개선하세요. \n
다음 가이드라인을 따르세요:
1. 특허 관련 기술 용어를 포함하세요
2. 검색 범위를 구체화하세요  
3. 유사한 의미의 다양한 표현을 사용하세요
4. 한국어로 자연스럽게 작성하세요 \n
개선된 질문만 출력하세요.
원래 질문: {question} \n
개선된 질문:
""",
                ),
                ("human", "원래 질문: {question}\n\n개선된 질문:"),
            ]
        )

        # 답변 생성 프롬프트
        self.rag_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """당신은 특허 분야 전문가이며, 제공된 특허 문서를 바탕으로 사용자의 질문에 정확하고 유용한 답변을 제공합니다. \n
다음 가이드라인을 따르세요:
1. 제공된 문서의 정보만을 사용하세요
2. 구체적인 특허 번호나 기술 내용을 인용하세요  
3. 명확하고 이해하기 쉽게 설명하세요
4. 확실하지 않은 정보는 추측하지 마세요
5. 한국어로 자연스럽게 답변하세요
질문: {question} \n 
문서: {context} \n 
답변:
""",
                ),
                ("human", "질문: {question}\n\n문서: {context}\n\n답변:"),
            ]
        )

        # 체인 생성
        self.retrieval_grader = (
            self.retrieval_grader_template | self.llm | StrOutputParser()
        )
        self.query_rewriter = (
            self.query_rewriter_template | self.llm | StrOutputParser()
        )
        self.rag_chain = self.rag_template | self.llm | StrOutputParser()

    def retrieve(self, state: GraphState) -> Dict[str, Any]:
        """
        고도화된 문서 검색 (하이브리드 검색 + 재랭킹)

        Args:
            state: 현재 그래프 상태

        Returns:
            Dict: 업데이트된 상태
        """
        logger.info("---고도화된 문서 검색 시작---")

        question = state["question"]
        process_log = state.get("process_log", [])

        try:
            # 하이브리드 검색 수행 (Jina reranking 포함)
            if self.enable_hybrid_search and self.hybrid_manager:
                logger.info("하이브리드 검색 + Jina 재랭킹 수행 중...")
                search_results = self.hybrid_manager.search(
                    query=question,
                    n_results=self.rerank_top_k,  # 최종 반환할 문서 수
                    vector_n_results=30,
                    bm25_n_results=30,
                    enable_reranking=self.enable_reranking,  # Jina 재랭킹 활성화
                    rerank_top_k=self.rerank_top_k,  # 재랭킹할 문서 수
                )

                # 하이브리드 검색 결과를 문서 형식으로 변환
                documents = []
                for result in search_results:
                    if hasattr(result, "content") and hasattr(result, "metadata"):
                        # SearchResult 객체의 경우
                        doc_dict = {
                            "content": result.content,
                            "metadata": result.metadata,
                            "vector_score": getattr(result, "vector_score", 0.0),
                            "bm25_score": getattr(result, "bm25_score", 0.0),
                            "combined_score": getattr(result, "combined_score", 0.0),
                        }

                        # Jina 재랭킹 정보 추가
                        if (
                            hasattr(result, "rerank_score")
                            and result.rerank_score is not None
                        ):
                            doc_dict["rerank_score"] = result.rerank_score
                            doc_dict["original_position"] = getattr(
                                result, "original_position", 0
                            )
                            doc_dict["rerank_position"] = getattr(
                                result, "rerank_position", 0
                            )

                        documents.append(doc_dict)
                    elif isinstance(result, dict):
                        documents.append(result)

                search_method = "하이브리드 검색"
                if self.enable_reranking:
                    search_method += " + Jina 재랭킹"

                process_log.append(f"{search_method} 완료: {len(documents)}개 문서")

                # 재랭킹 스코어 통계 로깅
                if self.enable_reranking and documents:
                    reranked_docs = [doc for doc in documents if "rerank_score" in doc]
                    if reranked_docs:
                        avg_score = sum(
                            doc["rerank_score"] for doc in reranked_docs
                        ) / len(reranked_docs)
                        max_score = max(doc["rerank_score"] for doc in reranked_docs)
                        min_score = min(doc["rerank_score"] for doc in reranked_docs)
                        process_log.append(
                            f"Jina 재랭킹 스코어 - 평균: {avg_score:.3f}, "
                            f"최고: {max_score:.3f}, 최저: {min_score:.3f}"
                        )

            else:
                logger.info("기본 Vector Store 검색 수행 중...")
                search_result = self.vector_store.search_similar(
                    question, n_results=self.rerank_top_k
                )
                documents = []

                # 검색 결과 변환
                if "documents" in search_result and search_result["documents"]:
                    for i, (doc, metadata) in enumerate(
                        zip(search_result["documents"], search_result["metadatas"])
                    ):
                        documents.append({"content": doc, "metadata": metadata or {}})

                process_log.append(f"Vector Store 검색 완료: {len(documents)}개 문서")

            logger.info(f"검색 완료: {len(documents)}개 문서")

        except Exception as e:
            logger.error(f"문서 검색 실패: {e}")
            # 폴백: 기본 검색 시도
            try:
                search_result = self.vector_store.search_similar(question, n_results=5)
                documents = []

                if "documents" in search_result and search_result["documents"]:
                    for i, (doc, metadata) in enumerate(
                        zip(search_result["documents"], search_result["metadatas"])
                    ):
                        documents.append({"content": doc, "metadata": metadata or {}})

                process_log.append(f"폴백 검색 완료: {len(documents)}개 문서")
            except Exception as fallback_error:
                logger.error(f"폴백 검색도 실패: {fallback_error}")
                documents = []
                process_log.append("문서 검색 실패")

        return {
            "documents": documents,
            "question": question,
            "process_log": process_log,
        }

    def grade_documents(self, state: GraphState) -> Dict[str, Any]:
        """
        검색된 문서들의 관련성 평가

        Args:
            state: 현재 그래프 상태

        Returns:
            Dict: 업데이트된 상태
        """
        logger.info("---문서 관련성 평가 시작---")

        question = state["question"]
        documents = state["documents"]
        process_log = state.get("process_log", [])

        # 관련성 있는 문서들을 저장할 리스트
        filtered_docs = []

        for i, doc in enumerate(documents):
            try:
                # 문서 텍스트 추출
                doc_content = doc.get("content", "")
                if isinstance(doc_content, dict):
                    doc_content = doc_content.get("page_content", str(doc_content))
                elif not isinstance(doc_content, str):
                    doc_content = str(doc_content)

                # 너무 짧은 문서는 건너뛰기
                if len(doc_content.strip()) < 50:
                    continue

                # 관련성 평가
                score = self.retrieval_grader.invoke(
                    {"question": question, "document": doc_content[:2000]}  # 길이 제한
                )

                grade = score.strip().lower()

                # 재랭킹 스코어가 있으면 추가 고려
                rerank_score = doc.get("rerank_score", 0.5)

                # yes 판정이거나 재랭킹 스코어가 높으면 관련 문서로 판단
                if "yes" in grade or rerank_score > 0.7:
                    filtered_docs.append(doc)
                    logger.debug(
                        f"문서 {i+1}: 관련됨 (grade: {grade}, rerank: {rerank_score:.3f})"
                    )
                else:
                    logger.debug(
                        f"문서 {i+1}: 관련없음 (grade: {grade}, rerank: {rerank_score:.3f})"
                    )

            except Exception as e:
                logger.warning(f"문서 {i+1} 평가 실패: {e}")
                # 재랭킹 스코어가 높으면 포함
                if doc.get("rerank_score", 0) > 0.6:
                    filtered_docs.append(doc)

        # 재랭킹 스코어가 있는 경우 스코어 순으로 정렬
        if filtered_docs and any(doc.get("rerank_score") for doc in filtered_docs):
            filtered_docs.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        process_log.append(
            f"문서 관련성 평가 완료: {len(documents)}개 중 {len(filtered_docs)}개 관련"
        )

        logger.info(f"관련 문서 필터링: {len(documents)} -> {len(filtered_docs)}")

        return {
            "documents": filtered_docs,
            "question": question,
            "process_log": process_log,
        }

    def generate(self, state: GraphState) -> Dict[str, Any]:
        """
        관련 문서를 바탕으로 답변 생성

        Args:
            state: 현재 그래프 상태

        Returns:
            Dict: 업데이트된 상태
        """
        logger.info("---답변 생성 시작---")

        question = state["question"]
        documents = state["documents"]
        process_log = state.get("process_log", [])

        try:
            if not documents:
                generation = (
                    "죄송합니다. 제공된 특허 데이터베이스에서 "
                    "관련된 정보를 찾을 수 없습니다."
                )
            else:
                # 문서 내용 결합 (상위 5개 문서만 사용)
                top_docs = documents[:5]
                context = ""

                for i, doc in enumerate(top_docs, 1):
                    doc_content = doc.get("content", "")
                    if isinstance(doc_content, dict):
                        doc_content = doc_content.get("page_content", str(doc_content))

                    metadata = doc.get("metadata", {})
                    patent_number = metadata.get("patent_number", f"문서-{i}")

                    # 재랭킹 스코어 표시
                    rerank_score = doc.get("rerank_score")
                    score_info = (
                        f" (관련도: {rerank_score:.3f})" if rerank_score else ""
                    )

                    context += (
                        f"\n\n[{patent_number}{score_info}]\n{doc_content[:1500]}"
                    )

                # RAG 체인으로 답변 생성
                generation = self.rag_chain.invoke(
                    {"context": context, "question": question}
                )

                process_log.append(f"답변 생성 완료 (사용 문서: {len(top_docs)}개)")

        except Exception as e:
            logger.error(f"답변 생성 실패: {e}")
            generation = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
            process_log.append(f"답변 생성 실패: {str(e)}")

        logger.info("답변 생성 완료")

        return {
            "question": question,
            "documents": documents,
            "generation": generation,
            "process_log": process_log,
        }

    def transform_query(self, state: GraphState) -> Dict[str, Any]:
        """
        검색 성능 개선을 위한 질문 재작성

        Args:
            state: 현재 그래프 상태

        Returns:
            Dict: 업데이트된 상태
        """
        logger.info("---질문 재작성 시작---")

        question = state["question"]
        process_log = state.get("process_log", [])
        retry_count = state.get("retry_count", 0)

        try:
            # 질문 재작성
            better_question = self.query_rewriter.invoke({"question": question})
            better_question = better_question.strip()

            # 재작성된 질문이 너무 짧거나 원본과 동일하면 다른 방식 시도
            if len(better_question) < 10 or better_question == question:
                # 특허 도메인 키워드 추가
                patent_keywords = ["특허", "기술", "발명", "구현", "방법", "시스템"]
                better_question = (
                    f"{question} {patent_keywords[retry_count % len(patent_keywords)]}"
                )

            process_log.append(f"질문 재작성: '{question}' -> '{better_question}'")
            logger.info(f"질문 재작성 완료: {better_question[:100]}...")

        except Exception as e:
            logger.error(f"질문 재작성 실패: {e}")
            better_question = question
            process_log.append(f"질문 재작성 실패: {str(e)}")

        return {
            **state,
            "question": better_question,
            "query_rewritten": True,
            "retry_count": retry_count + 1,
            "process_log": process_log,
        }

    def get_search_stats(self) -> Dict[str, Any]:
        """검색 시스템 통계 정보 반환"""
        stats = {
            "hybrid_search_enabled": self.enable_hybrid_search,
            "reranking_enabled": self.enable_reranking,
            "model_name": self.model_name,
            "rerank_top_k": self.rerank_top_k,
        }

        if self.hybrid_manager:
            stats.update(
                {
                    "hybrid_vector_weight": getattr(
                        self.hybrid_manager, "vector_weight", 0.6
                    ),
                    "hybrid_bm25_weight": getattr(
                        self.hybrid_manager, "bm25_weight", 0.4
                    ),
                }
            )

        if self.reranker:
            reranker_info = self.reranker.get_model_info()
            stats["reranker_model"] = reranker_info.get("model_name")
            stats["reranker_loaded"] = reranker_info.get("is_loaded", False)

        return stats
