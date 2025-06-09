"""
LangGraph 기반 Corrective RAG (CRAG) 파이프라인

한국어 특허 분석에 특화된 자기 수정형 RAG 시스템의 메인 클래스 (Vector Store 전용)
"""

import logging
from typing import Dict, Any, Optional
from langgraph.graph import END, StateGraph, START

from .state import GraphState
from .nodes import CRAGNodes

logger = logging.getLogger(__name__)


class CorrectiveRAGPipeline:
    """
    LangGraph 기반 Corrective RAG 파이프라인 (Vector Store 전용)

    한국어 특허 도메인에 특화된 자기 수정형 RAG 시스템으로,
    Vector Store 내 특허 문서만을 사용하여 문서 관련성 평가 및 질문 재작성을 수행합니다.
    """

    def __init__(
        self,
        vector_store,
        model_name: str = "gpt-4o-mini",
        max_retries: int = 2,
        enable_hybrid_search: bool = True,
        enable_reranking: bool = True,
        hybrid_vector_weight: float = 0.6,
        rerank_top_k: int = 10,
        jina_api_key: str = None,
        jina_model: str = "jina-reranker-v2-base-multilingual",
    ):
        """
        CorrectiveRAGPipeline 초기화

        Args:
            vector_store: 특허 벡터 스토어 인스턴스
            model_name: 사용할 LLM 모델명
            max_retries: 최대 재시도 횟수
            enable_hybrid_search: 하이브리드 검색 활성화 여부
            enable_reranking: Jina AI 재랭킹 활성화 여부
            hybrid_vector_weight: 하이브리드 검색에서 Vector Search 가중치
            rerank_top_k: 재랭킹할 상위 문서 수
            jina_api_key: Jina AI API 키
            jina_model: 사용할 Jina reranker 모델
        """
        self.vector_store = vector_store
        self.model_name = model_name
        self.max_retries = max_retries
        self.enable_hybrid_search = enable_hybrid_search
        self.enable_reranking = enable_reranking

        # 노드 함수들 초기화
        self.nodes = CRAGNodes(
            vector_store,
            model_name,
            enable_hybrid_search=enable_hybrid_search,
            enable_reranking=enable_reranking,
            hybrid_vector_weight=hybrid_vector_weight,
            rerank_top_k=rerank_top_k,
            jina_api_key=jina_api_key,
            jina_model=jina_model,
        )

        # 그래프 컴파일
        self.app = self._build_graph()

        logger.info(
            f"CorrectiveRAGPipeline 초기화 완료 - 모델: {model_name}, "
            f"하이브리드검색: {enable_hybrid_search}, Jina 재랭킹: {enable_reranking}"
        )

    def _build_graph(self) -> StateGraph:
        """LangGraph 워크플로우 구성 (웹 검색 제외)"""

        # StateGraph 생성
        workflow = StateGraph(GraphState)

        # 노드 추가 (웹 검색 노드 제외)
        workflow.add_node("retrieve", self.nodes.retrieve)
        workflow.add_node("grade_documents", self.nodes.grade_documents)
        workflow.add_node("generate", self.nodes.generate)
        workflow.add_node("transform_query", self.nodes.transform_query)

        # 엣지 설정
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")

        # 조건부 엣지: 문서 관련성에 따른 분기
        workflow.add_conditional_edges(
            "grade_documents",
            self._decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )

        # 질문 재작성 후 다시 검색으로 돌아가는 루프
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_edge("generate", END)

        # 그래프 컴파일
        return workflow.compile()

    def _decide_to_generate(self, state: GraphState) -> str:
        """
        문서 평가 결과에 따라 다음 액션 결정 (웹 검색 제외)

        Args:
            state: 현재 그래프 상태

        Returns:
            str: 다음 노드 이름
        """
        logger.info("---액션 결정 중 (Vector Store 전용)---")

        filtered_documents = state.get("documents", [])
        retry_count = state.get("retry_count", 0)

        # 재시도 횟수 초과 시 강제로 생성
        if retry_count >= self.max_retries:
            logger.info("최대 재시도 횟수 도달 - 답변 생성으로 진행")
            return "generate"

        # 관련 문서가 부족한 경우 질문 재작성
        if not filtered_documents:
            logger.info("관련 문서 부족 - 질문 재작성 후 재검색")
            return "transform_query"
        else:
            logger.info("관련 문서 존재 - 답변 생성")
            return "generate"

    def process_query(self, question: str) -> Dict[str, Any]:
        """
        질문을 처리하여 답변을 생성합니다.

        Args:
            question: 사용자 질문

        Returns:
            Dict: 처리 결과
        """
        logger.info(f"CRAG 파이프라인 시작 (Vector Store 전용): {question[:100]}...")

        # 초기 상태 설정
        initial_state = {
            "question": question,
            "documents": [],
            "generation": None,
            "web_search": "No",  # 웹 검색 비활성화
            "query_rewritten": False,
            "retry_count": 0,
            "process_log": [f"질문 처리 시작: {question}"],
        }

        try:
            # 그래프 실행
            result = None
            for output in self.app.stream(initial_state):
                result = output

            # 최종 결과 추출
            if result:
                final_state = list(result.values())[0]

                return {
                    "success": True,
                    "question": final_state.get("question", question),
                    "answer": final_state.get(
                        "generation", "답변을 생성할 수 없습니다."
                    ),
                    "documents": final_state.get("documents", []),
                    "process_log": final_state.get("process_log", []),
                    "metadata": {
                        "query_rewritten": final_state.get("query_rewritten", False),
                        "retry_count": final_state.get("retry_count", 0),
                        "web_search_used": False,  # 항상 False
                        "documents_found": len(final_state.get("documents", [])),
                        "vector_store_only": True,  # Vector Store 전용 표시
                    },
                }
            else:
                return {
                    "success": False,
                    "question": question,
                    "answer": "파이프라인 실행 중 오류가 발생했습니다.",
                    "documents": [],
                    "process_log": ["파이프라인 실행 실패"],
                    "metadata": {},
                }

        except Exception as e:
            logger.error(f"CRAG 파이프라인 실행 실패: {e}")
            return {
                "success": False,
                "question": question,
                "answer": f"파이프라인 실행 중 오류 발생: {str(e)}",
                "documents": [],
                "process_log": [f"오류 발생: {str(e)}"],
                "metadata": {"error": str(e)},
            }

    def get_pipeline_info(self) -> Dict[str, Any]:
        """파이프라인 정보 반환"""
        features = [
            "문서 관련성 평가",
            "질문 재작성",
            "자동 재시도",
            "한국어 특허 특화",
        ]

        workflow = [
            "1. 문서 검색",
            "2. 문서 관련성 평가",
            "3. 조건부 분기 (질문 재작성 또는 답변 생성)",
            "4. 질문 재작성 시 재검색",
            "5. 최종 답변 생성",
        ]

        if self.enable_hybrid_search:
            features.append("하이브리드 검색 (Vector + BM25)")
            workflow[0] = "1. 하이브리드 검색 (Vector + BM25)"
        else:
            features.append("Vector Store 검색")
            workflow[0] = "1. Vector Store 검색"

        if self.enable_reranking:
            features.append("Cross-encoder 재랭킹")
            workflow.insert(2, "2-1. Cross-encoder 재랭킹")

        # 검색 시스템 통계 추가
        search_stats = self.nodes.get_search_stats()

        return {
            "pipeline_type": "고도화된 Corrective RAG",
            "model_name": self.model_name,
            "max_retries": self.max_retries,
            "features": features,
            "workflow": workflow,
            "search_configuration": search_stats,
        }
