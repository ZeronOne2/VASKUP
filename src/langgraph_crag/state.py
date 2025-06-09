"""
LangGraph CRAG 시스템의 상태 정의

그래프 노드들 간에 공유되는 상태를 정의합니다.
"""

from typing import List, Optional
from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Corrective RAG 그래프의 상태를 나타냅니다.

    Attributes:
        question: 사용자 질문 (원본)
        documents: 검색된 문서 리스트
        generation: LLM이 생성한 답변
        web_search: 웹 검색 필요 여부 ("Yes" 또는 "No")
        query_rewritten: 질문이 재작성되었는지 여부
        retry_count: 재시도 횟수
        process_log: 처리 과정 로그
    """

    question: str
    documents: List[str]
    generation: Optional[str]
    web_search: str
    query_rewritten: bool
    retry_count: int
    process_log: List[str]
