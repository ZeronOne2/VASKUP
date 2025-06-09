"""
LangGraph CRAG 시스템의 문서 평가 모듈

검색된 문서의 관련성을 평가하는 LLM 기반 그레이더
"""

import os
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class GradeDocuments(BaseModel):
    """문서 관련성 평가 결과"""

    binary_score: str = Field(
        description="문서가 질문과 관련이 있는지 여부, 'yes' 또는 'no'"
    )
    reasoning: str = Field(description="판단 근거 설명")


class DocumentGrader:
    """
    검색된 문서의 관련성을 평가하는 LLM 기반 그레이더

    한국어 특허 도메인에 특화된 평가 로직을 제공합니다.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        """
        DocumentGrader 초기화

        Args:
            model_name: 사용할 LLM 모델명
            temperature: 모델 temperature 설정
        """
        self.model_name = model_name
        self.temperature = temperature

        # LLM 초기화
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        # 구조화된 출력 설정
        self.structured_llm_grader = self.llm.with_structured_output(GradeDocuments)

        # 한국어 특허 도메인 특화 프롬프트
        self.system_prompt = """당신은 한국어 특허 문서의 관련성을 평가하는 전문가입니다.

사용자의 질문과 검색된 특허 문서를 비교하여 관련성을 판단하세요.

평가 기준:
1. 문서가 질문의 키워드나 의미적 내용과 관련이 있는지
2. 특허의 기술 분야가 질문과 일치하는지  
3. 문서에서 질문에 대한 답변을 찾을 수 있는지

관련성이 있다면 'yes', 없다면 'no'로 판단하세요.
엄격할 필요는 없으며, 잘못된 검색 결과를 필터링하는 것이 목표입니다."""

        self.grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "검색된 문서: \n\n{document}\n\n사용자 질문: {question}"),
            ]
        )

        # 그레이더 체인 생성
        self.grader_chain = self.grade_prompt | self.structured_llm_grader

        logger.info(f"DocumentGrader 초기화 완료: {model_name}")

    def grade_document(self, question: str, document: str) -> Dict[str, Any]:
        """
        단일 문서의 관련성을 평가합니다.

        Args:
            question: 사용자 질문
            document: 평가할 문서 내용

        Returns:
            Dict: 평가 결과 {'binary_score': str, 'reasoning': str}
        """
        try:
            result = self.grader_chain.invoke(
                {"question": question, "document": document}
            )

            return {"binary_score": result.binary_score, "reasoning": result.reasoning}

        except Exception as e:
            logger.error(f"문서 평가 실패: {e}")
            return {"binary_score": "no", "reasoning": f"평가 중 오류 발생: {str(e)}"}

    def grade_documents_batch(self, question: str, documents: list) -> list:
        """
        여러 문서를 배치로 평가합니다.

        Args:
            question: 사용자 질문
            documents: 평가할 문서 리스트

        Returns:
            List[Dict]: 각 문서의 평가 결과 리스트
        """
        results = []

        for i, doc in enumerate(documents):
            logger.debug(f"문서 {i+1}/{len(documents)} 평가 중...")

            # 문서 내용 추출 (dict이면 content 키에서, 아니면 문자열로 처리)
            if isinstance(doc, dict):
                doc_content = doc.get("content", str(doc))
            else:
                doc_content = str(doc)

            grade_result = self.grade_document(question, doc_content)

            results.append(
                {"document_index": i, "document": doc, "grade": grade_result}
            )

        return results
