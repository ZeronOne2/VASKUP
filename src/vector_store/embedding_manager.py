#!/usr/bin/env python3
"""
임베딩 매니저 모듈

OpenAI의 text-embedding-3-small 모델을 사용하여
특허 텍스트를 1536차원 벡터로 변환합니다.
"""

import os
import logging
import time
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import openai
from openai import OpenAI

# 로거 설정
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """임베딩 결과 데이터 클래스"""

    text: str
    embedding: List[float]
    model: str
    dimensions: int
    token_count: int
    processing_time: float


class EmbeddingManager:
    """
    OpenAI 임베딩 모델을 사용한 텍스트 벡터화 관리자

    주요 기능:
    - 텍스트를 1536차원 벡터로 변환
    - 배치 처리 지원 (최대 100개 텍스트)
    - 속도 제한 및 오류 처리
    - 토큰 사용량 모니터링
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        임베딩 매니저 초기화

        Args:
            api_key: OpenAI API 키 (None이면 환경변수에서 읽음)
            model: 사용할 임베딩 모델명
            max_retries: 최대 재시도 횟수
            retry_delay: 재시도 간격 (초)
        """
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.dimensions = 1536  # text-embedding-3-small 기본 차원

        # OpenAI 클라이언트 초기화
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

        # 통계 추적
        self.total_tokens_used = 0
        self.total_requests = 0
        self.total_errors = 0

        logger.info(f"임베딩 매니저 초기화: {self.model} ({self.dimensions}차원)")

    def create_embedding(self, text: str) -> EmbeddingResult:
        """
        단일 텍스트의 임베딩 생성

        Args:
            text: 임베딩할 텍스트

        Returns:
            EmbeddingResult: 임베딩 결과

        Raises:
            Exception: 임베딩 생성 실패시
        """
        start_time = time.time()

        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model, input=text, encoding_format="float"
                )

                embedding_data = response.data[0]
                processing_time = time.time() - start_time

                # 통계 업데이트
                self.total_tokens_used += response.usage.total_tokens
                self.total_requests += 1

                result = EmbeddingResult(
                    text=text,
                    embedding=embedding_data.embedding,
                    model=self.model,
                    dimensions=len(embedding_data.embedding),
                    token_count=response.usage.total_tokens,
                    processing_time=processing_time,
                )

                logger.debug(
                    f"임베딩 생성 성공: {len(text)} 문자 → {result.dimensions}차원"
                )
                return result

            except Exception as e:
                self.total_errors += 1
                logger.warning(
                    f"임베딩 생성 시도 {attempt + 1}/{self.max_retries} 실패: {e}"
                )

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2**attempt))  # 지수적 백오프
                else:
                    logger.error(f"임베딩 생성 최종 실패: {text[:100]}...")
                    raise e

    def create_embeddings_batch(
        self, texts: List[str], batch_size: int = 100
    ) -> List[EmbeddingResult]:
        """
        여러 텍스트의 임베딩을 배치로 생성

        Args:
            texts: 임베딩할 텍스트 리스트
            batch_size: 배치 크기 (OpenAI 제한: 2048개, 권장: 100개)

        Returns:
            List[EmbeddingResult]: 임베딩 결과 리스트
        """
        if not texts:
            return []

        logger.info(f"배치 임베딩 시작: {len(texts)}개 텍스트, 배치크기 {batch_size}")

        all_results = []

        # 배치별로 처리
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_start = time.time()

            logger.debug(
                f"배치 {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} 처리 중..."
            )

            for attempt in range(self.max_retries):
                try:
                    response = self.client.embeddings.create(
                        model=self.model, input=batch_texts, encoding_format="float"
                    )

                    batch_time = time.time() - batch_start

                    # 결과 변환
                    batch_results = []
                    for j, embedding_data in enumerate(response.data):
                        result = EmbeddingResult(
                            text=batch_texts[j],
                            embedding=embedding_data.embedding,
                            model=self.model,
                            dimensions=len(embedding_data.embedding),
                            token_count=response.usage.total_tokens
                            // len(batch_texts),  # 근사치
                            processing_time=batch_time / len(batch_texts),
                        )
                        batch_results.append(result)

                    all_results.extend(batch_results)

                    # 통계 업데이트
                    self.total_tokens_used += response.usage.total_tokens
                    self.total_requests += 1

                    logger.debug(
                        f"배치 완료: {len(batch_texts)}개 → {batch_time:.2f}초"
                    )
                    break

                except Exception as e:
                    self.total_errors += 1
                    logger.warning(
                        f"배치 시도 {attempt + 1}/{self.max_retries} 실패: {e}"
                    )

                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2**attempt))
                    else:
                        logger.error(f"배치 최종 실패: 인덱스 {i}-{i+len(batch_texts)}")
                        raise e

            # 배치 간 딜레이 (API 속도 제한 방지)
            if i + batch_size < len(texts):
                time.sleep(0.1)

        logger.info(f"배치 임베딩 완료: {len(all_results)}개 결과")
        return all_results

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """
        임베딩 매니저 사용 통계 반환

        Returns:
            Dict: 통계 정보
        """
        return {
            "model": self.model,
            "dimensions": self.dimensions,
            "total_requests": self.total_requests,
            "total_tokens_used": self.total_tokens_used,
            "total_errors": self.total_errors,
            "error_rate": self.total_errors / max(1, self.total_requests),
            "avg_tokens_per_request": self.total_tokens_used
            / max(1, self.total_requests),
        }

    def reset_stats(self):
        """통계 초기화"""
        self.total_tokens_used = 0
        self.total_requests = 0
        self.total_errors = 0
        logger.info("임베딩 매니저 통계 초기화")


# Chroma용 커스텀 임베딩 함수
class ChromaEmbeddingFunction:
    """
    Chroma 벡터 스토어와 호환되는 임베딩 함수 래퍼
    """

    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        # Chroma가 요구하는 name (메서드여야 함)
        self._name_value = f"custom_openai_{embedding_manager.model}"

    def name(self) -> str:
        """Chroma가 요구하는 name 메서드"""
        return self._name_value

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Chroma가 요구하는 임베딩 함수 인터페이스

        Args:
            input: 텍스트 리스트

        Returns:
            List[List[float]]: 임베딩 벡터 리스트
        """
        # 입력 타입 확인 및 변환
        if not isinstance(input, list):
            input = [str(input)]

        # 빈 입력 처리
        if not input:
            return []

        # 모든 요소를 문자열로 변환
        text_inputs = [str(item) for item in input]

        try:
            results = self.embedding_manager.create_embeddings_batch(text_inputs)
            embeddings = [result.embedding for result in results]

            # 결과 검증
            if len(embeddings) != len(text_inputs):
                logger.warning(
                    f"입력({len(text_inputs)})과 출력({len(embeddings)}) 길이 불일치"
                )

            return embeddings

        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            # 실패 시 빈 벡터 반환 (차원 수는 embedding_manager의 차원과 동일)
            return [[0.0] * self.embedding_manager.dimensions for _ in text_inputs]
