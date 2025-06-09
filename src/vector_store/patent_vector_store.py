#!/usr/bin/env python3
"""
특허 벡터 스토어 모듈

Chroma 벡터 데이터베이스를 사용하여 특허 청크를 저장하고
유사도 기반 검색을 수행합니다.
"""

import os
import logging
import json
import uuid
from typing import List, Dict, Optional, Union, Any, Tuple
from dataclasses import asdict
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from .embedding_manager import EmbeddingManager, ChromaEmbeddingFunction
from .text_preprocessor import PatentChunkPreprocessor
from .performance_optimizer import PerformanceMonitor, performance_monitor
from ..patent_search.patent_parser import DocumentChunk, Patent

# 로거 설정
logger = logging.getLogger(__name__)


class PatentVectorStore:
    """
    특허 데이터를 위한 Chroma 벡터 스토어 관리자

    주요 기능:
    - 특허 청크 저장 및 관리
    - 유사도 기반 검색
    - 메타데이터 필터링
    - 배치 처리
    - 영구 저장 및 로드
    - CRUD 작업
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "patent_chunks",
        embedding_manager: Optional[EmbeddingManager] = None,
        reset_collection: bool = False,
        enable_preprocessing: bool = True,
        enable_performance_monitoring: bool = True,
    ):
        """
        특허 벡터 스토어 초기화

        Args:
            persist_directory: 영구 저장 디렉토리
            collection_name: 컬렉션 이름
            embedding_manager: 임베딩 매니저 (None이면 기본값 사용)
            reset_collection: 기존 컬렉션 삭제 여부
            enable_preprocessing: 텍스트 전처리 활성화 여부
            enable_performance_monitoring: 성능 모니터링 활성화 여부
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.enable_preprocessing = enable_preprocessing
        self.enable_performance_monitoring = enable_performance_monitoring

        # 임베딩 매니저 설정
        if embedding_manager is None:
            self.embedding_manager = EmbeddingManager()
        else:
            self.embedding_manager = embedding_manager

        # 전처리기 설정
        if self.enable_preprocessing:
            self.preprocessor = PatentChunkPreprocessor()
        else:
            self.preprocessor = None

        # 성능 모니터 설정
        if self.enable_performance_monitoring:
            self.performance_monitor = PerformanceMonitor()
        else:
            self.performance_monitor = None

        # Chroma 클라이언트 초기화
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # 임베딩 함수 설정
        self.embedding_function = ChromaEmbeddingFunction(self.embedding_manager)

        # 컬렉션 초기화
        self._init_collection(reset_collection)

        logger.info(f"벡터 스토어 초기화 완료: {persist_directory}/{collection_name}")

    def _init_collection(self, reset: bool = False):
        """컬렉션 초기화"""
        try:
            if reset:
                # 기존 컬렉션 삭제
                try:
                    existing_collections = [
                        col.name for col in self.client.list_collections()
                    ]
                    if self.collection_name in existing_collections:
                        self.client.delete_collection(self.collection_name)
                        logger.info(f"기존 컬렉션 삭제: {self.collection_name}")
                except Exception as e:
                    logger.warning(f"컬렉션 삭제 중 오류 (무시): {e}")

            # 컬렉션 목록 확인 후 생성 또는 로드
            existing_collections = [col.name for col in self.client.list_collections()]

            if self.collection_name in existing_collections:
                # 기존 컬렉션 로드
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                )
                logger.info(f"기존 컬렉션 로드: {self.collection_name}")
            else:
                # 새 컬렉션 생성
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": "cosine"},  # 코사인 유사도
                )
                logger.info(f"새 컬렉션 생성: {self.collection_name}")

        except Exception as e:
            logger.error(f"컬렉션 초기화 실패: {e}")
            raise

    def add_document_chunk(self, chunk: DocumentChunk) -> bool:
        """
        단일 문서 청크 추가

        Args:
            chunk: 추가할 문서 청크

        Returns:
            bool: 성공 여부
        """
        import time

        start_time = time.time()
        success = True

        try:
            # 전처리 적용 (활성화된 경우)
            if self.preprocessor:
                processed_chunk = self.preprocessor.preprocess_chunk(chunk)
            else:
                processed_chunk = chunk

            # 메타데이터 준비 (Chroma는 문자열, 숫자, 불린만 지원)
            metadata = self._prepare_metadata(processed_chunk.metadata)

            self.collection.add(
                ids=[processed_chunk.chunk_id],
                documents=[processed_chunk.content],
                metadatas=[metadata],
            )

            logger.debug(f"청크 추가 성공: {processed_chunk.chunk_id}")
            return True

        except Exception as e:
            success = False
            logger.error(f"청크 추가 실패 {chunk.chunk_id}: {e}")
            return False

        finally:
            if self.performance_monitor:
                duration = time.time() - start_time
                self.performance_monitor.record_operation(
                    "add_single_chunk", duration, success, {"chunk_id": chunk.chunk_id}
                )

    def add_document_chunks_batch(self, chunks: List[DocumentChunk]) -> Tuple[int, int]:
        """
        여러 문서 청크를 배치로 추가

        Args:
            chunks: 추가할 청크 리스트

        Returns:
            Tuple[int, int]: (성공 개수, 실패 개수)
        """
        import time

        start_time = time.time()

        if not chunks:
            return 0, 0

        logger.info(f"배치 청크 추가 시작: {len(chunks)}개")

        success_count = 0
        error_count = 0

        # 전처리 적용 (활성화된 경우)
        if self.preprocessor:
            logger.debug("배치 전처리 적용 중...")
            processed_chunks = self.preprocessor.preprocess_chunks_batch(chunks)
        else:
            processed_chunks = chunks

        # 데이터 준비
        ids = []
        documents = []
        metadatas = []

        for chunk in processed_chunks:
            try:
                ids.append(chunk.chunk_id)
                documents.append(chunk.content)
                metadatas.append(self._prepare_metadata(chunk.metadata))
            except Exception as e:
                logger.warning(f"청크 준비 실패 {chunk.chunk_id}: {e}")
                error_count += 1

        # 배치 추가
        batch_success = True
        if ids:
            try:
                self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
                success_count = len(ids)
                logger.info(f"배치 추가 성공: {success_count}개")
            except Exception as e:
                batch_success = False
                logger.error(f"배치 추가 실패: {e}")
                error_count += len(ids)

        # 성능 모니터링 기록
        if self.performance_monitor:
            duration = time.time() - start_time
            self.performance_monitor.record_operation(
                "add_batch_chunks",
                duration,
                batch_success,
                {
                    "chunk_count": len(chunks),
                    "success_count": success_count,
                    "error_count": error_count,
                },
            )

        return success_count, error_count

    def add_patent_chunks(
        self, patent: Patent, chunks: List[DocumentChunk]
    ) -> Dict[str, Any]:
        """
        특허 전체 청크 추가 (특허 정보 포함)

        Args:
            patent: 특허 객체
            chunks: 특허의 청크 리스트

        Returns:
            Dict: 추가 결과
        """
        logger.info(f"특허 청크 추가: {patent.patent_number} ({len(chunks)}개)")

        # 각 청크에 특허 정보 추가
        enhanced_chunks = []
        for chunk in chunks:
            # 메타데이터에 특허 정보 추가
            enhanced_metadata = chunk.metadata.copy()
            enhanced_metadata.update(
                {
                    "patent_number": patent.patent_number,
                    "patent_id": patent.patent_id,
                    "title": patent.title[:200],  # 길이 제한
                    "filing_date": patent.filing_date,
                    "publication_date": patent.publication_date,
                    "applicant": patent.applicant[:100] if patent.applicant else "",
                    "inventor": patent.inventor[:100] if patent.inventor else "",
                }
            )

            enhanced_chunk = DocumentChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                section=chunk.section,
                metadata=enhanced_metadata,
            )
            enhanced_chunks.append(enhanced_chunk)

        # 배치 추가
        success, errors = self.add_document_chunks_batch(enhanced_chunks)

        result = {
            "patent_number": patent.patent_number,
            "total_chunks": len(chunks),
            "success_count": success,
            "error_count": errors,
            "success_rate": success / len(chunks) if chunks else 0,
        }

        logger.info(f"특허 추가 완료: {result}")
        return result

    def search_similar(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        include_distances: bool = True,
    ) -> Dict[str, Any]:
        """
        유사도 기반 검색

        Args:
            query: 검색 쿼리
            n_results: 반환할 결과 수
            where: 메타데이터 필터 조건
            include_distances: 거리 정보 포함 여부

        Returns:
            Dict: 검색 결과
        """
        import time

        start_time = time.time()
        search_success = True

        try:
            include_list = ["documents", "metadatas"]
            if include_distances:
                include_list.append("distances")

            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                include=include_list,
            )

            # 결과 정리
            formatted_results = {
                "query": query,
                "total_results": len(results["ids"][0]) if results["ids"] else 0,
                "results": [],
            }

            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    result_item = {
                        "chunk_id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                    }

                    if include_distances and "distances" in results:
                        result_item["distance"] = results["distances"][0][i]
                        result_item["similarity"] = (
                            1 - results["distances"][0][i]
                        )  # 코사인 유사도

                    formatted_results["results"].append(result_item)

            logger.debug(
                f"검색 완료: '{query}' → {formatted_results['total_results']}개 결과"
            )
            return formatted_results

        except Exception as e:
            search_success = False
            logger.error(f"검색 실패: {e}")
            return {"query": query, "total_results": 0, "results": [], "error": str(e)}

        finally:
            # 성능 모니터링 기록
            if self.performance_monitor:
                duration = time.time() - start_time
                self.performance_monitor.record_operation(
                    "search_similar",
                    duration,
                    search_success,
                    {"query_length": len(query), "n_results": n_results},
                )

    def search_by_patent(
        self, patent_number: str, section: Optional[str] = None, n_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        특정 특허의 청크 검색

        Args:
            patent_number: 특허 번호
            section: 섹션 필터 (선택적)
            n_results: 최대 결과 수

        Returns:
            List[Dict]: 검색 결과
        """
        where_clause = {"patent_number": patent_number}
        if section:
            where_clause["section"] = section

        try:
            results = self.collection.get(
                where=where_clause, limit=n_results, include=["documents", "metadatas"]
            )

            formatted_results = []
            if results["ids"]:
                for i in range(len(results["ids"])):
                    formatted_results.append(
                        {
                            "chunk_id": results["ids"][i],
                            "content": results["documents"][i],
                            "metadata": results["metadatas"][i],
                        }
                    )

            logger.debug(
                f"특허 검색 완료: {patent_number} → {len(formatted_results)}개"
            )
            return formatted_results

        except Exception as e:
            logger.error(f"특허 검색 실패: {e}")
            return []

    def delete_patent(self, patent_number: str) -> bool:
        """
        특정 특허의 모든 청크 삭제

        Args:
            patent_number: 삭제할 특허 번호

        Returns:
            bool: 성공 여부
        """
        try:
            # 해당 특허의 모든 청크 ID 찾기
            results = self.collection.get(
                where={"patent_number": patent_number}, include=[]  # ID만 필요
            )

            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(
                    f"특허 삭제 완료: {patent_number} ({len(results['ids'])}개 청크)"
                )
                return True
            else:
                logger.warning(f"삭제할 특허가 없음: {patent_number}")
                return False

        except Exception as e:
            logger.error(f"특허 삭제 실패 {patent_number}: {e}")
            return False

    def update_chunk(
        self, chunk_id: str, content: str, metadata: Dict[str, Any]
    ) -> bool:
        """
        특정 청크 업데이트

        Args:
            chunk_id: 청크 ID
            content: 새 내용
            metadata: 새 메타데이터

        Returns:
            bool: 성공 여부
        """
        try:
            prepared_metadata = self._prepare_metadata(metadata)

            self.collection.update(
                ids=[chunk_id], documents=[content], metadatas=[prepared_metadata]
            )

            logger.debug(f"청크 업데이트 성공: {chunk_id}")
            return True

        except Exception as e:
            logger.error(f"청크 업데이트 실패 {chunk_id}: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        벡터 스토어 통계 정보

        Returns:
            Dict: 통계 정보
        """
        try:
            count = self.collection.count()

            # 임베딩 매니저 통계
            embedding_stats = self.embedding_manager.get_stats()

            return {
                "collection_name": self.collection_name,
                "total_chunks": count,
                "persist_directory": self.persist_directory,
                "embedding_stats": embedding_stats,
            }

        except Exception as e:
            logger.error(f"통계 조회 실패: {e}")
            return {"error": str(e)}

    def _prepare_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Chroma용 메타데이터 준비 (문자열, 숫자, 불린만 허용)

        Args:
            metadata: 원본 메타데이터

        Returns:
            Dict: 정제된 메타데이터
        """
        prepared = {}

        for key, value in metadata.items():
            if value is None:
                continue
            elif isinstance(value, (str, int, float, bool)):
                prepared[key] = value
            elif isinstance(value, (list, dict)):
                # 복잡한 타입은 JSON 문자열로 변환
                prepared[key] = json.dumps(value, ensure_ascii=False)
            else:
                # 기타 타입은 문자열로 변환
                prepared[key] = str(value)

        return prepared

    def reset_collection(self):
        """컬렉션 완전 초기화"""
        self._init_collection(reset=True)
        logger.info("컬렉션 초기화 완료")

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        벡터 스토어의 모든 문서 반환 (하이브리드 검색용)

        Returns:
            List[Dict]: 모든 문서 리스트
        """
        try:
            # 컬렉션에서 모든 문서 가져오기
            all_data = self.collection.get(include=["documents", "metadatas", "ids"])

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

            logger.debug(f"전체 문서 추출 완료: {len(documents)}개")
            return documents

        except Exception as e:
            logger.error(f"전체 문서 추출 실패: {e}")
            return []

    def close(self):
        """리소스 정리"""
        # Chroma는 자동으로 영구 저장됨
        logger.info("벡터 스토어 종료")
