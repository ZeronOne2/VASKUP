# 벡터 스토어 라이브러리 비교 분석

## 🎯 프로젝트 요구사항
- **특허 청크 저장**: 수만~수십만 개 텍스트 청크
- **메타데이터 지원**: 특허번호, 섹션, 날짜 등
- **유사도 검색**: 효율적인 semantic search
- **배치 처리**: 대량 데이터 일괄 처리
- **지속성**: 영구 저장 및 로드
- **Python 통합**: Streamlit/LangChain 호환성

## 📊 후보 라이브러리 비교

### 1. **Chroma** ⭐ (권장)
**장점**:
- ✅ **간단한 설정**: `pip install chromadb`만으로 설치
- ✅ **메타데이터 필터링**: 강력한 where 조건 지원
- ✅ **Python Native**: Python 우선 설계
- ✅ **LangChain 통합**: 기본 지원
- ✅ **영구 저장**: SQLite 기반 로컬 저장
- ✅ **임베딩 모델 자유도**: OpenAI, HuggingFace 등 호환

**단점**:
- ⚠️ 대규모 프로덕션 환경에서는 성능 제약
- ⚠️ 상대적으로 새로운 라이브러리

**사용 사례**: 연구/개발 환경, 중소규모 애플리케이션

### 2. **FAISS** (Facebook AI)
**장점**:
- ✅ **고성능**: 수백만 벡터 처리 가능
- ✅ **다양한 인덱스**: IVF, HNSW 등 최적화 옵션
- ✅ **검증된 안정성**: Facebook에서 개발/사용

**단점**:
- ❌ **메타데이터 지원 제한**: 별도 DB 필요
- ❌ **복잡한 설정**: 인덱스 선택 및 튜닝 필요
- ❌ **Python 래퍼**: C++ 기반으로 디버깅 어려움

### 3. **Pinecone** (클라우드)
**장점**:
- ✅ **관리형 서비스**: 인프라 관리 불필요
- ✅ **확장성**: 자동 스케일링
- ✅ **메타데이터 지원**: 풍부한 필터링

**단점**:
- ❌ **비용**: 사용량 기반 과금
- ❌ **네트워크 의존성**: 인터넷 연결 필수
- ❌ **데이터 주권**: 외부 서버 저장

### 4. **Annoy** (Spotify)
**장점**:
- ✅ **메모리 효율성**: 디스크 기반 인덱스
- ✅ **빠른 조회**: 트리 기반 검색

**단점**:
- ❌ **메타데이터 미지원**: ID만 저장
- ❌ **업데이트 불가**: 인덱스 재구축 필요

## 🏆 결론: Chroma 선택

### 선택 이유:
1. **PRD 권장사항** 준수
2. **메타데이터 지원**: 특허 정보 필터링에 필수
3. **LangChain 호환성**: RAG 시스템 구축에 최적
4. **개발 효율성**: 빠른 프로토타이핑 및 테스트
5. **지속성**: 로컬 파일 기반 영구 저장
6. **확장성**: 추후 Chroma 클라우드로 업그레이드 가능

### 구현 계획:
- **로컬 개발**: SQLite 백엔드
- **임베딩 모델**: OpenAI `text-embedding-3-small` (1536 차원)
- **컬렉션 구조**: `patent_chunks` 컬렉션
- **메타데이터**: `patent_id`, `section`, `chunk_index`, `filing_date` 등

## 📦 설치 및 초기 설정

```bash
pip install chromadb
pip install openai  # 임베딩 모델용
```

```python
import chromadb
from chromadb.config import Settings

# 영구 저장소 설정
client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

# 컬렉션 생성
collection = client.create_collection(
    name="patent_chunks",
    embedding_function=openai_ef,  # OpenAI 임베딩
    metadata={"hnsw:space": "cosine"}  # 코사인 유사도
)
```

## 🔄 다음 단계
1. Chroma 설치 및 기본 설정
2. OpenAI 임베딩 모델 통합
3. 특허 데이터 스키마 설계
4. 기본 CRUD 기능 구현 