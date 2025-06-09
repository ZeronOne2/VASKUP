# VASKUP - 특허 분석 RAG 시스템 📄

Advanced Agentic RAG 기반 특허 분석 도구

## 📋 프로젝트 개요

이 시스템은 엑셀 파일에 포함된 특허 번호를 기반으로 특허 정보를 자동 수집하고, AI를 활용하여 특허에 대한 질문에 답변하는 Streamlit 웹 애플리케이션입니다.

## ✨ 주요 기능

- 📤 **파일 업로드**: 특허 번호가 포함된 엑셀 파일 업로드
- 🔍 **특허 검색**: SerpAPI를 통한 자동 특허 정보 수집
- 🕷️ **웹 스크래핑**: 특허 상세 설명 자동 추출
- 🗃️ **벡터 저장소**: 특허 데이터의 효율적 저장 및 검색
- 🤖 **Advanced Agentic RAG**: LangGraph 기반 지능형 분석
- 🔄 **Jina AI Reranker**: 고도화된 3단계 하이브리드 검색 (Vector + BM25 + Jina 재랭킹)
- ❓ **질문 관리**: 사용자 정의 질문 설정
- 📊 **결과 출력**: 분석 결과를 엑셀로 내보내기
- 📈 **투명성 리포트**: HTML 형태의 상세 분석 과정 제공

## 🛠️ 기술 스택

### Frontend
- **Streamlit**: 웹 인터페이스

### Backend & AI
- **Python**: 핵심 언어
- **LangChain**: AI/ML 프레임워크
- **LangGraph**: Advanced Agentic RAG 구현
- **Chroma**: 벡터 데이터베이스
- **Jina AI**: 고성능 리랭킹 시스템

### 데이터 처리
- **pandas**: 데이터 조작
- **openpyxl**: 엑셀 파일 처리
- **requests + BeautifulSoup**: 웹 스크래핑

### API 연동
- **SerpAPI**: 특허 검색
- **OpenAI**: AI 모델
- **Jina AI**: 재랭킹 API

## 🚀 설치 및 실행

### 1. 저장소 클론
```bash
git clone https://github.com/ZeronOne2/VASKUP.git
cd VASKUP
```

### 2. 가상환경 생성 및 활성화
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정
`.env` 파일을 편집하여 API 키를 설정하세요:

```env
SERPAPI_API_KEY=your_actual_serpapi_key
OPENAI_API_KEY=your_actual_openai_key
JINA_API_KEY=your_actual_jina_key
```

### 5. 애플리케이션 실행
```bash
streamlit run streamlit_app.py
```

## 📁 프로젝트 구조

```
VASKUP/
├── streamlit_app.py          # 메인 애플리케이션
├── requirements.txt          # 의존성 목록
├── .env                     # 환경 변수 (API 키)
├── .gitignore              # Git 무시 파일
├── README.md               # 프로젝트 문서
├── src/                    # 소스 코드
│   ├── hybrid_search/      # 하이브리드 검색 시스템
│   ├── langgraph_crag/     # CRAG 파이프라인
│   ├── patent_search/      # 특허 검색 모듈
│   └── vector_store/       # 벡터 저장소 관리
├── .taskmaster/           # Task Master 설정
└── test_chroma_*          # 테스트 데이터
```

## 🔑 API 키 획득 방법

### SerpAPI
1. [SerpAPI 웹사이트](https://serpapi.com/)에서 계정 생성
2. 대시보드에서 API 키 복사
3. `.env` 파일의 `SERPAPI_API_KEY`에 설정

### OpenAI
1. [OpenAI Platform](https://platform.openai.com/)에서 계정 생성
2. API 키 생성
3. `.env` 파일의 `OPENAI_API_KEY`에 설정

### Jina AI
1. [Jina AI Cloud](https://cloud.jina.ai/)에서 계정 생성
2. API 키 생성
3. `.env` 파일의 `JINA_API_KEY`에 설정

## 📖 사용 방법

1. **파일 업로드**: 특허 번호가 포함된 엑셀 파일을 업로드합니다.
2. **특허 검색**: 시스템이 자동으로 특허 정보를 수집합니다.
3. **질문 설정**: 분석하고 싶은 질문들을 설정합니다.
4. **RAG 분석**: AI가 특허를 분석하고 질문에 답변합니다.
5. **결과 확인**: 분석 결과를 확인하고 엑셀로 다운로드합니다.

## 🔍 고도화된 검색 시스템

VASKUP은 3단계 하이브리드 검색 파이프라인을 사용합니다:

1. **Vector Search**: 의미적 유사성 기반 벡터 검색
2. **BM25 Search**: 키워드 기반 통계적 검색
3. **Jina AI Reranking**: 최신 AI 모델을 통한 고성능 재랭킹

이 시스템은 검색 정확도와 관련성을 크게 향상시킵니다.

## 🤝 기여 방법

1. Fork 저장소
2. 기능 브랜치 생성 (`git checkout -b feature/AmazingFeature`)
3. 변경사항 커밋 (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치에 푸시 (`git push origin feature/AmazingFeature`)
5. Pull Request 생성

## 📄 라이센스

이 프로젝트는 MIT 라이센스하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 🐛 이슈 신고

버그나 기능 요청이 있으시면 [GitHub Issues](https://github.com/ZeronOne2/VASKUP/issues)에 신고해주세요.

## 📞 문의

프로젝트에 대한 문의사항이 있으시면 이슈를 통해 연락해주세요.

---

**개발 현황**: 🚧 Jina AI Reranker 통합 완료, 고도화된 검색 시스템 구현 완료
