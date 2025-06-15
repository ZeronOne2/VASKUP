# LangGraph Components Library

LangGraph 워크플로우에서 자주 사용되는 재사용 가능한 컴포넌트들을 정리한 라이브러리입니다.

## LLM Chains

### 1. 기본 LLM Chain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 기본 체인
llm = ChatOpenAI(model="gpt-4o", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])
chain = prompt | llm | StrOutputParser()
```

### 2. Structured Output Chain

```python
from pydantic import BaseModel, Field
from typing import List

class Plan(BaseModel):
    """작업 계획"""
    steps: List[str] = Field(description="실행할 단계들")
    
class RouteQuery(BaseModel):
    """쿼리 라우팅"""
    datasource: Literal["vectorstore", "web_search"]

# 구조화된 출력
structured_llm = llm.with_structured_output(Plan)
router = llm.with_structured_output(RouteQuery)

# 프롬프트와 결합
planner = prompt | structured_llm
question_router = route_prompt | router
```

### 3. Function Calling Chain

```python
from langchain_core.tools import tool

@tool
def calculate(expression: str) -> str:
    """수학 표현식 계산"""
    return str(eval(expression))

# Function calling 설정
llm_with_tools = llm.bind_tools([calculate])
chain_with_tools = prompt | llm_with_tools
```

### 4. Streaming Chain

```python
# 스트리밍 지원
async def stream_chain(question: str):
    async for chunk in chain.astream({"input": question}):
        yield chunk
```

## Retrievers

### 1. Vector Store Retriever

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 벡터스토어 생성
embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# 문서 분할 및 저장
splits = text_splitter.split_documents(documents)
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings
)

# Retriever 생성
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
```

### 2. Multi-Query Retriever

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

# 쿼리 생성 프롬프트
query_prompt = ChatPromptTemplate.from_messages([
    ("system", "Generate 3 different versions of the given question to retrieve relevant documents."),
    ("human", "{question}")
])

# Multi-query retriever
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm,
    prompt=query_prompt
)
```

### 3. Contextual Compression Retriever

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# 압축기 생성
compressor = LLMChainExtractor.from_llm(llm)

# 압축 retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

### 4. Ensemble Retriever

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# BM25 retriever
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 2

# Ensemble retriever (BM25 + Vector)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, retriever],
    weights=[0.5, 0.5]
)
```

### 5. Parent Document Retriever

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

# 부모 문서 저장소
parent_store = InMemoryStore()

# Parent document retriever
parent_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=parent_store,
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=400),
    parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000)
)
```

## Rerankers

### 1. Cross-Encoder Reranker

```python
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 5):
        # 점수 계산
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)
        
        # 정렬
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in doc_scores[:top_k]]
```

### 2. Cohere Reranker

```python
import cohere

class CohereReranker:
    def __init__(self, api_key: str):
        self.co = cohere.Client(api_key)
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 5):
        docs_text = [doc.page_content for doc in documents]
        
        results = self.co.rerank(
            model="rerank-english-v2.0",
            query=query,
            documents=docs_text,
            top_n=top_k
        )
        
        return [documents[r.index] for r in results]
```

### 3. BGE Reranker

```python
from FlagEmbedding import FlagReranker

class BGEReranker:
    def __init__(self, model_name="BAAI/bge-reranker-large"):
        self.reranker = FlagReranker(model_name)
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 5):
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.reranker.compute_score(pairs)
        
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in doc_scores[:top_k]]
```

### 4. LLM-based Reranker

```python
class LLMReranker:
    def __init__(self, llm):
        self.llm = llm
        
    def rerank(self, query: str, documents: List[Document], top_k: int = 5):
        # 관련성 평가 프롬프트
        eval_prompt = ChatPromptTemplate.from_template(
            """Rate the relevance of the document to the query on a scale of 1-10.
            Query: {query}
            Document: {document}
            
            Output only the numeric score."""
        )
        
        chain = eval_prompt | self.llm | StrOutputParser()
        
        # 각 문서 평가
        doc_scores = []
        for doc in documents:
            score = float(chain.invoke({
                "query": query,
                "document": doc.page_content
            }))
            doc_scores.append((doc, score))
        
        # 정렬 및 반환
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in doc_scores[:top_k]]
```

### 5. Custom Scoring Reranker

```python
class CustomScoringReranker:
    def __init__(self, scoring_functions: List[callable]):
        self.scoring_functions = scoring_functions
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 5):
        doc_scores = []
        
        for doc in documents:
            # 여러 점수 함수의 가중 평균
            scores = [fn(query, doc) for fn in self.scoring_functions]
            final_score = sum(scores) / len(scores)
            doc_scores.append((doc, final_score))
        
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in doc_scores[:top_k]]

# 사용 예시
def keyword_score(query, doc):
    keywords = query.lower().split()
    content = doc.page_content.lower()
    return sum(1 for k in keywords if k in content)

def length_score(query, doc):
    return min(len(doc.page_content) / 1000, 1.0)

reranker = CustomScoringReranker([keyword_score, length_score])
```

## Tools

### 1. Search Tools

```python
from langchain_teddynote.tools.tavily import TavilySearch
from langchain_community.tools import DuckDuckGoSearchRun

# Tavily 검색
tavily_tool = TavilySearch(max_results=5)

# DuckDuckGo 검색
ddg_search = DuckDuckGoSearchRun()

# 커스텀 검색 도구
@tool
def search_arxiv(query: str) -> str:
    """ArXiv 논문 검색"""
    # ArXiv API 호출 로직
    return results
```

### 2. Code Execution Tools

```python
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.utilities import PythonREPL

# Python REPL
python_tool = PythonREPLTool()

# 커스텀 코드 실행 도구
@tool
def execute_code(code: str, language: str = "python") -> str:
    """코드 실행 도구"""
    if language == "python":
        repl = PythonREPL()
        try:
            result = repl.run(code)
            return f"실행 성공:\n{result}"
        except Exception as e:
            return f"실행 실패: {str(e)}"
```

### 3. File Operations Tools

```python
from pathlib import Path

@tool
def read_file(file_path: str) -> str:
    """파일 읽기"""
    return Path(file_path).read_text()

@tool
def write_file(file_path: str, content: str) -> str:
    """파일 쓰기"""
    Path(file_path).write_text(content)
    return f"파일 저장 완료: {file_path}"

@tool
def list_files(directory: str = ".") -> List[str]:
    """디렉토리 파일 목록"""
    return [f.name for f in Path(directory).iterdir()]
```

### 4. API Integration Tools

```python
import requests

@tool
def call_api(
    url: str,
    method: str = "GET",
    headers: dict = None,
    data: dict = None
) -> str:
    """API 호출 도구"""
    response = requests.request(
        method=method,
        url=url,
        headers=headers,
        json=data
    )
    return response.json()

# 특화된 API 도구
@tool
def github_search(query: str, repo: str = None) -> str:
    """GitHub 검색"""
    # GitHub API 로직
    pass
```

## Prompts & Templates

### 1. RAG Prompts

```python
# 기본 RAG 프롬프트
rag_prompt = ChatPromptTemplate.from_template("""
다음 문맥을 사용하여 질문에 답하세요.

문맥:
{context}

질문: {question}

답변:
""")

# 출처 포함 RAG 프롬프트
rag_with_sources = ChatPromptTemplate.from_template("""
다음 문서들을 참고하여 질문에 답하세요.

문서:
{documents}

질문: {question}

답변 시 출처를 반드시 명시하세요.
""")
```

### 2. Agent System Prompts

```python
# Research Agent 프롬프트
research_prompt = """
당신은 전문 리서치 에이전트입니다.
주어진 주제에 대해 깊이 있는 조사를 수행하고,
신뢰할 수 있는 출처에서 정보를 수집합니다.
"""

# Planning Agent 프롬프트
planning_prompt = """
당신은 작업 계획을 수립하는 전문가입니다.
복잡한 작업을 단계별로 분해하고,
각 단계의 실행 순서를 정합니다.
"""
```

### 3. Router Prompts

```python
# 쿼리 라우터 프롬프트
router_prompt = ChatPromptTemplate.from_messages([
    ("system", """질문을 분석하여 적절한 데이터 소스로 라우팅합니다.
    - vectorstore: 내부 문서 관련 질문
    - web_search: 최신 정보나 외부 정보가 필요한 질문"""),
    ("human", "{question}")
])

# 에이전트 라우터 프롬프트
agent_router_prompt = """
작업을 분석하여 적절한 에이전트를 선택하세요:
- Researcher: 정보 수집이 필요한 경우
- Writer: 문서 작성이 필요한 경우
- Coder: 코드 작성이 필요한 경우
"""
```

### 4. Evaluation Prompts

```python
# 문서 관련성 평가
relevance_prompt = ChatPromptTemplate.from_template("""
문서가 질문과 관련이 있는지 평가하세요.

질문: {question}
문서: {document}

관련성 (yes/no):
""")

# 답변 품질 평가
quality_prompt = ChatPromptTemplate.from_template("""
생성된 답변의 품질을 평가하세요.

질문: {question}
답변: {answer}

평가 기준:
1. 정확성
2. 완전성
3. 명확성

점수 (1-10):
""")
```

## Utilities

### 1. Document Loaders

```python
from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    TextLoader,
    CSVLoader,
    JSONLoader
)

# PDF 로더
pdf_loader = PyPDFLoader("document.pdf")

# 웹 로더
web_loader = WebBaseLoader(
    web_paths=["https://example.com"],
    header_template={
        "User-Agent": "Mozilla/5.0"
    }
)

# JSON 로더
json_loader = JSONLoader(
    file_path="data.json",
    jq_schema=".messages[]",
    text_content=False
)
```

### 2. Text Splitters

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownTextSplitter
)

# 재귀적 문자 분할
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# 토큰 기반 분할
token_splitter = TokenTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Markdown 분할
md_splitter = MarkdownTextSplitter(
    chunk_size=1000,
    chunk_overlap=0
)
```

### 3. Embeddings

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# OpenAI 임베딩
openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# HuggingFace 임베딩
hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# 커스텀 임베딩
class CustomEmbeddings:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 커스텀 임베딩 로직
        pass
    
    def embed_query(self, text: str) -> List[float]:
        # 쿼리 임베딩 로직
        pass
```

### 4. Vector Stores

```python
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS, Pinecone
from langchain_postgres.vectorstores import PGVector

# Chroma
chroma_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# FAISS
faiss_store = FAISS.from_documents(
    documents=documents,
    embedding=embeddings
)

# PostgreSQL + pgvector
pg_store = PGVector.from_documents(
    documents=documents,
    embedding=embeddings,
    connection_string="postgresql://user:pass@localhost/db"
)
```

### 5. Document Graders

```python
from pydantic import BaseModel, Field

class GradeDocuments(BaseModel):
    """문서 관련성 평가"""
    binary_score: str = Field(description="관련성 여부 'yes' 또는 'no'")

class GradeHallucination(BaseModel):
    """환각 평가"""
    binary_score: str = Field(description="사실 기반 여부 'yes' 또는 'no'")

class GradeAnswer(BaseModel):
    """답변 품질 평가"""
    binary_score: str = Field(description="답변 적절성 'yes' 또는 'no'")

# 평가기 생성
doc_grader = llm.with_structured_output(GradeDocuments)
hallucination_grader = llm.with_structured_output(GradeHallucination)
answer_grader = llm.with_structured_output(GradeAnswer)
```

## 사용 예시

### RAG 파이프라인 구성

```python
# 1. 문서 로드 및 분할
loader = PyPDFLoader("document.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
splits = text_splitter.split_documents(documents)

# 2. 벡터스토어 생성
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever(k=10)

# 3. Reranker 적용
reranker = CrossEncoderReranker()
def retrieve_and_rerank(query: str):
    docs = retriever.invoke(query)
    return reranker.rerank(query, docs, top_k=3)

# 4. RAG 체인 구성
rag_chain = (
    {"context": retrieve_and_rerank, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)
```

이 라이브러리는 LangGraph 워크플로우에서 바로 사용할 수 있는 컴포넌트들을 제공합니다. 각 컴포넌트는 독립적으로 사용하거나 조합하여 복잡한 워크플로우를 구성할 수 있습니다.