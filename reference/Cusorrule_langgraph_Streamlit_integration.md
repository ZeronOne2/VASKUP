# LangGraph Streamlit Integration Guide

LangGraph 워크플로우를 Streamlit 웹 애플리케이션에 통합하는 방법을 단계별로 설명합니다.

## 아키텍처 개요

### 프로젝트 구조

```
project/
├── main.py              # Streamlit 애플리케이션
├── streamlit_wrapper.py # Streamlit 통합 래퍼
├── prompts/             # 프롬프트 템플릿 디렉토리
│   ├── code-rag-prompt.yaml
│   ├── router-prompt.yaml
│   ├── grader-prompt.yaml
│   └── agent-prompt.yaml
└── modules/             # 핵심 모듈 디렉토리
    ├── __init__.py      # 모듈 초기화
    ├── states.py        # 워크플로우 상태 정의
    ├── nodes.py         # 워크플로우 노드 구현
    ├── chains.py        # LLM 체인 컬렉션
    ├── rag.py           # RAG 체인 구성
    ├── retrievers.py    # 벡터 DB 리트리버
    ├── tools.py         # 도구 구현체
    ├── base.py          # 도구 추상 클래스
    ├── agent.py         # 에이전트 생성
    └── utils.py         # 유틸리티 함수
```

### Import 구조 변경

```python
# streamlit_wrapper.py
from modules.states import GraphState
from modules.nodes import *
from modules.chains import create_question_router_chain
from modules.rag import create_rag_chain
from modules.retrievers import init_retriever
from modules.tools import WebSearchTool
from modules.agent import create_agent_executor

# main.py
from streamlit_wrapper import create_graph, stream_graph
from modules.utils import convert_notebook_to_md
```

### 모듈 초기화 설정

```python
# modules/__init__.py
"""LangGraph 애플리케이션 핵심 모듈"""

from .states import GraphState
from .nodes import (
    BaseNode,
    RouteQuestionNode, 
    RetrieveNode,
    GeneralAnswerNode,
    RagAnswerNode,
    FilteringDocumentsNode,
    WebSearchNode,
    AnswerGroundednessCheckNode,
    AgentNode
)
from .chains import (
    create_question_router_chain,
    create_question_rewrite_chain,
    create_retrieval_grader_chain,
    create_groundedness_checker_chain,
    create_answer_grade_chain
)
from .rag import create_rag_chain
from .retrievers import init_retriever
from .tools import WebSearchTool
from .agent import create_agent_executor
from .base import BaseTool
from .prompts import prompt_manager

__all__ = [
    # States
    'GraphState',
    
    # Nodes
    'BaseNode',
    'RouteQuestionNode',
    'RetrieveNode', 
    'GeneralAnswerNode',
    'RagAnswerNode',
    'FilteringDocumentsNode',
    'WebSearchNode',
    'AnswerGroundednessCheckNode',
    'AgentNode',
    
    # Chains
    'create_question_router_chain',
    'create_question_rewrite_chain',
    'create_retrieval_grader_chain',
    'create_groundedness_checker_chain',
    'create_answer_grade_chain',
    
    # RAG & Retrievers
    'create_rag_chain',
    'init_retriever',
    
    # Tools & Agent
    'BaseTool',
    'WebSearchTool',
    'create_agent_executor',
    
    # Prompt Management
    'prompt_manager'
]
```

이렇게 구조화하면 다음과 같이 깔끔하게 임포트할 수 있습니다:

```python
# streamlit_wrapper.py
from modules import (
    GraphState,
    RouteQuestionNode,
    RetrieveNode,
    create_rag_chain,
    init_retriever,
    WebSearchTool,
    prompt_manager
)

# 또는 개별 임포트
from modules.states import GraphState
from modules.nodes import RouteQuestionNode, RetrieveNode
from modules.prompts import prompt_manager
```

## 핵심 컴포넌트 분석

### 1. 상태 관리 (states.py)

```python
from typing import List
from typing_extensions import TypedDict, Annotated

class GraphState(TypedDict):
    """워크플로우 상태를 정의하는 핵심 데이터 모델"""
    question: Annotated[str, "User question"]
    generation: Annotated[str, "LLM generated answer"]
    documents: Annotated[List[str], "List of documents"]
```

**특징:**

- TypedDict를 사용한 타입 안정성
- Annotated로 각 필드의 의미 명시
- 워크플로우 전체에서 공유되는 상태

### 2. 노드 시스템 (nodes.py)

```python
from abc import ABC, abstractmethod

class BaseNode(ABC):
    """모든 노드의 기본 클래스"""
    def __init__(self, **kwargs):
        self.name = "BaseNode"
        self.verbose = kwargs.get("verbose", False)
    
    @abstractmethod
    def execute(self, state: GraphState) -> GraphState:
        pass
    
    def __call__(self, state: GraphState):
        return self.execute(state)
```

**노드 구현 예시:**

```python
class RouteQuestionNode(BaseNode):
    """질문을 적절한 경로로 라우팅"""
    def execute(self, state: GraphState) -> str:
        question = state["question"]
        evaluation = self.router_chain.invoke({"question": question})
        
        if evaluation.binary_score == "yes":
            return "query_expansion"
        else:
            return "general_answer"

class RetrieveNode(BaseNode):
    """문서 검색 노드"""
    def execute(self, state: GraphState) -> GraphState:
        question = state["question"]
        documents = self.retriever.invoke(question)
        return GraphState(documents=documents)
```

**특징:**

- 추상 기본 클래스 패턴으로 일관성 유지
- 각 노드는 단일 책임 원칙 준수
- 로깅 기능 내장
- 함수형 호출 지원 (`__call__`)

### 3. 체인 관리 (chains.py)

```python
# 구조화된 출력을 위한 Pydantic 모델
class RouteQuery(BaseModel):
    binary_score: Literal["yes", "no"] = Field(...)

def create_question_router_chain():
    """질문 라우팅 체인 생성"""
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    structured_llm = llm.with_structured_output(RouteQuery)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    
    return prompt | structured_llm
```

**특징:**

- 팩토리 패턴으로 체인 생성
- 구조화된 출력 활용
- 프롬프트 템플릿 분리

### 4. RAG 체인 (rag.py)

```python
def create_rag_chain(prompt_name="code-rag-prompt", model_name="gpt-4o"):
    """RAG 체인 생성"""
    # YAML 파일에서 프롬프트 로드
    rag_prompt = load_prompt(f"prompts/{prompt_name}.yaml")
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    
    rag_chain = (
        {
            "question": itemgetter("question"),
            "context": itemgetter("context"),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
```

**특징:**

- YAML 파일로 프롬프트 관리
- LCEL(LangChain Expression Language) 활용
- 파라미터화된 모델 선택

### 5. 프롬프트 관리 시스템

#### YAML 프롬프트 구조

```yaml
# prompts/code-rag-prompt.yaml
_type: "prompt"
template: |
  You are an CODE Copilot Assistant. You must use the following pieces of retrieved source code or documentation to answer the question. 
  
  When answering questions, follow these guidelines:
  1. Use only the information provided in the context. 
  2. Include as many example code snippets as possible.
  3. Writing a full code snippet is highly recommended.
  4. Include sources your answer next to any relevant statements. For example, for source # 1 use [1].
  
  ### Retrieved Context
  {context}
  
  ### Question
  {question}
  
  Your answer to the question with the source:

input_variables: ["question", "context"]
```

#### 프롬프트 파일 예시

**1. Router 프롬프트 (router-prompt.yaml)**

```yaml
_type: "prompt"
template: |
  You are an expert at routing a user question. 
  The vectorstore contains documents related to {topic}.
  
  Return 'yes' if the question is related to the source code or documentation, otherwise return 'no'.
  If you don't know the answer, return 'yes'.
  
  Question: {question}

input_variables: ["topic", "question"]
```

**2. Document Grader 프롬프트 (grader-prompt.yaml)**

```yaml
_type: "prompt"
template: |
  You are a grader assessing relevance of a retrieved document to a user question.
  
  Retrieved document: 
  {document}
  
  User question: 
  {question}
  
  Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.

input_variables: ["document", "question"]
```

**3. Agent 시스템 프롬프트 (agent-prompt.yaml)**

```yaml
_type: "prompt"
template: |
  You are an helpful AI Assistant like Perplexity. Your mission is to answer the user's question.
  
  Available tools:
  {tools}
  
  Instructions:
  1. Use numbered sources in your report (e.g., [1], [2])
  2. Use markdown format
  3. Write in the same language as the user's question
  4. Include sources section:
  
  **출처**
  [1] Link or Document name
  [2] Link or Document name
  
  Question: {question}

input_variables: ["tools", "question"]
```

#### 프롬프트 로드 유틸리티

```python
# modules/prompts.py
from langchain_core.prompts import load_prompt
from pathlib import Path
from typing import Dict, Any

class PromptManager:
    """프롬프트 관리 클래스"""
    
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        self._cache: Dict[str, Any] = {}
    
    def load_prompt(self, prompt_name: str):
        """프롬프트 로드 (캐싱 적용)"""
        if prompt_name not in self._cache:
            prompt_path = self.prompts_dir / f"{prompt_name}.yaml"
            if not prompt_path.exists():
                raise FileNotFoundError(f"Prompt not found: {prompt_path}")
            
            self._cache[prompt_name] = load_prompt(str(prompt_path))
        
        return self._cache[prompt_name]
    
    def list_prompts(self):
        """사용 가능한 프롬프트 목록"""
        return [p.stem for p in self.prompts_dir.glob("*.yaml")]

# 싱글톤 인스턴스
prompt_manager = PromptManager()
```

#### 프롬프트 사용 예시

```python
# modules/chains.py
from modules.prompts import prompt_manager

def create_question_router_chain():
    """프롬프트 매니저를 사용한 체인 생성"""
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    
    # YAML에서 프롬프트 로드
    router_prompt = prompt_manager.load_prompt("router-prompt")
    
    # 동적 파라미터 설정
    router_prompt = router_prompt.partial(
        topic="RAG(Retrieval Augmented Generation) source code"
    )
    
    return router_prompt | llm.with_structured_output(RouteQuery)
```

#### 프롬프트 관리의 장점

1. **버전 관리**: Git으로 프롬프트 변경 이력 추적
2. **재사용성**: 여러 체인에서 동일 프롬프트 공유
3. **가독성**: YAML 형식으로 복잡한 프롬프트도 읽기 쉽게 관리
4. **동적 로딩**: 코드 수정 없이 프롬프트 업데이트 가능
5. **템플릿화**: 변수를 통한 유연한 프롬프트 구성

### 6. 리트리버 (retrievers.py)

```python
def init_retriever(db_index="LANGCHAIN_DB_INDEX", fetch_k=30, top_n=8):
    """벡터 DB 리트리버 초기화"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # FAISS 벡터 DB 로드
    langgraph_db = FAISS.load_local(
        db_index, embeddings, allow_dangerous_deserialization=True
    )
    
    # 기본 리트리버
    code_retriever = langgraph_db.as_retriever(search_kwargs={"k": fetch_k})
    
    # Reranker 적용
    compressor = JinaRerank(model="jina-reranker-v2-base-multilingual", top_n=top_n)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=code_retriever
    )
    return compression_retriever
```

**특징:**

- FAISS 벡터 DB 활용
- JinaRerank로 결과 재순위화
- 파라미터화된 검색 설정

## Streamlit 통합 래퍼 (streamlit_wrapper.py)

### 그래프 생성 함수

```python
def create_graph():
    """LangGraph 워크플로우 생성 및 구성"""
    retriever = init_retriever()
    rag_chain = create_rag_chain()
    
    # 그래프 초기화
    workflow = StateGraph(GraphState)
    
    # 노드 추가
    workflow.add_node("query_expand", QueryRewriteNode())
    workflow.add_node("web_search", WebSearchNode())
    workflow.add_node("retrieve", RetrieveNode(retriever))
    workflow.add_node("grade_documents", FilteringDocumentsNode())
    workflow.add_node("general_answer", GeneralAnswerNode(llm))
    workflow.add_node("rag_answer", RagAnswerNode(rag_chain))
    
    # 조건부 엣지 설정
    workflow.add_conditional_edges(
        START,
        RouteQuestionNode(),
        {
            "query_expansion": "query_expand",
            "general_answer": "general_answer"
        }
    )
    
    # 그래프 컴파일 (체크포인터 포함)
    app = workflow.compile(checkpointer=MemorySaver())
    return app
```

### 스트리밍 실행 함수

```python
def stream_graph(app, query: str, streamlit_container, thread_id: str):
    """그래프 실행 및 Streamlit UI 업데이트"""
    config = RunnableConfig(
        recursion_limit=30, 
        configurable={"thread_id": thread_id}
    )
    
    inputs = GraphState(question=query)
    
    # 각 노드별 UI 메시지
    actions = {
        "retrieve": "🔍 문서를 조회하는 중입니다.",
        "grade_documents": "👀 조회한 문서 중 중요한 내용을 추려내는 중입니다.",
        "rag_answer": "🔥 문서를 기반으로 답변을 생성하는 중입니다.",
        "web_search": "🛜 웹 검색을 진행하는 중입니다."
    }
    
    try:
        with streamlit_container.status("😊 열심히 생각중 입니다...", expanded=True) as status:
            st.write("🧑‍💻 질문의 의도를 분석하는 중입니다.")
            
            # 스트리밍 실행
            for output in app.stream(inputs, config=config):
                for key, value in output.items():
                    if key in actions:
                        st.write(actions[key])
            
            status.update(label="답변 완료", state="complete", expanded=False)
    except GraphRecursionError as e:
        print(f"Recursion limit reached: {e}")
    
    # 최종 상태 반환
    return app.get_state(config={"configurable": {"thread_id": thread_id}}).values
```

**핵심 기능:**

1. **실시간 진행 상황 표시**: status 컨테이너로 각 단계 시각화
2. **스레드 관리**: thread_id로 대화 세션 추적
3. **에러 처리**: 재귀 제한 등 예외 상황 관리
4. **상태 반환**: 최종 생성된 답변 반환

## Streamlit 애플리케이션 (main.py)

### 초기 설정

```python
import streamlit as st
from streamlit_wrapper import create_graph, stream_graph
from langchain_teddynote import logging
from langsmith import Client

# LangSmith 추적 설정
LANGSMITH_PROJECT = "Github-Code-QA-RAG"
logging.langsmith(LANGSMITH_PROJECT)

# 페이지 설정
st.set_page_config(
    page_title="LangGraph 코드 어시스턴트 💬",
    page_icon="💬",
    layout="wide"
)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = random_uuid()
if "graph" not in st.session_state:
    st.session_state["graph"] = create_graph()
```

### 메시지 관리

```python
def print_messages():
    """저장된 메시지 히스토리 출력"""
    for chat_message in st.session_state["messages"]:
        avatar = "🙎‍♂️" if chat_message.role == "user" else "😊"
        st.chat_message(chat_message.role, avatar=avatar).write(chat_message.content)

def add_message(role, message):
    """새 메시지 추가"""
    st.session_state["messages"].append(
        ChatMessage(role=role, content=message)
    )
```

### 메인 상호작용 루프

```python
# 사용자 입력 처리
user_input = st.chat_input("궁금한 내용을 물어보세요!")

if user_input:
    # 사용자 메시지 표시
    st.chat_message("user", avatar="🙎‍♂️").write(user_input)
    
    # AI 응답 생성
    with st.chat_message("assistant", avatar="😊"):
        streamlit_container = st.empty()
        
        # 그래프 실행 (스트리밍)
        response = stream_graph(
            st.session_state["graph"],
            user_input,
            streamlit_container,
            thread_id=st.session_state["thread_id"]
        )
        
        # 최종 답변 표시
        ai_answer = response["generation"]
        st.write(ai_answer)
    
    # 대화 히스토리에 추가
    add_message("user", user_input)
    add_message("assistant", ai_answer)
```

### 피드백 시스템

```python
@st.dialog("답변 평가")
def feedback():
    """사용자 피드백 수집"""
    eval1 = st.number_input("올바른 답변", min_value=1, max_value=5, value=5)
    eval2 = st.number_input("도움됨", min_value=1, max_value=5, value=5)
    eval3 = st.number_input("구체성", min_value=1, max_value=5, value=5)
    comment = st.text_area("의견(선택)")
    
    if st.button("제출", type="primary"):
        # LangSmith에 피드백 전송
        submit_feedback()
        st.rerun()
```

## 에이전트 시스템 구현

### 1. 도구 추상화 패턴 (base.py)

```python
from typing import Any, TypeVar, Generic
from abc import ABC, abstractmethod

T = TypeVar("T")  # 제네릭 타입 변수

class BaseTool(ABC, Generic[T]):
    """모든 도구의 기본 추상 클래스"""
    
    @abstractmethod
    def _create_tool(self) -> T:
        """실제 도구 객체를 생성"""
        pass
    
    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> T:
        """팩토리 메서드 패턴"""
        instance = cls(*args, **kwargs)
        return instance._create_tool()
```

**특징:**

- 제네릭을 활용한 타입 안정성
- 팩토리 메서드 패턴으로 일관된 도구 생성
- 추상 클래스로 인터페이스 통일

### 2. 도구 구현 (tools.py)

```python
class WebSearchTool(BaseTool[TavilySearch]):
    """웹 검색 도구 구현"""
    
    def __init__(
        self,
        topic: str = "general",
        max_results: int = 3,
        include_answer: bool = False,
        include_raw_content: bool = False,
        include_images: bool = False,
        format_output: bool = False,
        include_domains: List[str] = [],
        exclude_domains: List[str] = [],
    ):
        self.topic = topic
        self.max_results = max_results
        # ... 기타 설정
    
    def _create_tool(self) -> TavilySearch:
        """TavilySearch 인스턴스 생성 및 구성"""
        search = TavilySearch(
            topic=self.topic,
            max_results=self.max_results,
            # ... 기타 파라미터
        )
        search.name = "web_search"
        search.description = "Use this tool to search on the web"
        return search
```

**사용 예시:**

```python
# 팩토리 메서드로 도구 생성
search_tool = WebSearchTool.create(
    max_results=5,
    include_answer=True,
    format_output=True
)

# 특정 도메인만 검색
news_tool = WebSearchTool.create(
    topic="news",
    include_domains=["reuters.com", "bbc.com"],
    max_results=10
)
```

### 3. ReAct 에이전트 생성 (agent.py)

```python
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

def create_agent_executor(model_name="gpt-4o", tools=[]):
    """ReAct 패턴 에이전트 생성"""
    
    # 메모리 설정 (대화 히스토리 유지)
    memory = MemorySaver()
    
    # LLM 모델 설정
    model = ChatOpenAI(model_name=model_name)
    
    # 시스템 프롬프트 (Perplexity 스타일)
    system_prompt = """You are an helpful AI Assistant like Perplexity. 
    Your mission is to answer the user's question.

    Here are the tools you can use:
    {tools}

    Please follow these instructions:
    1. Use numbered sources in your report (e.g., [1], [2])
    2. Use markdown format
    3. Write in the same language as the user's question
    4. Include sources section at the end:
    
    **출처**
    [1] Link or Document name
    [2] Link or Document name
    """
    
    # ReAct 에이전트 생성
    agent_executor = create_react_agent(
        model, 
        tools=tools, 
        checkpointer=memory,
        state_modifier=system_prompt
    )
    
    return agent_executor
```

**핵심 기능:**

- **메모리 관리**: MemorySaver로 대화 컨텍스트 유지
- **시스템 프롬프트**: 구조화된 답변 형식 강제
- **출처 표시**: 신뢰성 있는 답변 제공
- **다국어 지원**: 사용자 언어로 응답

### 4. 에이전트 통합 워크플로우

#### 노드에서 에이전트 활용

```python
class AgentNode(BaseNode):
    """에이전트를 활용하는 노드"""
    
    def __init__(self, tools=[], **kwargs):
        super().__init__(**kwargs)
        self.agent = create_agent_executor(tools=tools)
    
    def execute(self, state: GraphState) -> GraphState:
        question = state["question"]
        
        # 에이전트 실행
        response = self.agent.invoke({
            "messages": [HumanMessage(content=question)]
        })
        
        # 응답에서 답변 추출
        answer = response["messages"][-1].content
        
        return GraphState(generation=answer)
```

#### 도구를 포함한 워크플로우 구성

```python
def create_agent_workflow():
    """에이전트 기반 워크플로우 생성"""
    
    # 도구 준비
    tools = [
        WebSearchTool.create(max_results=5),
        # 추가 도구들...
    ]
    
    # 그래프 초기화
    workflow = StateGraph(GraphState)
    
    # 에이전트 노드 추가
    workflow.add_node("agent", AgentNode(tools=tools))
    
    # 간단한 플로우
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)
    
    return workflow.compile()
```

### 5. Streamlit에서 에이전트 활용

#### 에이전트 래퍼 확장

```python
def create_agent_graph(tools=[]):
    """에이전트 기반 그래프 생성"""
    
    # 기본 도구 설정
    if not tools:
        tools = [
            WebSearchTool.create(
                max_results=5,
                format_output=True
            )
        ]
    
    # 에이전트 생성
    agent = create_agent_executor(tools=tools)
    
    # 그래프 구성
    workflow = StateGraph(GraphState)
    
    # 에이전트 노드 정의
    def agent_node(state: GraphState):
        response = agent.invoke({
            "messages": [HumanMessage(content=state["question"])]
        })
        return {"generation": response["messages"][-1].content}
    
    workflow.add_node("agent", agent_node)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)
    
    return workflow.compile(checkpointer=MemorySaver())
```

#### Streamlit UI 통합

```python
# main.py에 추가
def initialize_agent_mode():
    """에이전트 모드 초기화"""
    
    # 도구 선택 UI
    st.sidebar.subheader("🛠️ 도구 설정")
    
    use_web_search = st.sidebar.checkbox("웹 검색 활성화", value=True)
    
    tools = []
    if use_web_search:
        max_results = st.sidebar.slider("검색 결과 수", 1, 10, 5)
        tools.append(
            WebSearchTool.create(
                max_results=max_results,
                format_output=True
            )
        )
    
    # 에이전트 그래프 생성
    if "agent_graph" not in st.session_state:
        st.session_state["agent_graph"] = create_agent_graph(tools)
    
    return st.session_state["agent_graph"]

# 메인 앱에서 사용
if st.sidebar.toggle("에이전트 모드", value=False):
    graph = initialize_agent_mode()
else:
    graph = st.session_state["graph"]  # 기본 워크플로우
```

### 6. 고급 에이전트 패턴

#### 멀티 에이전트 시스템

```python
def create_multi_agent_system():
    """전문화된 에이전트들의 협업 시스템"""
    
    # 연구 에이전트
    research_agent = create_agent_executor(
        tools=[WebSearchTool.create(topic="research", max_results=10)]
    )
    
    # 분석 에이전트
    analysis_agent = create_agent_executor(
        tools=[DataAnalysisTool.create()]
    )
    
    # 작성 에이전트
    writing_agent = create_agent_executor(
        tools=[FormattingTool.create()]
    )
    
    # 에이전트 오케스트레이션
    class MultiAgentNode(BaseNode):
        def execute(self, state: GraphState) -> GraphState:
            question = state["question"]
            
            # 1. 연구 단계
            research = research_agent.invoke({
                "messages": [HumanMessage(content=f"Research: {question}")]
            })
            
            # 2. 분석 단계
            analysis = analysis_agent.invoke({
                "messages": [
                    HumanMessage(content=f"Analyze: {research['messages'][-1].content}")
                ]
            })
            
            # 3. 최종 작성
            final = writing_agent.invoke({
                "messages": [
                    HumanMessage(content=f"Write report: {analysis['messages'][-1].content}")
                ]
            })
            
            return GraphState(generation=final["messages"][-1].content)
```

#### 도구 동적 선택

```python
class AdaptiveAgentNode(BaseNode):
    """질문에 따라 도구를 동적으로 선택하는 에이전트"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tool_selector = create_tool_selector_chain()
        
    def execute(self, state: GraphState) -> GraphState:
        question = state["question"]
        
        # 필요한 도구 결정
        required_tools = self.tool_selector.invoke({"question": question})
        
        # 도구 인스턴스 생성
        tools = []
        for tool_name in required_tools:
            if tool_name == "web_search":
                tools.append(WebSearchTool.create())
            elif tool_name == "calculator":
                tools.append(CalculatorTool.create())
            # ... 추가 도구들
        
        # 선택된 도구로 에이전트 생성
        agent = create_agent_executor(tools=tools)
        
        # 실행
        response = agent.invoke({
            "messages": [HumanMessage(content=question)]
        })
        
        return GraphState(generation=response["messages"][-1].content)
```

## 통합 구현 가이드

### 1. 기본 설정

```python
# 필수 환경 변수 (.env)
OPENAI_API_KEY=your_api_key
LANGCHAIN_API_KEY=your_langsmith_key
TAVILY_API_KEY=your_tavily_key
JINA_API_KEY=your_jina_key
```

### 2. 그래프 생성 및 초기화

```python
# 세션 시작 시 한 번만 생성
if "graph" not in st.session_state:
    st.session_state["graph"] = create_graph()
```

### 3. 사용자 입력 처리

```python
# chat_input으로 사용자 입력 받기
user_input = st.chat_input("질문을 입력하세요")

if user_input:
    # 빈 컨테이너 생성 (실시간 업데이트용)
    container = st.empty()
    
    # 그래프 실행
    response = stream_graph(
        graph=st.session_state["graph"],
        query=user_input,
        streamlit_container=container,
        thread_id=st.session_state["thread_id"]
    )
```

### 4. 실시간 진행 상황 표시

```python
# streamlit_wrapper.py의 stream_graph 함수가 자동으로 처리
# status 컨테이너를 통해 각 단계별 진행 상황 표시
with container.status("😊생각중 입니다...", expanded=True) as status:
    st.write("🧑‍💻 질문의 의도를 분석하는 중입니다.")
    for output in app.stream(inputs, config=config):
		# 출력된 결과에서 키와 값을 순회합니다.
		for key, value in output.items():
			# 노드의 이름과 해당 노드에서 나온 출력을 출력합니다.
			if key in actions:
				st.write(actions[key])
		# 출력 값을 예쁘게 출력합니다.
	status.update(label="답변 완료", state="complete", expanded=False)
```

### 5. 세션 관리

```python
# 새 대화 시작
if st.button("새로운 주제로 질문"):
    st.session_state["messages"] = []
    st.session_state["thread_id"] = random_uuid()
    st.rerun()
```

## 고급 기능

### 1. 커스텀 노드 추가

```python
class CustomAnalysisNode(BaseNode):
    def __init__(self, analyzer, **kwargs):
        super().__init__(**kwargs)
        self.analyzer = analyzer
    
    def execute(self, state: GraphState) -> GraphState:
        # 커스텀 분석 로직
        analysis = self.analyzer.analyze(state["documents"])
        return GraphState(analysis_result=analysis)

# 워크플로우에 추가
workflow.add_node("custom_analysis", CustomAnalysisNode(analyzer))
```

### 2. 동적 UI 업데이트

```python
def stream_graph_with_metrics(app, query, container):
    """메트릭과 함께 스트리밍"""
    col1, col2, col3 = container.columns(3)
    
    with col1:
        doc_count = st.metric("검색된 문서", 0)
    with col2:
        relevance = st.metric("관련성 점수", 0)
    with col3:
        time_elapsed = st.metric("처리 시간", "0s")
    
    # 스트리밍 중 메트릭 업데이트
    for output in app.stream(inputs, config):
        if "documents" in output:
            doc_count.metric("검색된 문서", len(output["documents"]))
```

### 3. 에러 처리 및 복구

```python
try:
    response = stream_graph(graph, query, container, thread_id)
except GraphRecursionError:
    st.error("처리 중 문제가 발생했습니다. 질문을 다시 구성해주세요.")
    # 폴백 처리
    response = {"generation": "죄송합니다. 답변 생성에 실패했습니다."}
except Exception as e:
    st.error(f"예상치 못한 오류: {str(e)}")
    # 상태 초기화
    st.session_state["thread_id"] = random_uuid()
```

## 배포 고려사항

### 1. 성능 최적화

```python
# 그래프 캐싱
@st.cache_resource
def get_cached_graph():
    return create_graph()

# 리트리버 캐싱
@st.cache_resource
def get_cached_retriever():
    return init_retriever()
```

### 2. 동시성 처리

```python
# 세션별 독립적인 thread_id 관리
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = f"user_{st.session_state.user_id}_{random_uuid()}"
```

### 3. 리소스 관리

```python
# 메모리 사용량 모니터링
if len(st.session_state["messages"]) > 100:
    # 오래된 메시지 정리
    st.session_state["messages"] = st.session_state["messages"][-50:]
```

## 요약

이 통합 가이드는 LangGraph 워크플로우를 Streamlit 애플리케이션에 효과적으로 통합하는 방법을 제시합니다. 핵심은:

1. **모듈화**: 각 컴포넌트를 독립적으로 관리
2. **래퍼 패턴**: streamlit_wrapper로 복잡성 캡슐화
3. **실시간 피드백**: 스트리밍 API로 사용자 경험 향상
4. **세션 관리**: thread_id로 대화 컨텍스트 유지
5. **에러 처리**: 안정적인 사용자 경험 제공
6. **에이전트 통합**: ReAct 패턴과 도구 추상화로 확장성 확보

### 아키텍처 요약

```
📁 project/
├── 📄 main.py              → Streamlit 앱
├── 📄 streamlit_wrapper.py → 통합 래퍼
├── 📁 prompts/             → 프롬프트 템플릿
│   ├── 📄 code-rag-prompt.yaml
│   ├── 📄 router-prompt.yaml
│   ├── 📄 grader-prompt.yaml
│   └── 📄 agent-prompt.yaml
└── 📁 modules/             → 핵심 모듈
    ├── 📄 __init__.py      → 모듈 초기화
    ├── 📄 states.py        → 워크플로우 상태 정의
    ├── 📄 nodes.py         → 추상 노드 클래스와 구현체
    ├── 📄 chains.py        → LLM 체인 팩토리
    ├── 📄 rag.py           → RAG 체인 구성
    ├── 📄 retrievers.py    → 벡터 DB 관리
    ├── 📄 base.py          → 도구 추상 클래스
    ├── 📄 tools.py         → 도구 구현체
    ├── 📄 agent.py         → ReAct 에이전트 생성
    ├── 📄 prompts.py       → 프롬프트 관리자
    └── 📄 utils.py         → 유틸리티 함수
```

이 패턴을 활용하면 복잡한 LangGraph 워크플로우도 직관적인 웹 인터페이스로 제공할 수 있으며, 에이전트와 도구를 통해 더욱 강력한 기능을 구현할 수 있습니다.