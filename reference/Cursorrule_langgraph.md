# LangGraph AI Workflow 개발을 위한 Cursor Rules

## 프로젝트 개요

이 프로젝트는 LangGraph를 사용하여 고도화된 AI workflow를 구축합니다. 복잡한 상태 관리, 다중 에이전트 협업, 동적 라우팅, 인간 검토 프로세스를 포함한 엔터프라이즈급 AI 애플리케이션을 개발합니다.

## 핵심 아키텍처 원칙

### 1. State-First Design

- 모든 workflow는 명확하게 정의된 State로 시작합니다
- TypedDict를 사용하여 타입 안정성을 보장합니다
- Annotated를 활용하여 상태 업데이트 전략을 명시합니다
- MessagesState를 기반으로 메시지 중심 워크플로우를 구축합니다

### 2. Node Composition Pattern

- 각 노드는 단일 책임 원칙을 따릅니다
- 노드는 순수 함수로 작성하여 테스트 가능성을 높입니다
- 상태 변경은 명시적으로 반환값을 통해서만 수행합니다
- 복잡한 로직은 여러 노드로 분해합니다

### 3. Edge Management Strategy

- 명시적 edge는 결정적 흐름에 사용합니다
- conditional_edges는 동적 라우팅에 활용합니다
- 라우팅 함수는 상태 기반으로 명확한 결정을 내립니다
- 순환 참조를 방지하고 명확한 종료 조건을 설정합니다

### 4. Error Handling & Resilience

- 각 노드에서 예외 처리를 구현합니다
- 재시도 로직과 폴백 메커니즘을 포함합니다
- 상태 검증을 통해 데이터 무결성을 보장합니다
- 체크포인팅으로 장애 복구를 지원합니다

## 주요 워크플로우 패턴

### 1. Adaptive RAG Pattern

쿼리 유형에 따라 웹 검색과 벡터스토어를 동적으로 선택하는 패턴입니다.

```python
# 상태 정의
class GraphState(TypedDict):
    question: Annotated[str, "User question"]
    generation: Annotated[str, "LLM generated answer"]
    documents: Annotated[List[str], "List of documents"]

# 라우팅 로직
def route_question(state: GraphState):
    """질문을 웹 검색 또는 벡터스토어로 라우팅"""
    question = state["question"]
    source = question_router.invoke({"question": question})
    
    if source.datasource == "web_search":
        return "web_search"
    elif source.datasource == "vectorstore":
        return "vectorstore"

# 그래프 구성
workflow = StateGraph(GraphState)
workflow.add_node("web_search", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)

# 조건부 라우팅
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
    }
)
```

### 2. Plan-and-Execute Pattern

복잡한 작업을 계획하고 단계별로 실행하는 패턴입니다.

```python
# 상태 정의
class PlanExecute(TypedDict):
    input: Annotated[str, "User's input"]
    plan: Annotated[List[str], "Current plan"]
    past_steps: Annotated[List[Tuple], operator.add]
    response: Annotated[str, "Final response"]

# 계획 수립 노드
def plan_step(state: PlanExecute):
    plan = planner.invoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}

# 실행 노드
def execute_step(state: PlanExecute):
    task = state["plan"][0]
    agent_response = agent_executor.invoke({"messages": [("user", task)]})
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }

# 재계획 노드
def replan_step(state: PlanExecute):
    output = replanner.invoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}
```

### 3. Multi-Agent Collaboration Pattern

여러 전문 에이전트가 협업하여 작업을 수행하는 패턴입니다.

```python
# 에이전트 노드 팩토리
class AgentFactory:
    def create_agent_node(self, agent, name: str):
        def agent_node(state):
            result = agent.invoke(state)
            return {
                "messages": [
                    HumanMessage(content=result["messages"][-1].content, name=name)
                ]
            }
        return agent_node

# Research Agent 노드
research_agent = create_react_agent(llm, tools=[tavily_tool])
research_node = agent_factory.create_agent_node(research_agent, "researcher")

# Chart Generator 노드
chart_agent = create_react_agent(llm, [python_repl_tool])
chart_node = agent_factory.create_agent_node(chart_agent, "chart_generator")
```

### 4. Hierarchical Team Pattern

Supervisor가 여러 팀을 관리하는 계층적 구조 패턴입니다.

```python
# 팀 Supervisor 생성
def create_team_supervisor(model_name, system_prompt, members):
    options_for_next = ["FINISH"] + members
    
    class RouteResponse(BaseModel):
        next: Literal[*options_for_next]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Given the conversation above, who should act next?")
    ])
    
    return prompt | llm.with_structured_output(RouteResponse)

# 계층적 그래프 구성
super_graph = StateGraph(State)
super_graph.add_node("ResearchTeam", research_team_graph)
super_graph.add_node("WritingTeam", writing_team_graph)
super_graph.add_node("Supervisor", supervisor_node)
```

## 구현 가이드라인

### 1. State 설계 원칙

```python
# 기본 State
class WorkflowState(TypedDict):
    # 필수 필드
    messages: Annotated[List[BaseMessage], add_messages]
    current_step: str
    
    # 집계 필드 (operator.add 사용)
    results: Annotated[List[dict], operator.add]
    errors: Annotated[List[str], operator.add]
    
    # 선택적 필드
    metadata: Optional[dict]
    context: Optional[dict]

# 특화된 State (RAG)
class RAGState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    relevance_scores: List[float]

# 멀티 에이전트 State
class MultiAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    sender: str  # 마지막 메시지 발신자
    team_members: List[str]
    next: str  # 다음 실행할 에이전트
```

### 2. Node 구현 베스트 프랙티스

```python
# 기본 노드 패턴
def process_node(state: WorkflowState) -> dict:
    """상태를 받아 업데이트할 필드만 반환"""
    try:
        # 1. 상태 검증
        validate_state(state)
        
        # 2. 비즈니스 로직
        result = perform_task(state)
        
        # 3. 부분 상태 업데이트 반환
        return {
            "results": [result],
            "current_step": "next_step"
        }
    except Exception as e:
        return {
            "errors": [str(e)],
            "current_step": "error_handler"
        }

# RAG 노드 패턴
def retrieve(state: RAGState):
    """문서 검색 수행"""
    print("==== [RETRIEVE] ====")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents}

def grade_documents(state: RAGState):
    """문서 관련성 평가"""
    print("==== [CHECK DOCUMENT RELEVANCE] ====")
    filtered_docs = []
    for doc in state["documents"]:
        score = retrieval_grader.invoke({
            "question": state["question"],
            "document": doc.page_content
        })
        if score.binary_score == "yes":
            filtered_docs.append(doc)
    return {"documents": filtered_docs}
```

### 3. Conditional Edge 패턴

```python
# 단순 라우팅
def route_by_score(state):
    if state["score"] > 0.8:
        return "high_quality"
    elif state["score"] > 0.5:
        return "medium_quality"
    else:
        return "low_quality"

# 복잡한 라우팅 (Adaptive RAG)
def decide_to_generate(state):
    """문서 평가 후 생성 여부 결정"""
    if not state["documents"]:
        print("==== [DECISION: TRANSFORM QUERY] ====")
        return "transform_query"
    else:
        print("==== [DECISION: GENERATE] ====")
        return "generate"

# Supervisor 라우팅
def supervisor_route(state):
    """Supervisor의 결정에 따른 라우팅"""
    decision = state["next"]
    if decision == "FINISH":
        return END
    return decision
```

### 4. Tool Integration

```python
# Tavily 검색 도구
from langchain_teddynote.tools.tavily import TavilySearch
tavily_tool = TavilySearch(max_results=5)

# Python REPL 도구
@tool
def python_repl_tool(code: Annotated[str, "Python code to execute"]):
    """Python 코드 실행 도구"""
    try:
        result = python_repl.run(code)
        return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    except Exception as e:
        return f"Failed to execute. Error: {repr(e)}"

# 웹 스크래핑 도구
@tool
def scrape_webpages(urls: List[str]) -> str:
    """웹 페이지 스크래핑"""
    loader = WebBaseLoader(web_path=urls)
    docs = loader.load()
    return "\n\n".join([
        f'<Document>{doc.page_content}</Document>'
        for doc in docs
    ])
```

### 5. Graph 구성 패턴

```python
# 기본 그래프 구성
def build_basic_graph():
    workflow = StateGraph(State)
    
    # 노드 추가
    workflow.add_node("start", start_node)
    workflow.add_node("process", process_node)
    workflow.add_node("end", end_node)
    
    # 엣지 추가
    workflow.add_edge(START, "start")
    workflow.add_edge("start", "process")
    workflow.add_edge("process", "end")
    workflow.add_edge("end", END)
    
    # 체크포인터 추가
    return workflow.compile(checkpointer=MemorySaver())

# 조건부 그래프 구성
def build_conditional_graph():
    workflow = StateGraph(State)
    
    # 노드 추가
    workflow.add_node("router", router_node)
    workflow.add_node("path_a", path_a_node)
    workflow.add_node("path_b", path_b_node)
    
    # 조건부 엣지
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "path_a": "path_a",
            "path_b": "path_b",
            END: END
        }
    )
    
    return workflow.compile()

# 계층적 그래프 구성
def build_hierarchical_graph():
    # 하위 그래프 정의
    team_a = build_team_graph("TeamA", ["worker1", "worker2"])
    team_b = build_team_graph("TeamB", ["worker3", "worker4"])
    
    # 상위 그래프
    super_graph = StateGraph(State)
    super_graph.add_node("TeamA", team_a)
    super_graph.add_node("TeamB", team_b)
    super_graph.add_node("Supervisor", supervisor)
    
    # 계층 연결
    super_graph.add_conditional_edges(
        "Supervisor",
        get_next_team,
        {
            "TeamA": "TeamA",
            "TeamB": "TeamB",
            "FINISH": END
        }
    )
    
    return super_graph.compile()
```

## 실행 및 모니터링

### 1. 그래프 실행

```python
from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import stream_graph, invoke_graph

# Config 설정
config = RunnableConfig(
    recursion_limit=50,  # 재귀 제한
    configurable={"thread_id": str(uuid.uuid4())}
)

# 동기 실행
result = graph.invoke({"messages": [HumanMessage(content="질문")]}, config)

# 스트리밍 실행
stream_graph(app, inputs, config, ["agent", "generate"])

# 상태 확인
state = app.get_state(config).values
```

### 2. 로깅 및 디버깅

```python
# LangSmith 설정
from langchain_teddynote import logging
logging.langsmith("project-name")

# 노드별 로깅
def logged_node(state):
    print(f"==== [NODE_NAME] ====")
    print(f"Input state: {state}")
    
    result = process(state)
    
    print(f"Output: {result}")
    return result

# 시각화
from langchain_teddynote.graphs import visualize_graph
visualize_graph(app, xray=True)
```

## 성능 최적화 가이드

### 1. 상태 최적화

- 큰 데이터는 외부 저장소 사용
- 필요한 필드만 상태에 포함
- 메시지 히스토리 크기 제한

### 2. 노드 최적화

- 병렬 처리 가능한 작업은 Send API 활용
- 무거운 연산은 비동기 처리
- 캐싱 적극 활용

### 3. 모델 최적화

- 작업별 적절한 모델 선택
- 프롬프트 길이 최적화
- 구조화된 출력 활용

## 일반적인 함정과 해결책

### 1. 무한 루프 방지

```python
# 반복 횟수 제한
MAX_ITERATIONS = 10
if state["iteration_count"] >= MAX_ITERATIONS:
    return "finish"

# 재귀 제한 설정
config = RunnableConfig(recursion_limit=50)
```

### 2. 상태 동기화 문제

```python
# 명시적 상태 업데이트
return {
    "field_to_update": new_value,
    # 다른 필드는 건드리지 않음
}

# Annotated 필드 활용
messages: Annotated[List[BaseMessage], operator.add]
```

### 3. 에러 전파

```python
# 각 노드에서 에러 처리
try:
    result = risky_operation()
except Exception as e:
    return {"errors": [str(e)], "next": "error_handler"}

# 전역 에러 핸들러
workflow.add_node("error_handler", handle_errors)
```