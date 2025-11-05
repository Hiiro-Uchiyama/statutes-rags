# 技術調査レポート

examples/ 配下の実装に必要な技術の調査結果をまとめます。

## 1. LangGraph

### 概要

LangGraphは、LangChainを拡張したフレームワークで、複雑なエージェントワークフローの構築に特化しています。

### 特徴

- グラフ構造によるワークフロー定義
- 状態管理（StateGraph）
- 条件分岐、ループ、並列処理のサポート
- LangChainとの互換性
- チェックポイント機能（処理の中断・再開）
- 決定論的な制御が可能

### インストール

```bash
pip install langgraph
```

### 基本的な使用例

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

# 状態の定義
class State(TypedDict):
    query: str
    documents: list
    answer: str
    iteration: int

# グラフの作成
workflow = StateGraph(State)

# ノード（処理ステップ）の追加
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("reason", reason_about_documents)
workflow.add_node("validate", validate_answer)

# エッジ（フロー）の追加
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "reason")
workflow.add_conditional_edges(
    "reason",
    should_continue,  # 条件判定関数
    {
        "continue": "retrieve",  # 再検索
        "validate": "validate",  # 検証へ
    }
)
workflow.add_edge("validate", END)

# コンパイル
app = workflow.compile()

# 実行
result = app.invoke({"query": "質問文"})
```

### 利点

- 複雑なエージェントワークフローの可視化
- 既存のLangChain資産の活用
- 状態管理が明示的
- デバッグが容易

### 欠点

- 学習曲線がやや急
- ドキュメントが発展途上（2024年時点）

### 採用根拠

- 既存のRAGPipelineがLangChainベースである
- Agentic RAGに必要な反復処理とstate管理が容易
- 複雑度別のワークフロー分岐が実装しやすい

## 2. Model Context Protocol (MCP)

### 概要

MCPは、AIモデルと外部データソース・ツールを接続するための標準化されたプロトコルです。

### 現状

2024年11月時点での調査では、以下の状況が確認されました：

- Anthropic社が開発・推進
- Python実装の可能性あり
- 公式ドキュメントやGitHubリポジトリの情報が限定的

### 暫定的な実装方針

公式SDKの詳細が不明確なため、以下のアプローチを採用します：

#### アプローチ1: 簡易的なプロトコル実装

MCPの概念を参考に、独自の軽量プロトコルを実装：

```python
class SimpleMCPServer:
    """MCPの概念に基づく簡易サーバ"""
    
    def __init__(self):
        self.tools = {}
    
    def register_tool(self, name, func, description):
        """ツールの登録"""
        self.tools[name] = {
            "function": func,
            "description": description
        }
    
    def execute(self, tool_name, **kwargs):
        """ツールの実行"""
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")
        
        return self.tools[tool_name]["function"](**kwargs)
    
    def list_tools(self):
        """利用可能なツール一覧"""
        return [
            {"name": name, "description": tool["description"]}
            for name, tool in self.tools.items()
        ]
```

#### アプローチ2: LangChain Toolsの活用

LangChainの既存Toolインターフェースを活用：

```python
from langchain.tools import Tool

# e-Gov API用ツールの定義
egov_search_tool = Tool(
    name="search_law",
    func=search_law_function,
    description="法令名または法令番号で法令を検索します"
)

egov_get_tool = Tool(
    name="get_law_content",
    func=get_law_content_function,
    description="法令番号を指定して法令全文を取得します"
)

# エージェントでの使用
from langchain.agents import initialize_agent

agent = initialize_agent(
    tools=[egov_search_tool, egov_get_tool],
    llm=llm,
    agent="zero-shot-react-description"
)
```

### 採用方針

- MVP実装では **アプローチ2（LangChain Tools）** を採用
- 将来的にMCPの公式仕様が確立されれば移行を検討
- 独自実装は最小限に抑え、標準的なインターフェースを使用

## 3. e-Gov法令API

### 概要

デジタル庁が提供する法令データAPIです。

### API仕様

#### ベースURL

```
https://elaws.e-gov.go.jp/api/1/
```

#### 主要エンドポイント

1. **法令一覧取得**
   ```
   GET /lawlists/1
   ```
   
2. **法令データ取得**
   ```
   GET /lawdata/{法令番号}
   ```

3. **キーワード検索**（v2で実装予定）
   ```
   現在は限定的
   ```

### Python実装例

```python
import requests
from xml.etree import ElementTree
import xmltodict

class EGovAPIClient:
    """e-Gov法令API クライアント"""
    
    BASE_URL = "https://elaws.e-gov.go.jp/api/1"
    
    def __init__(self, timeout=30):
        self.timeout = timeout
        self.session = requests.Session()
    
    def get_law_data(self, law_number):
        """法令データの取得"""
        url = f"{self.BASE_URL}/lawdata/{law_number}"
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # XMLをdictに変換
            data = xmltodict.parse(response.content)
            return data
            
        except requests.Timeout:
            raise Exception(f"Timeout getting law {law_number}")
        except requests.HTTPError as e:
            raise Exception(f"HTTP error: {e}")
    
    def search_laws(self, keyword):
        """法令検索（法令一覧から検索）"""
        url = f"{self.BASE_URL}/lawlists/1"
        
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        
        data = xmltodict.parse(response.content)
        
        # キーワードでフィルタリング
        laws = data.get('DataRoot', {}).get('ApplData', {}).get('LawNameListInfo', [])
        filtered = [
            law for law in laws
            if keyword in law.get('LawName', '')
        ]
        
        return filtered
```

### 制約事項

- レート制限：公開情報なし（常識的な範囲で使用）
- タイムアウト：30秒を推奨
- データ形式：XML（JSON化が必要）
- 公式Python SDK：なし

### 採用方針

- requestsライブラリで独自実装
- xmltodictでXML→JSON変換
- エラーハンドリングとリトライ機構を実装
- ローカルキャッシュとのハイブリッド運用

## 4. Agentic RAG実装パターン

### パターン1: Reflexion（反射）

エージェントが自己評価し、改善を繰り返すパターン：

```python
def agentic_rag_reflexion(query, max_iterations=3):
    iteration = 0
    answer = None
    
    while iteration < max_iterations:
        # 検索と回答生成
        documents = retrieve(query)
        answer = generate_answer(query, documents)
        
        # 自己評価
        evaluation = self_evaluate(answer, documents)
        
        if evaluation["score"] > 0.8:
            break
        
        # 改善のための追加情報取得
        query = refine_query(query, evaluation["feedback"])
        iteration += 1
    
    return answer
```

### パターン2: ReAct（Reasoning + Acting）

推論とアクションを交互に実行するパターン：

```python
def agentic_rag_react(query):
    thought = analyze_query(query)
    
    if thought["needs_search"]:
        action = "search"
        observation = search(thought["search_query"])
        
        thought = reason_about_observation(observation)
        
        if thought["needs_more_info"]:
            action = "refine_search"
            observation = search(thought["refined_query"])
        
        answer = generate_final_answer(observation)
    else:
        answer = generate_direct_answer(query)
    
    return answer
```

### パターン3: Multi-Agent Collaboration

複数の専門エージェントが協調するパターン：

```python
def agentic_rag_multi_agent(query):
    # Manager が分析
    task_plan = manager_agent.plan(query)
    
    # Retrieval Agent が検索
    documents = retrieval_agent.search(
        query, 
        strategy=task_plan["search_strategy"]
    )
    
    # Reasoning Agent が推論
    reasoning = reasoning_agent.analyze(documents, query)
    
    # Validation Agent が検証
    validation = validation_agent.verify(reasoning, documents)
    
    if validation["is_valid"]:
        return reasoning
    else:
        # 再試行
        return agentic_rag_multi_agent(
            manager_agent.refine_query(query, validation["feedback"])
        )
```

### 採用方針

01_agentic_ragでは **パターン3（Multi-Agent Collaboration）** を採用：

- 役割分担が明確
- 各エージェントを独立してテスト可能
- 拡張性が高い
- 複雑度別の処理分岐が容易

## 5. 評価データセット

### 既存データセット

#### デジタル庁 4択法令データ（lawqa_jp）

```
datasets/lawqa_jp/data/selection.json
```

- 問題数：調査が必要（おそらく数百問）
- 形式：4択問題
- カバー範囲：主要法令
- 難易度：基礎から中級

### 追加候補データセット

#### 1. 司法試験過去問

- 入手可能性：公開されている年度あり
- 難易度：高
- 形式：多肢選択、論述
- ライセンス：要確認

#### 2. 法律相談サイトのQ&A

- 入手可能性：スクレイピング（著作権要注意）
- 難易度：実務的
- 形式：自由形式
- ライセンス：通常は不可

#### 3. 自作評価セット

複雑度別の評価セット作成：

```python
evaluation_cases = {
    "simple": [
        # 単一条文の直接的な質問
        {
            "question": "民法第1条の基本原則は何ですか？",
            "expected_law": "民法",
            "expected_article": "1",
            "complexity": "low"
        }
    ],
    "medium": [
        # 複数条文の関連付けが必要
        {
            "question": "未成年者が契約を取り消す場合の要件は？",
            "expected_laws": ["民法"],
            "expected_articles": ["5", "121"],
            "complexity": "medium"
        }
    ],
    "complex": [
        # 複数法令にまたがる推論が必要
        {
            "question": "個人情報を含むデータの第三者提供において...",
            "expected_laws": ["個人情報保護法", "民法"],
            "expected_articles": ["27", "709"],
            "complexity": "high"
        }
    ]
}
```

### 採用方針

- 基本評価：既存の4択データを使用
- 複雑度評価：自作の評価セットを作成（各難易度10-20問）
- 将来的拡張：公開可能な司法試験問題を追加検討

## 6. 依存関係まとめ

### 必須ライブラリ

```toml
[project.optional-dependencies]
examples = [
    # エージェントフレームワーク
    "langgraph>=0.2.0",
    
    # API関連
    "httpx>=0.25.0",  # 既存に含まれる
    "xmltodict>=0.13.0",  # XML→JSON変換
    
    # トレーシング・デバッグ（オプション）
    "langsmith>=0.1.0",
]
```

### インストールコマンド

```bash
# 既存の依存関係
uv pip install -e .

# examples用の追加依存
uv pip install langgraph xmltodict

# オプション: トレーシング用
uv pip install langsmith
```

## 7. 実装優先順位

### Phase 1: Agentic RAG（01_agentic_rag）

理由：
- LangGraphの学習コストを最初に吸収
- 既存RAGコンポーネントの理解が深まる
- 他の実装の基盤となる

### Phase 2: MCP e-Gov Agent（02_mcp_egov_agent）

理由：
- API連携パターンの確立
- 外部データソース統合の経験
- 他の実装でも活用可能

### Phase 3: Multi-Agent Debate（03_multi_agent_debate）

理由：
- Phase 1のエージェント設計を応用
- より高度なエージェント間連携

### Phase 4: Legal Case Generator（04_legal_case_generator）

理由：
- 最も複雑なワークフロー
- 前段階の経験を活用

## 8. 技術的リスクと対策

### リスク1: LangGraphの学習曲線

対策：
- 公式ドキュメントとサンプルコードの精読
- 小規模なプロトタイプから開始
- コミュニティフォーラムの活用

### リスク2: e-Gov APIの不安定性

対策：
- ローカルデータへのフォールバック
- リトライ機構の実装
- キャッシング戦略

### リスク3: LLM呼び出しコストと時間

対策：
- エージェント呼び出しの最小化
- キャッシングの積極的活用
- 簡単な質問は既存RAGで処理

### リスク4: 評価の客観性

対策：
- 複数の評価指標を併用
- ベースラインとの定量比較
- エラーケースの詳細分析

## まとめ

本調査に基づき、以下の技術スタックで実装を進めます：

- エージェントフレームワーク: LangGraph
- API連携: LangChain Tools + 独自実装
- e-Gov API: requests + xmltodict
- 評価: 既存データ + 自作評価セット
- その他: 既存のRAGコンポーネントを最大限活用

次のステップは、01_agentic_ragの詳細設計とドキュメント作成です。
