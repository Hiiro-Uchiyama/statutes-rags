#!/usr/bin/env python3
"""
MCPエージェントベンチマーク評価スクリプト

4択法令データを用いてMCPエージェント（e-Gov API統合）の性能を評価します。
既存のVectorベース評価と同じフォーマットで結果を出力します。
"""
import argparse
import json
import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import re
from datetime import datetime
from dotenv import load_dotenv

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

# .envファイルの読み込み
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)

from app.core.rag_config import load_config as load_base_config
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever

# MCPモジュールの動的インポート
import importlib
mcp_module = importlib.import_module('examples.02_mcp_egov_agent')
MCPEgovPipeline = mcp_module.MCPEgovPipeline
load_mcp_config = mcp_module.load_config


def create_retriever(config):
    """設定に基づいてRetrieverを作成"""
    retriever_type = config.retriever.retriever_type
    index_path = Path(config.vector_store_path)
    
    if retriever_type == "vector":
        return VectorRetriever(
            embedding_model=config.embedding.model_name,
            index_path=str(index_path / "vector"),
            use_mmr=config.retriever.use_mmr,
            mmr_lambda=config.retriever.mmr_lambda,
            mmr_fetch_k_max=config.retriever.mmr_fetch_k_max
        )
    elif retriever_type == "bm25":
        return BM25Retriever(
            index_path=str(index_path / "bm25"),
            tokenizer=config.retriever.bm25_tokenizer
        )
    else:
        vector_retriever = VectorRetriever(
            embedding_model=config.embedding.model_name,
            index_path=str(index_path / "vector"),
            use_mmr=config.retriever.use_mmr,
            mmr_lambda=config.retriever.mmr_lambda,
            mmr_fetch_k_max=config.retriever.mmr_fetch_k_max
        )
        bm25_retriever = BM25Retriever(
            index_path=str(index_path / "bm25"),
            tokenizer=config.retriever.bm25_tokenizer
        )
        return HybridRetriever(
            vector_retriever,
            bm25_retriever,
            fusion_method=config.retriever.fusion_method,
            vector_weight=config.retriever.vector_weight,
            bm25_weight=config.retriever.bm25_weight,
            rrf_k=config.retriever.rrf_k,
            fetch_k_multiplier=config.retriever.fetch_k_multiplier
        )


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """評価データセットを読み込む"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'questions' in data:
        return data['questions']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unknown dataset format: {dataset_path}")


def create_multiple_choice_prompt(question: str, choices: str, context: str = "") -> str:
    """4択問題用のプロンプトを作成"""
    prompt = """あなたは日本の法律に精通した法律アシスタントです。以下の法令条文に基づいて、4択問題に答えてください。

【法令条文】
{context}

【問題文】
{question}

【選択肢】
{choices}

【指示】
上記の法令条文に基づいて、選択肢a、b、c、dの中から最も適切なものを1つ選んでください。
回答は必ず「a」「b」「c」「d」のいずれか1文字のみを返してください。説明は不要です。

回答: """
    
    return prompt.format(context=context, question=question, choices=choices)


def extract_answer(response: str) -> str:
    """LLM応答から回答(a/b/c/d)を抽出"""
    response = response.strip().lower()
    
    # パターン1: 単独の a, b, c, d
    if response in ['a', 'b', 'c', 'd']:
        return response
    
    # パターン2: "回答: a" や "答え: a" などの形式
    match = re.search(r'[回答答え][:：]\s*([abcd])', response)
    if match:
        return match.group(1)
    
    # パターン3: 最初に出現する a, b, c, d
    match = re.search(r'\b([abcd])\b', response)
    if match:
        return match.group(1)
    
    # 抽出失敗
    return ""


def evaluate_question(
    pipeline: MCPEgovPipeline,
    question_data: Dict[str, Any],
    force_api: bool = False,
    force_local: bool = False
) -> Dict[str, Any]:
    """
    1問を評価
    
    Returns:
        既存フォーマットと互換性のある結果辞書
    """
    question_text = question_data.get("question", "")
    choices = question_data.get("choices", {})
    correct_answer = question_data.get("answer", "").lower()
    question_id = question_data.get("id", "")
    
    # 選択肢の整形
    choices_text = "\n".join([
        f"{key}. {value}" for key, value in choices.items()
    ])
    
    start_time = time.time()
    
    try:
        # パイプラインで回答を取得
        result = pipeline.query(
            question_text,
            force_api=force_api,
            force_local=force_local
        )
        
        # コンテキストを取得
        contexts = result.get("contexts", [])
        context_text = "\n\n".join([
            f"[{i+1}] {ctx.get('law_title', '不明')} 第{ctx.get('article', '?')}条\n{ctx.get('text', '')}"
            for i, ctx in enumerate(contexts[:5])
        ])
        
        # 4択プロンプトを作成してLLMに送信
        prompt = create_multiple_choice_prompt(question_text, choices_text, context_text)
        response = pipeline.llm.invoke(prompt)
        
        # 回答を抽出
        predicted_answer = extract_answer(str(response))
        
        response_time = time.time() - start_time
        
        # 正答判定
        is_correct = (predicted_answer == correct_answer)
        
        # データソースの判定
        source = result.get("source", "unknown")
        metadata = result.get("metadata", {})
        api_used = metadata.get("num_api_documents", 0) > 0
        
        return {
            "question_id": question_id,
            "question": question_text,
            "correct": is_correct,
            "predicted": predicted_answer,
            "ground_truth": correct_answer,
            "response_time": response_time,
            "mcp_source": source,  # MCP特有の情報
            "api_used": api_used,
            "num_api_docs": metadata.get("num_api_documents", 0),
            "num_local_docs": metadata.get("num_local_documents", 0),
            "error": None
        }
    
    except Exception as e:
        response_time = time.time() - start_time
        return {
            "question_id": question_id,
            "question": question_text,
            "correct": False,
            "predicted": "",
            "ground_truth": correct_answer,
            "response_time": response_time,
            "mcp_source": "error",
            "api_used": False,
            "num_api_docs": 0,
            "num_local_docs": 0,
            "error": str(e)
        }


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """評価指標を計算"""
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    
    # データソース別の集計
    by_source = {}
    for r in results:
        source = r.get("mcp_source", "unknown")
        if source not in by_source:
            by_source[source] = {"total": 0, "correct": 0}
        by_source[source]["total"] += 1
        if r["correct"]:
            by_source[source]["correct"] += 1
    
    # ソース別の正答率を計算
    for source, stats in by_source.items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
    
    # API使用状況
    api_used_count = sum(1 for r in results if r.get("api_used", False))
    api_success_count = sum(1 for r in results if r.get("api_used", False) and r["correct"])
    
    # 平均応答時間
    avg_response_time = sum(r["response_time"] for r in results) / total if total > 0 else 0.0
    
    # エラー率
    error_count = sum(1 for r in results if r.get("error") is not None)
    
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "avg_time": avg_response_time,
        "error_rate": error_count / total if total > 0 else 0.0,
        "by_source": by_source,
        "api_metrics": {
            "api_used_count": api_used_count,
            "api_success_count": api_success_count,
            "api_usage_rate": api_used_count / total if total > 0 else 0.0,
            "api_success_rate": api_success_count / api_used_count if api_used_count > 0 else 0.0
        }
    }


def main():
    parser = argparse.ArgumentParser(description="MCPエージェントベンチマーク評価")
    parser.add_argument(
        "--data",
        type=Path,
        default=project_root / "datasets" / "lawqa_jp" / "data" / "selection.json",
        help="評価データセットのパス"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mcp_benchmark_results.json"),
        help="結果の出力先"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["api_preferred", "local_preferred", "api_only", "local_only"],
        default="api_preferred",
        help="MCPエージェントモード (api_preferred: API優先/ローカルフォールバック, local_preferred: ローカル優先, api_only: API強制, local_only: ローカル強制)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="評価する問題数（デバッグ用）"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MCPエージェント 4択問題ベンチマーク評価")
    print("=" * 80)
    print(f"データセット: {args.data}")
    print(f"モード: {args.mode}")
    print(f"出力先: {args.output}")
    print()
    
    # データセットの読み込み
    print("📥 データセットを読み込み中...")
    dataset = load_dataset(args.data)
    
    if args.samples:
        dataset = dataset[:args.samples]
        print(f"   問題数を{args.samples}件に制限")
    
    print(f"   総問題数: {len(dataset)}")
    print()
    
    # 設定の読み込み
    print("⚙️  設定を読み込み中...")
    base_config = load_base_config()
    mcp_config = load_mcp_config(validate=False)
    
    # モードに応じた設定
    if args.mode == "local_preferred":
        mcp_config.prefer_api = False
    elif args.mode == "api_only":
        mcp_config.prefer_api = True
        mcp_config.fallback_to_local = False
    elif args.mode == "local_only":
        mcp_config.prefer_api = False
        mcp_config.fallback_to_local = False
    
    print(f"   API優先: {mcp_config.prefer_api}")
    print(f"   ローカルフォールバック: {mcp_config.fallback_to_local}")
    print()
    
    # Retrieverの作成
    print("🔍 Retrieverを初期化中...")
    retriever = create_retriever(base_config)
    print("   ✓ Retriever初期化完了")
    print()
    
    # パイプラインの作成
    print("🤖 MCPエージェントパイプラインを初期化中...")
    pipeline = MCPEgovPipeline(config=mcp_config, retriever=retriever)
    print("   ✓ パイプライン初期化完了")
    print()
    
    # 評価の実行
    print("🚀 評価を開始します...")
    print()
    
    force_api = (args.mode == "api_only")
    force_local = (args.mode == "local_only")
    
    results = []
    for question_data in tqdm(dataset, desc="評価中", unit="問"):
        result = evaluate_question(
            pipeline,
            question_data,
            force_api=force_api,
            force_local=force_local
        )
        results.append(result)
    
    # 指標の計算
    print()
    print("=" * 80)
    print("📊 評価結果")
    print("=" * 80)
    
    metrics = calculate_metrics(results)
    
    print(f"全体正答率: {metrics['accuracy']:.2%}")
    print(f"総問題数: {metrics['total']}")
    print(f"正答数: {metrics['correct']}")
    print(f"平均応答時間: {metrics['avg_time']:.2f}秒")
    print(f"エラー率: {metrics['error_rate']:.2%}")
    print()
    
    print("データソース別正答率:")
    for source, stats in metrics['by_source'].items():
        print(f"  {source}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    print()
    
    if metrics['api_metrics']['api_used_count'] > 0:
        print("API使用状況:")
        print(f"  API使用率: {metrics['api_metrics']['api_usage_rate']:.2%}")
        print(f"  API使用回数: {metrics['api_metrics']['api_used_count']}")
        print(f"  API経由正答率: {metrics['api_metrics']['api_success_rate']:.2%}")
        print()
    
    # 結果の保存
    output_data = {
        "evaluation_type": "mcp_agent",
        "mode": args.mode,
        "timestamp": datetime.now().isoformat(),
        "dataset": str(args.data),
        "metrics": metrics,
        "results": results
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 結果を {args.output} に保存しました。")
    print()


if __name__ == "__main__":
    main()
