"""
BM25ベースのキーワード検索Retriever
"""
import pickle
from pathlib import Path
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import warnings

try:
    import MeCab
    MECAB_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    MECAB_AVAILABLE = False
    # MeCabが使えない場合は警告を出さずに静かにフォールバック
    # warnings.warn(f"MeCab not available: {e}. Using simple tokenizer as fallback.")

from .base import BaseRetriever, Document


class BM25Retriever(BaseRetriever):
    """BM25を使ったキーワード検索"""
    
    def __init__(self, index_path: str = None, use_mecab: bool = True):
        self.index_path = index_path
        self.bm25 = None
        self.documents = []
        self.use_mecab = use_mecab and MECAB_AVAILABLE
        
        if self.use_mecab:
            try:
                # MeCab環境変数の設定を試みる
                import os
                mecab_paths = [
                    "/home/jovyan/work/legal-rag/setup/lib/mecab/etc/mecabrc",
                    "/usr/local/etc/mecabrc",
                    "/etc/mecabrc"
                ]
                for path in mecab_paths:
                    if os.path.exists(path):
                        os.environ['MECABRC'] = path
                        break
                
                self.tokenizer = MeCab.Tagger("-Owakati")
            except Exception as e:
                # MeCabが使えない場合は警告を出さずに静かにフォールバック
                # warnings.warn(f"Failed to initialize MeCab: {e}. Using simple tokenizer.")
                self.use_mecab = False
                self.tokenizer = None
        else:
            self.tokenizer = None
        
        if index_path and Path(index_path).exists():
            self.load_index()
    
    def tokenize(self, text: str) -> List[str]:
        """日本語テキストをトークン化"""
        if self.use_mecab and self.tokenizer:
            try:
                return self.tokenizer.parse(text).strip().split()
            except Exception as e:
                warnings.warn(f"MeCab tokenization failed: {e}. Using simple tokenizer.")
                self.use_mecab = False
        
        # フォールバック: シンプルなトークン化
        return self._simple_tokenize(text)
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """シンプルなトークン化（MeCab不使用時のフォールバック）"""
        import re
        
        # 日本語文字（ひらがな、カタカナ、漢字）、英数字で分割
        tokens = []
        
        # パターン: 日本語1文字以上 または 英数字の連続
        pattern = r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+|[a-zA-Z0-9]+'
        matches = re.findall(pattern, text)
        
        for match in matches:
            # 日本語の場合は1文字ずつに分割（簡易的なn-gram）
            if re.match(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', match):
                # 2文字以上の場合はそのまま追加
                if len(match) >= 2:
                    tokens.append(match)
                # 1文字の場合も追加
                tokens.extend(list(match))
            else:
                # 英数字はそのまま
                tokens.append(match.lower())
        
        return tokens
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """ドキュメントを追加してBM25インデックスを構築"""
        self.documents.extend(documents)
        
        tokenized_corpus = [
            self.tokenize(doc.get("text", ""))
            for doc in self.documents
        ]
        
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        """BM25検索を実行"""
        if self.bm25 is None or not self.documents:
            return []
        
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = scores.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            text = doc.get("text", "")
            metadata = {k: v for k, v in doc.items() if k != "text"}
            
            results.append(Document(
                page_content=text,
                metadata=metadata,
                score=float(scores[idx])
            ))
        
        return results
    
    def save_index(self):
        """インデックスを保存"""
        if not self.index_path or not self.bm25:
            return
        
        index_path = Path(self.index_path)
        index_path.mkdir(parents=True, exist_ok=True)
        
        with open(index_path / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)
        
        with open(index_path / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)
        
        print(f"BM25 index saved to {index_path}")
    
    def load_index(self):
        """インデックスをロード"""
        index_path = Path(self.index_path)
        
        bm25_path = index_path / "bm25.pkl"
        docs_path = index_path / "documents.pkl"
        
        if bm25_path.exists() and docs_path.exists():
            with open(bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)
            
            with open(docs_path, "rb") as f:
                self.documents = pickle.load(f)
            
            print(f"BM25 index loaded from {index_path}")
