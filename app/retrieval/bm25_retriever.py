"""
BM25ベースのキーワード検索Retriever
"""
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from rank_bm25 import BM25Okapi
import re

from .base import BaseRetriever, Document

logger = logging.getLogger(__name__)


class BM25Retriever(BaseRetriever):
    """BM25を使ったキーワード検索"""
    
    def __init__(
        self, 
        index_path: str = None, 
        tokenizer: Literal["auto", "sudachi", "janome", "mecab", "ngram", "simple"] = "auto"
    ):
        """
        Args:
            index_path: インデックス保存パス
            tokenizer: 使用するトークナイザー
                - "auto": 利用可能なものを自動選択（sudachi > janome > mecab > ngram）
                - "sudachi": SudachiPy（推奨、管理者権限不要）
                - "janome": Janome（軽量、管理者権限不要）
                - "mecab": MeCab（要インストール）
                - "ngram": 改良版n-gram（辞書不要）
                - "simple": 簡易トークナイザー（後方互換性用）
        """
        self.index_path = index_path
        self.bm25 = None
        self.documents = []
        self.tokenizer_type = tokenizer
        self.tokenizer = None
        self.tokenized_corpus: List[List[str]] = []
        
        # トークナイザーの初期化
        if not self._init_tokenizer():
            logger.warning("Failed to initialize any tokenizer, falling back to simple tokenizer")
            self.tokenizer_type = "simple"
        
        if index_path and Path(index_path).exists():
            self.load_index()
    
    def _init_tokenizer(self) -> bool:
        """トークナイザーの初期化
        
        Returns:
            bool: 初期化成功時True、失敗時False
        """
        if self.tokenizer_type == "auto":
            # 優先順位: sudachi > janome > mecab > ngram
            if self._init_sudachi():
                return True
            if self._init_janome():
                return True
            if self._init_mecab():
                return True
            # ngramは常に成功
            self._init_ngram()
            return True
        elif self.tokenizer_type == "sudachi":
            if not self._init_sudachi():
                logger.error("Failed to initialize Sudachi tokenizer")
                return False
            return True
        elif self.tokenizer_type == "janome":
            if not self._init_janome():
                logger.error("Failed to initialize Janome tokenizer")
                return False
            return True
        elif self.tokenizer_type == "mecab":
            if not self._init_mecab():
                logger.error("Failed to initialize MeCab tokenizer")
                return False
            return True
        elif self.tokenizer_type == "ngram":
            self._init_ngram()
            return True
        else:  # simple
            self.tokenizer_type = "simple"
            logger.info("Using simple tokenizer")
            return True
    
    def _init_sudachi(self) -> bool:
        """SudachiPyの初期化"""
        try:
            from sudachipy import tokenizer as sudachi_tokenizer
            from sudachipy import dictionary
            
            self.tokenizer = dictionary.Dictionary().create()
            self.tokenizer_type = "sudachi"
            logger.info("SudachiPy tokenizer initialized successfully")
            return True
        except ImportError:
            logger.debug("SudachiPy not available. Install with: pip install sudachipy sudachidict_core")
            return False
        except Exception as e:
            logger.debug(f"SudachiPy initialization failed: {e}")
            return False
    
    def _init_janome(self) -> bool:
        """Janomeの初期化"""
        try:
            from janome.tokenizer import Tokenizer
            
            self.tokenizer = Tokenizer()
            self.tokenizer_type = "janome"
            logger.info("Janome tokenizer initialized successfully")
            return True
        except ImportError:
            logger.debug("Janome not available. Install with: pip install janome")
            return False
        except Exception as e:
            logger.debug(f"Janome initialization failed: {e}")
            return False
    
    def _init_mecab(self) -> bool:
        """MeCabの初期化"""
        try:
            import MeCab
            self.tokenizer = MeCab.Tagger("-Owakati")
            self.tokenizer_type = "mecab"
            logger.info("MeCab tokenizer initialized successfully")
            return True
        except ImportError:
            logger.debug("MeCab not available. Install with: pip install mecab-python3")
            return False
        except Exception as e:
            logger.debug(f"MeCab initialization failed: {e}")
            return False
    
    def _init_ngram(self):
        """n-gramトークナイザーの初期化（常に成功）"""
        self.tokenizer_type = "ngram"
        logger.info("Using improved n-gram tokenizer")
    
    def tokenize(self, text: str) -> List[str]:
        """日本語テキストをトークン化"""
        try:
            if self.tokenizer_type == "sudachi":
                return self._tokenize_sudachi(text)
            elif self.tokenizer_type == "janome":
                return self._tokenize_janome(text)
            elif self.tokenizer_type == "mecab":
                return self._tokenize_mecab(text)
            elif self.tokenizer_type == "ngram":
                return self._tokenize_ngram(text)
            else:  # simple
                return self._simple_tokenize(text)
        except Exception as e:
            logger.warning(f"{self.tokenizer_type} tokenization failed: {e}. Using simple fallback.")
            return self._simple_tokenize(text)
    
    def _tokenize_sudachi(self, text: str) -> List[str]:
        """SudachiPyでトークン化"""
        from sudachipy import tokenizer
        
        mode = tokenizer.Tokenizer.SplitMode.C  # 最長一致
        tokens = [m.surface() for m in self.tokenizer.tokenize(text, mode)]
        return tokens
    
    def _tokenize_janome(self, text: str) -> List[str]:
        """Janomeでトークン化"""
        tokens = [token.surface for token in self.tokenizer.tokenize(text)]
        return tokens
    
    def _tokenize_mecab(self, text: str) -> List[str]:
        """MeCabでトークン化"""
        result = self.tokenizer.parse(text).strip()
        tokens = result.split()
        return tokens
    
    def _tokenize_ngram(self, text: str) -> List[str]:
        """改良版n-gramトークナイザー（2-gram + 3-gram）"""
        tokens = []
        
        # 日本語と英数字を分離
        pattern = r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+|[a-zA-Z0-9]+'
        matches = re.findall(pattern, text)
        
        for match in matches:
            if re.match(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', match):
                # 日本語: 2-gramと3-gramを生成
                if len(match) >= 2:
                    # 2-gram
                    for i in range(len(match) - 1):
                        tokens.append(match[i:i+2])
                    # 3-gram
                    if len(match) >= 3:
                        for i in range(len(match) - 2):
                            tokens.append(match[i:i+3])
                    # 完全一致用に全体も追加
                    tokens.append(match)
                else:
                    # 1文字の場合はそのまま
                    tokens.append(match)
            else:
                # 英数字: そのまま（小文字化）
                tokens.append(match.lower())
        
        return tokens
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """シンプルなトークン化（MeCabフォールバック）"""
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
    
    def _rebuild_index(self):
        """現在のトークナイズ済みコーパスからBM25インデックスを再構築"""
        if not self.tokenized_corpus:
            self.bm25 = None
            return

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def add_documents(self, documents: List[Dict[str, Any]], *, rebuild_index: bool = True):
        """ドキュメントを追加し、必要に応じてBM25インデックスを再構築"""
        if not documents:
            logger.warning("Empty document list provided, nothing to add")
            return
        
        new_documents = []
        new_tokens = []

        for doc in documents:
            text = doc.get("text", "")
            new_documents.append(doc)
            new_tokens.append(self.tokenize(text))

        self.documents.extend(new_documents)
        self.tokenized_corpus.extend(new_tokens)

        if rebuild_index:
            self._rebuild_index()
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        """BM25検索を実行"""
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if not query or not query.strip():
            logger.warning("Empty query provided, returning empty results")
            return []
        if self.bm25 is None or not self.documents:
            return []
        
        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            logger.warning(f"Query tokenization resulted in empty token list for query: '{query[:50]}...'")
            return []
        
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
        if not self.index_path:
            logger.warning("Cannot save BM25 index - index path not set")
            return

        if self.bm25 is None:
            logger.debug("BM25 index missing at save time, rebuilding before persistence")
            self._rebuild_index()
            
            # 再構築後も失敗した場合
            if self.bm25 is None:
                logger.warning("Cannot save BM25 index - BM25 not initialized after rebuild attempt")
                return
        
        try:
            index_path = Path(self.index_path)
            index_path.mkdir(parents=True, exist_ok=True)
            
            with open(index_path / "bm25.pkl", "wb") as f:
                pickle.dump(self.bm25, f)
            
            with open(index_path / "documents.pkl", "wb") as f:
                pickle.dump(self.documents, f)
            
            # トークナイズ済みコーパスを保存（メモリ節約のため）
            with open(index_path / "tokenized_corpus.pkl", "wb") as f:
                pickle.dump(self.tokenized_corpus, f)
            
            # トークナイザータイプを保存（互換性チェック用）
            with open(index_path / "tokenizer_info.pkl", "wb") as f:
                pickle.dump({"tokenizer_type": self.tokenizer_type}, f)
            
            logger.info(f"BM25 index saved to {index_path} ({len(self.documents)} documents, tokenizer: {self.tokenizer_type})")
        except Exception as e:
            logger.error(f"Error saving BM25 index to {self.index_path}: {e}", exc_info=True)
            raise
    
    def load_index(self):
        """インデックスをロード"""
        if not self.index_path:
            logger.warning("BM25 index path not set")
            return
        
        index_path = Path(self.index_path)
        
        bm25_path = index_path / "bm25.pkl"
        docs_path = index_path / "documents.pkl"
        tokenizer_info_path = index_path / "tokenizer_info.pkl"
        
        if not (bm25_path.exists() and docs_path.exists()):
            logger.info(f"BM25 index not found at {index_path}, will be created on first use")
            return
        
        try:
            # トークナイザー情報の確認
            if tokenizer_info_path.exists():
                with open(tokenizer_info_path, "rb") as f:
                    tokenizer_info = pickle.load(f)
                    saved_tokenizer = tokenizer_info.get("tokenizer_type", "unknown")
                    
                    if saved_tokenizer != self.tokenizer_type:
                        logger.warning(
                            f"Tokenizer mismatch: index was built with '{saved_tokenizer}', "
                            f"but current tokenizer is '{self.tokenizer_type}'. "
                            f"This may cause inconsistent results."
                        )
            
            with open(bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)
            
            with open(docs_path, "rb") as f:
                self.documents = pickle.load(f)
            
            # トークナイズ済みコーパスをロード（メモリ節約）
            tokenized_corpus_path = index_path / "tokenized_corpus.pkl"
            if tokenized_corpus_path.exists():
                with open(tokenized_corpus_path, "rb") as f:
                    self.tokenized_corpus = pickle.load(f)
                logger.info(f"BM25 index loaded from {index_path} ({len(self.documents)} documents, tokenizer: {self.tokenizer_type})")
            else:
                # 旧バージョンとの互換性: tokenized_corpus.pklが存在しない場合は再構築
                logger.warning(
                    f"tokenized_corpus.pkl not found at {index_path}. "
                    f"Rebuilding tokenized corpus (this may use significant memory)."
                )
                if self.documents:
                    self.tokenized_corpus = [
                        self.tokenize(doc.get("text", ""))
                        for doc in self.documents
                    ]
                else:
                    self.tokenized_corpus = []
                logger.info(f"BM25 index loaded from {index_path} ({len(self.documents)} documents, tokenizer: {self.tokenizer_type})")
        except Exception as e:
            logger.error(f"Error loading BM25 index from {index_path}: {e}", exc_info=True)
            # 状態をクリアして続行（新規インデックス作成可能な状態に）
            self.bm25 = None
            self.documents = []
            self.tokenized_corpus = []
