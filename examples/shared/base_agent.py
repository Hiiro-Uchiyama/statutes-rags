"""
基底エージェントクラス

全てのエージェントの基底となるクラスを定義します。
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging
import time

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    全エージェントの基底クラス
    
    各エージェントは、このクラスを継承し、execute()メソッドを実装します。
    """
    
    def __init__(self, llm, config):
        """
        Args:
            llm: LLMインスタンス（LangChain LLM）
            config: 設定オブジェクト
        """
        self.llm = llm
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        エージェントの主要な処理を実行
        
        Args:
            input_data: 入力データ
        
        Returns:
            処理結果
        """
        raise NotImplementedError
    
    def _format_documents(self, documents: List[Any]) -> str:
        """
        ドキュメントリストをLLM用のコンテキスト文字列に整形
        
        Args:
            documents: Documentオブジェクトのリスト
        
        Returns:
            整形されたコンテキスト文字列
        """
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            meta = doc.metadata
            law_title = meta.get("law_title", "不明")
            article = meta.get("article", "")
            paragraph = meta.get("paragraph", "")
            item = meta.get("item", "")
            
            header = f"[{i}] {law_title}"
            if article:
                header += f" 第{article}条"
            if paragraph:
                header += f" 第{paragraph}項"
            if item:
                header += f" 第{item}号"
            
            context_parts.append(f"{header}\n{doc.page_content}\n")
        
        return "\n".join(context_parts)
    
    def _safe_llm_invoke(self, prompt: str, timeout: Optional[int] = None) -> Optional[str]:
        """
        LLM呼び出しのラッパー（エラーハンドリング付き）
        
        Args:
            prompt: プロンプト文字列
            timeout: タイムアウト秒数（オプション）
        
        Returns:
            LLMの応答、またはエラー時はNone
        """
        attempts = max(int(getattr(self.config, "llm_retry_attempts", 1)), 1)
        delay = float(getattr(self.config, "llm_retry_delay", 0.0))
        last_error: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            try:
                kwargs = {}
                if timeout is not None:
                    # 一部のLLMクラスでは request_timeout を参照する
                    kwargs["timeout"] = timeout
                    kwargs["request_timeout"] = timeout

                response = self.llm.invoke(prompt, **kwargs)
                return response.strip() if isinstance(response, str) else str(response).strip()
            except TimeoutError as exc:
                last_error = exc
                self.logger.warning(
                    "LLM request timeout (attempt %d/%d)", attempt, attempts
                )
            except Exception as exc:  # pylint: disable=broad-except
                last_error = exc
                self.logger.warning(
                    "LLM invocation error (attempt %d/%d): %s",
                    attempt,
                    attempts,
                    exc,
                    exc_info=True
                )

            if attempt < attempts and delay > 0:
                time.sleep(delay)

        if last_error:
            self.logger.error("LLM invocation failed after %d attempts: %s", attempts, last_error)
        return None
    
    def _parse_json_response(self, response: str, default: Optional[Dict] = None) -> Dict[str, Any]:
        """
        JSON形式の応答をパース
        
        Args:
            response: LLMの応答文字列
            default: パース失敗時のデフォルト値
        
        Returns:
            パースされた辞書、または失敗時はdefault
        """
        import json
        import re
        
        if default is None:
            default = {}
        
        try:
            # JSONブロックを抽出（```json...```形式）
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                response = json_match.group(1)
            
            # パース
            return json.loads(response)
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON parse error: {e}")
            return default
        except Exception as e:
            self.logger.error(f"Unexpected error parsing JSON: {e}")
            return default

