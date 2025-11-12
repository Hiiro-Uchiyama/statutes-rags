"""
判例データローダー

判例JSONファイルを読み込み、評価用のデータ形式に変換する。
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import random

logger = logging.getLogger(__name__)


class PrecedentLoader:
    """判例データローダー"""
    
    def __init__(self, precedent_dir: Path):
        """
        Args:
            precedent_dir: 判例データディレクトリのパス
        """
        self.precedent_dir = Path(precedent_dir)
        if not self.precedent_dir.exists():
            raise ValueError(f"Precedent directory not found: {precedent_dir}")
    
    def load_all_precedents(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        全ての判例をロード
        
        Args:
            limit: 読み込む判例数の上限（Noneの場合は全て）
        
        Returns:
            判例のリスト
        """
        precedents = []
        
        # 各年代ディレクトリを探索
        for year_dir in sorted(self.precedent_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            
            logger.info(f"Loading precedents from {year_dir.name}")
            
            # JSONファイルを検索（list.jsonは除外）
            json_files = [
                f for f in year_dir.glob("*.json")
                if f.name != "list.json" and f.name != "listup_info.json"
            ]
            
            for json_file in json_files:
                try:
                    precedent = self._load_precedent_file(json_file)
                    if precedent:
                        precedents.append(precedent)
                        
                        if limit and len(precedents) >= limit:
                            logger.info(f"Reached limit: {limit} precedents")
                            return precedents
                            
                except Exception as e:
                    logger.warning(f"Failed to load {json_file}: {e}")
                    continue
        
        logger.info(f"Loaded {len(precedents)} precedents")
        return precedents
    
    def _load_precedent_file(self, json_file: Path) -> Optional[Dict[str, Any]]:
        """
        1つの判例JSONファイルをロード
        
        Args:
            json_file: JSONファイルのパス
        
        Returns:
            判例データ（Noneの場合はスキップ）
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # リスト形式の場合はスキップ
        if isinstance(data, list):
            return None
        
        # 辞書形式でない場合はスキップ
        if not isinstance(data, dict):
            return None
        
        # 必要な情報が揃っているか確認
        if not data.get("case_gist") and not data.get("contents"):
            return None
        
        return {
            "file_path": str(json_file),
            "case_name": data.get("case_name", ""),
            "case_number": data.get("case_number", ""),
            "court_name": data.get("court_name", ""),
            "gist": data.get("gist", ""),
            "case_gist": data.get("case_gist", ""),
            "contents": data.get("contents", ""),
            "date": data.get("date", {}),
            "trial_type": data.get("trial_type", ""),
            "lawsuit_id": data.get("lawsuit_id", "")
        }
    
    def create_evaluation_question(self, precedent: Dict[str, Any]) -> Dict[str, Any]:
        """
        判例から評価用の質問を作成
        
        Args:
            precedent: 判例データ
        
        Returns:
            評価用の質問データ
        """
        # 事件の要旨を質問として使用
        question = precedent.get("case_gist", "")
        if not question:
            # case_gistがない場合はcontentsから要約を抽出
            contents = precedent.get("contents", "")
            # 最初の数行を質問として使用
            question = "\n".join(contents.split("\n")[:5])
        
        # 判例の要旨を正解として使用
        correct_answer = precedent.get("gist", "")
        if not correct_answer:
            # gistがない場合はcontentsから結論部分を抽出
            contents = precedent.get("contents", "")
            # "主文"や"理由"の前までを正解として使用
            if "主文" in contents:
                parts = contents.split("主文")
                if len(parts) > 1:
                    correct_answer = parts[1].split("\n")[0:10]  # 主文の最初の10行
                    correct_answer = "\n".join(correct_answer)
            else:
                # 最初の500文字を正解として使用
                correct_answer = contents[:500]
        
        return {
            "precedent_id": precedent.get("lawsuit_id", ""),
            "case_name": precedent.get("case_name", ""),
            "case_number": precedent.get("case_number", ""),
            "court_name": precedent.get("court_name", ""),
            "question": question,
            "correct_answer": correct_answer,
            "full_contents": precedent.get("contents", ""),
            "metadata": {
                "file_path": precedent.get("file_path", ""),
                "date": precedent.get("date", {}),
                "trial_type": precedent.get("trial_type", "")
            }
        }
    
    def load_evaluation_dataset(
        self,
        limit: Optional[int] = None,
        random_seed: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        評価用データセットをロード
        
        Args:
            limit: 読み込む判例数の上限
            random_seed: ランダムシード（指定するとランダムサンプリング）
        
        Returns:
            評価用データのリスト
        """
        precedents = self.load_all_precedents(limit=None)
        
        # ランダムサンプリング
        if random_seed is not None:
            random.seed(random_seed)
            if limit and limit < len(precedents):
                precedents = random.sample(precedents, limit)
        
        # 評価用の質問に変換
        evaluation_data = []
        for precedent in precedents:
            try:
                eval_data = self.create_evaluation_question(precedent)
                evaluation_data.append(eval_data)
            except Exception as e:
                logger.warning(f"Failed to create evaluation question: {e}")
                continue
        
        if limit and len(evaluation_data) > limit:
            evaluation_data = evaluation_data[:limit]
        
        logger.info(f"Created {len(evaluation_data)} evaluation questions")
        return evaluation_data

