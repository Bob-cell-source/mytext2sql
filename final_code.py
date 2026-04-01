import os
import json
import re
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from sql_exe import execute_sql_with_pymysql     
import difflib       
import uuid
from openai import OpenAI
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import numpy as np
from dataclasses import dataclass
import pickle
import hashlib
import logging
from datetime import datetime

output_lock = threading.Lock()
debug_file_lock = threading.Lock()
log_lock = threading.Lock()

def setup_logging():
    """配置日志系统"""
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"sql_generation_{timestamp}.log"
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # 创建prompt日志目录
    prompt_log_dir = log_dir / f"prompts_{timestamp}"
    prompt_log_dir.mkdir(exist_ok=True)
    
    return log_file, prompt_log_dir

def log_prompt(prompt_log_dir: Path, sql_id: str, stage: str, content: str, attempt: int = 0):
    """记录prompt内容到单独文件"""
    with log_lock:
        filename = f"{sql_id}_attempt{attempt}_{stage}.txt"
        prompt_file = prompt_log_dir / filename
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(f"=== SQL ID: {sql_id} ===\n")
            f.write(f"=== Stage: {stage} ===\n")
            f.write(f"=== Attempt: {attempt} ===\n")
            f.write(f"=== Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
            f.write(content)
        logging.info(f"📝 Prompt saved: {filename}")

SYSTEM_PROMPT = """
        # Role
        StarRocks SQL Expert. 严格遵守 MySQL 5.7+ / StarRocks 语法规范与数仓逻辑。

        # 1. 语法与开发红线 (Critical)
        - **方言约束**: 仅限 StarRocks/MySQL。严禁 Hive/Spark 语法 (如 `LATERAL VIEW`, `to_date()`, 隐式 `date_sub`, `FIELD`).
            - 拼接: `CONCAT(a,b)` (禁用 `||`)
            - 日期: `DATE_ADD/SUB(dt, INTERVAL 1 DAY)`
            - 聚合: 严格遵守 `ONLY_FULL_GROUP_BY`
        - **防错机制**:
            - 分母防零: `COALESCE(sum(a)/NULLIF(sum(b),0), 0)`
            - 排除逻辑: `LEFT JOIN ... WHERE b.id IS NULL` (优于 `NOT IN`)
            - 字符串日期: 禁止 `date_format`，使用 `SUBSTR` 或 `STR_TO_DATE`。

        # 2. 数仓与表规范
        - **表后缀逻辑**:
            - `_di` (增量): **必须**带时间范围 `dtstatdate BETWEEN 'start' AND 'end'`。
            - `_df` (全量): **必须**指定单日快照 `dtstatdate = 'target_date'`。
        - **ID 体系 (绝对禁止直接关联)**:
            - **Global ID**: `suserid` (QQ/WX, 跨游戏/大盘)
            - **Game ID**: `vplayerid`/`iuserid` (单游戏角色)
            - **关联路径**: 必须通过 `dim_*_gplayerid2qqwxid_df` 进行转换 (GameID -> Mapping -> GlobalID)。

        # 3. 核心业务逻辑
        - **新进判定**: `iregdate = dtstatdate`。
        - **FPS 对局统计**:
            - 局数: `COUNT(DISTINCT matchid)`
            - 人数: `COUNT(DISTINCT vplayerid)`
        - **特殊位图 (Bitmap)**:
            - 通用: 左第1位 = 当日。
            - **勇者盟约 (iactivity)**: 右第1位 = 当日。计算留存必须先 `REVERSE` 或反向截取。

        # 4. 生成策略
        1. **思考**: 检查 ID 类型是否匹配？表后缀是否对应正确的日期过滤？语法是否含 Hive 特性？
        2. **结构**: 优先使用 CTE (`WITH ... AS`) 拆解逻辑。
        3. **输出**: SQL 必须可直接执行，字段名严格匹配需求。
        """

DIAGNOSTIC_PROMPT = """
        在生成最终SQL之前,你需要先执行一些预备sql。请生成3条诊断SQL,用于最终生成SQL时参考。

        要求:
        - 查询你需要使用的表格的关键字段的取值和分布情况
        - 查询要对结果很有帮助且简洁高效
        - 每条SQL单独一行
        - 只返回SQL语句,不要任何解释
        - 使用合法的StarRocks语法

        请输出诊断SQL(每行一条):
        """

FINAL_SQL_PROMPT = """
        根据上述诊断查询的结果,现在请生成最终的StarRocks查询语句来回答用户问题。

        要求:
        - 只输出一条完整的SQL语句
        - 不要添加任何解释文字
        - 确保StarRocks语法正确且能直接执行
        """

QUESTION_REWRITE_PROMPT = """
你是一个专业的数据分析需求理解专家。你的任务是评估用户的原始问题是否清晰，并在必要时进行重述。

重述原则：
1. **优先保持原问题**：如果原问题已经清晰、完整、无歧义，直接返回原问题（可做轻微语言润色）
2. **使用自然语言**：重述时使用业务语言，避免直接列举字段名、枚举值等技术细节
   - ❌ 错误示例："sgamecode in ('game1','game2','game3')"
   - ✅ 正确示例："指定的几款竞品游戏"
3. **理解业务意图**：结合业务知识（knowledge）理解术语含义
4. **明确查询目标**：说清楚要统计什么、条件是什么、输出什么
5. **消除歧义**：如果原问题有多种理解，补充必要的上下文
6. **保持简洁**：不要添加解释性文字或冗余信息

输出要求：
- 只输出一个问题描述（原问题或重述后的问题）
- 不要有任何前缀如"重述："、"问题："、"优化后："等
- 使用通俗易懂的业务语言，让不懂SQL的人也能理解
"""

DATASET_PATH = r".\data\final_dataset.json"
SCHEMA_PATH =  r".\data\schema_with.json"
OUTPUT_PATH = r".\data\final_sql.json"
INSERT_SQL_PATH = r".\data\insert_sql.json"
API_KEY = "sk-b7e572574abe48979d107aab4878aa8e"

# 问题重述配置
ENABLE_QUESTION_REWRITE = True  # 是否启用问题重述
REQUIRE_USER_CONFIRMATION = True  # 是否需要用户确认重述结果

@dataclass
class SQLVectorEntry:
    """SQL向量条目数据类"""
    sql_id: str
    question: str
    table_list: List[str]
    knowledge: str
    schema: str
    key_fields: Optional[Dict[str, str]]
    logic_pattern: Optional[List[str]]
    nl_to_schema_map: Optional[Dict[str, str]]
    common_pitfalls: Optional[List[str]]
    embedding: np.ndarray

class SQLVectorDatabase:
    """SQL向量数据库类"""
    
    def __init__(self, api_key: str = None):
        """
        初始化向量数据库
        
        Args:
            api_key: DashScope API密钥，如果不提供则从环境变量读取
        """
        self.api_key = API_KEY
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.vector_entries: List[SQLVectorEntry] = []
        
    def _get_embedding(self, text: str, dimensions: int = 1024) -> np.ndarray:
        """
        获取文本的向量表示
        
        Args:
            text: 输入文本
            dimensions: 向量维度
            
        Returns:
            向量数组
        """
        try:
            response = self.client.embeddings.create(
                model="text-embedding-v4",
                input=text,
                dimensions=dimensions,
                encoding_format="float"
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"获取向量失败: {e}")
            return np.zeros(dimensions)
    
    def _combine_text_for_embedding(self, question: str, knowledge: str, table_list: List[str]) -> str:
        """
        组合question、knowledge和table_list为一个文本用于向量化
        
        Args:
            question: 问题描述
            knowledge: 知识内容
            table_list: 表列表
            
        Returns:
            组合后的文本
        """
        tables_text = "相关表: " + ", ".join(table_list)
        combined = f"问题: {question}\n\n{tables_text}\n\n知识: {knowledge}"
        return combined
    
    def load_database(self, filepath: str = "vector_database.pkl"):
        """
        从文件加载向量数据库
        
        Args:
            filepath: 文件路径
        """
        with open(filepath, 'rb') as f:
            self.vector_entries = pickle.load(f)
        print(f"向量数据库已从 {filepath} 加载，共 {len(self.vector_entries)} 条记录")
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算余弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            相似度分数
        """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def search(self, 
               question: str, 
               knowledge: str = "", 
               table_list: List[str] = None,
               top_k: int = 3) -> List[Dict[str, Any]]:
        """
        检索最相似的SQL示例
        
        Args:
            question: 查询问题
            knowledge: 相关知识（可选）
            table_list: 表列表（可选）
            top_k: 返回前k个结果
            
        Returns:
            检索结果列表
        """
        if not self.vector_entries:
            raise ValueError("向量数据库为空，请先构建或加载数据库")
        
        # 准备查询文本并生成向量
        if table_list is None:
            table_list = []
        query_text = self._combine_text_for_embedding(question, knowledge, table_list)
        query_embedding = self._get_embedding(query_text)
        
        # 计算相似度
        similarities = []
        for entry in self.vector_entries:
            similarity = self._cosine_similarity(query_embedding, entry.embedding)
            similarities.append((similarity, entry))
        
        # 排序并返回top_k结果
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for similarity, entry in similarities[:top_k]:
            result = {
                'similarity': float(similarity),
                'question': entry.question,
                'table_list': entry.table_list,
                'knowledge': entry.knowledge,
                'schema': entry.schema,
                'key_fields': entry.key_fields,
                'logic_pattern': entry.logic_pattern,
                'nl_to_schema_map': entry.nl_to_schema_map,
                'common_pitfalls': entry.common_pitfalls
            }
            results.append(result)
        
        return results

def extract_and_normalize_result(result_list: List[Dict[str, Any]]) -> str:
    """
    将 result 列表规范化为可哈希的字符串指纹。
    逻辑：
    1. 将每一行转为 tuple(str(v)...)，忽略列名，保留列值顺序。
    2. 使用 Counter 统计每种行出现的次数（忽略行顺序）。
    3. 将 Counter 排序并序列化为字符串。
    """
    if not result_list:
        return "empty_result"

    counter: Counter = Counter()
    for item in result_list:
        if isinstance(item, dict):
            row_vals = tuple("" if v is None else str(v) for v in item.values())
        else:
            row_vals = (str(item),)
        counter[row_vals] += 1
    
    sorted_items = sorted(counter.items(), key=lambda x: x[0])
    return json.dumps(sorted_items, ensure_ascii=False)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_schema_index(schema_json: List[Dict]) -> Dict[str, Dict]:
    """
    将 schema.json 转为 {table_name: {description: str, columns: set([...]), column_details: Dict[col, Dict]}} 的索引。
    """
    index = {}
    for t in schema_json:
        name = t.get("table_name")
        cols = set()
        col_details = {}
        for c in t.get("columns", []):
            colname = c.get("col")
            if colname:
                cols.add(colname)
                col_details[colname] = {
                    "type": c.get("type", ""),
                    "description": c.get("description", ""),
                    "top_values": c.get("top_values", [])  # 添加高频值支持
                }
        index[name] = {
            "description": t.get("table_description", ""),
            "columns": cols,
            "column_details": col_details
        }
    return index

def truncate_insert_values(sql_text: str, max_rows: int = 3) -> str:
    """
    将多行 INSERT 中的 VALUES 选择最不相似的 max_rows 行。
    """
    try:
        m = re.search(r'\bVALUES\b', sql_text, flags=re.IGNORECASE)
        if not m:
            return sql_text
        header = sql_text[:m.end()]
        rest = sql_text[m.end():]

        all_rows = []
        i = 0
        n = len(rest)
        in_str = False

        while i < n:
            ch = rest[i]
            if ch == "'":
                in_str = not in_str
                i += 1
                continue
            if not in_str and ch == '(':
                depth = 1
                j = i + 1
                in_str2 = False
                while j < n and depth > 0:
                    c = rest[j]
                    if c == "'" and rest[j - 1] != '\\':
                        in_str2 = not in_str2
                    elif not in_str2:
                        if c == '(':
                            depth += 1
                        elif c == ')':
                            depth -= 1
                    j += 1
                group = rest[i:j]
                all_rows.append(group)
                i = j
            else:
                i += 1

        if not all_rows:
            return sql_text

        if len(all_rows) <= max_rows:
            selected_rows = all_rows
        else:
            selected_rows = [all_rows[0]]
            remaining_rows = all_rows[1:]
            
            while len(selected_rows) < max_rows and remaining_rows:
                best_row = None
                max_diff_score = -1
                
                for row in remaining_rows:
                    min_similarity = 1.0
                    for selected_row in selected_rows:
                        similarity = difflib.SequenceMatcher(None, row, selected_row).ratio()
                        if similarity < min_similarity:
                            min_similarity = similarity
                    
                    diff_score = 1.0 - min_similarity
                    if diff_score > max_diff_score:
                        max_diff_score = diff_score
                        best_row = row
                
                if best_row:
                    selected_rows.append(best_row)
                    remaining_rows.remove(best_row)
                else:
                    selected_rows.append(remaining_rows.pop(0))

        header_out = header.rstrip() + "\n"
        out_sql = header_out + ",\n ".join(selected_rows) + ";"
        return out_sql
    except Exception:
        return sql_text

def _get_llm_client_and_model() -> Tuple[OpenAI, str]:
    """
    读取 API Key 与 Base URL，返回 OpenAI 兼容客户端与模型名。
    """
    if OpenAI is None:
        raise RuntimeError("OpenAI client not available (openai 库未安装或无法导入)")

    api_key = API_KEY
    if not api_key:
        raise RuntimeError("未配置 OPENAI_API_KEY 或 DASHSCOPE_API_KEY，无法调用大模型")

    base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model_name = os.getenv("LLM_MODEL", "qwen3-max")

    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model_name

def _build_schema_snippets_for_tables(table_list: List[str], schema_index: Dict[str, Dict]) -> List[str]:
    try:
        insert_items = load_json(INSERT_SQL_PATH)
    except Exception:
        insert_items = []
    insert_index: Dict[str, List[Dict[str, str]]] = {}
    for obj in insert_items:
        tname = obj.get("table_name")
        sql_txt = obj.get("insert_sql")
        sid = obj.get("sql_id")
        if tname and sql_txt:
            insert_index.setdefault(tname, []).append({"sql_id": sid, "insert_sql": sql_txt})

    schema_snippets = []
    for tbl in table_list or []:
        meta = schema_index.get(tbl, {})
        cols = meta.get("columns", [])
        desc = meta.get("description", "")
        col_details = []
        for col in cols:
            col_info = meta.get("column_details", {}).get(col, {})
            col_type = col_info.get("type", "")
            col_desc = col_info.get("description", "")
            # 添加高频值信息（如果有）
            top_values = col_info.get("top_values", [])
            col_line = f"{col}({col_type}): {col_desc}"
            if top_values:
                col_line += f" [常见值: {', '.join(str(v) for v in top_values[:5])}]"
            col_details.append(col_line)

        snippet = f"Table: {tbl}\nDescription: {desc}\nColumns:\n" + "\n".join(col_details)

        samples = insert_index.get(tbl, [])
        if samples:
            first_sample = samples[0]["insert_sql"]
            snippet += f"\n\nExample rows:\n"
            snippet += truncate_insert_values(first_sample, max_rows=2)
        schema_snippets.append(snippet)
    return schema_snippets

def extract_golden_examples(dataset_path: str) -> List[Dict]:
    """提取 golden_sql = True 的样本"""
    dataset = load_json(dataset_path)
    return [item for item in dataset if item.get("golden_sql") is True]

def get_top_k_similar_examples(question: str, examples: List[Dict], k: int = 3) -> List[Dict]:
    """使用 difflib 计算问题相似度并返回 top-k"""
    similarities = []
    for ex in examples:
        ex_question = ex.get("question", "")
        ratio = difflib.SequenceMatcher(None, question, ex_question).ratio()
        similarities.append((ratio, ex))
    
    top_k = sorted(similarities, key=lambda x: x[0], reverse=True)[:k]
    return [ex for _, ex in top_k]

def build_few_shot_examples(examples: List[Dict]) -> str:
    """构造 few-shot 示例文本"""
    parts = []
    for idx, ex in enumerate(examples):
        q = ex.get("question", "").strip()
        sql = ex.get("sql", "").strip()
        parts.append(f"Example {idx+1}:\nQuestion: {q}\nSQL:\n{sql}")
    return "\n\n".join(parts)

def run_sql_and_get_feedback(sql_id: str, sql_text: str, max_preview_rows: int = 20, save_to_file: bool = True
                             ) -> Tuple[str, Optional[str], Optional[List[Dict[str, Any]]], Optional[int]]:
    """执行SQL并返回结果"""
    # print(f"执行 SQL ID={sql_id}")
    
    tmp_dir = Path("./tmp_new")
    tmp_dir.mkdir(exist_ok=True)
    
    tmp_in = tmp_dir / f"generated_one_sql_{sql_id}.json"
    tmp_out = tmp_dir / f"generated_one_sql_res_{sql_id}.json"

    def clean_sql_string(s):
        s = s.strip()
        s = re.sub(r'^```[sS][qQ][lL]\s*\n?', '', s)
        s = re.sub(r'^```\s*\n?', '', s)
        s = re.sub(r'\n?```\s*$', '', s)
        return s.strip()
    
    sql_text = clean_sql_string(sql_text)
    if not sql_text.endswith(';'):
        sql_text += ';'
    
    payload = [{"sql_id": sql_id, "sql": sql_text}]

    with open(tmp_in, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    db_config = {
        "host": os.getenv("DB_HOST", "127.0.0.1"),
        "user": os.getenv("DB_USER", "root"),
        "password": os.getenv("DB_PASSWORD", ""),
        "db": os.getenv("DB_NAME", "TGAC"),
        "port": int(os.getenv("DB_PORT", "9030")),
    }

    executor = execute_sql_with_pymysql()
    executor.execute_sql_with_pymysql(str(tmp_in), str(tmp_out), db_config=db_config)

    if not tmp_out.exists():
        return "error", "执行器未产生结果文件", None, None

    try:
        content = None
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
        
        for encoding in encodings:
            try:
                content = tmp_out.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue

        results = json.loads(content)
    except Exception as e:
        return "error", f"结果文件解析失败: {e}", None, None
    finally:
        # 如果是采样/临时执行（save_to_file=False），执行后删除临时输入/输出文件，避免保留大量文件。
        # 如果需要保留文件（save_to_file=True），则不删除。
        if not save_to_file:
            try:
                tmp_in.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                tmp_out.unlink(missing_ok=True)
            except Exception:
                pass

    if not isinstance(results, list) or not results:
        return {"status": "error", "error_message": "结果文件为空或格式不正确", "result_fingerprint": "", "row_count": 0, "result_data": []}

    rec = results[0]
    status = rec.get("status")
    result_meta = {
        "status": "error",
        "error_message": "",
        "result_data": [],
        "result_fingerprint": "", 
        "row_count": 0
    }

    if status == "success":
        rows = rec.get("result")
        if not isinstance(rows, list):
            result_meta["error_message"] = "结果格式不正确:result 非列表"
            return result_meta
            
        result_meta["result_data"] = rows
        result_meta["row_count"] = len(rows)
        
        # 计算指纹
        fingerprint_str = extract_and_normalize_result(rows)
        result_meta["result_fingerprint"] = hashlib.md5(fingerprint_str.encode('utf-8')).hexdigest()
        
        if len(rows) == 0:
            result_meta["status"] = "success_empty"
        else:
            result_meta["status"] = "success_non_empty"
            # 仅保留预览数据以节省内存，但在指纹计算后
            # result_meta["result_data"] = rows[:max_preview_rows] 
        
        return result_meta
    else:
        result_meta["error_message"] = rec.get("error_message") or "未知执行错误"
        return result_meta

def gen_special_prompt(
    question: str,
    knowledge: str,
    table_list: List[str],
    vector_db: SQLVectorDatabase,
    top_k: int = 1
) -> str:
    """
    从向量数据库检索最相似的 schema 信息
    
    Args:
        question: 问题描述
        knowledge: 知识内容
        table_list: 表列表
        vector_db: 向量数据库实例
        top_k: 返回前k个结果
        
    Returns:
        组合后的 prompt
    """
    try:
        logging.info(f"🔍 开始向量检索: tables={table_list}, top_k={top_k}")
        # 使用向量数据库检索
        results = vector_db.search(
            question=question,
            knowledge=knowledge,
            table_list=table_list,
            top_k=top_k
        )
        logging.info(f"✅ 向量检索完成: 找到 {len(results)} 条相关示例")
        if not results:
            logging.warning("⚠️ 向量检索无结果")
            return ""
        
        # 组合所有检索结果的信息
        prompt_parts = []
        for idx, result in enumerate(results, 1):
            parts = []
            
            # 添加各个字段
            if result.get('schema'):
                parts.append(f"### Tips:\n{result['schema']}")
            
            if result.get('key_fields'):
                parts.append(f"### 关键字段:\n{json.dumps(result['key_fields'], ensure_ascii=False, indent=2)}")
            
            if result.get('logic_pattern'):
                parts.append(f"### 逻辑:\n{json.dumps(result['logic_pattern'], ensure_ascii=False, indent=2)}")
            
            if result.get('nl_to_schema_map'):
                parts.append(f"### 映射:\n{json.dumps(result['nl_to_schema_map'], ensure_ascii=False, indent=2)}")
            
            if result.get('common_pitfalls'):
                parts.append(f"### 注意事项:\n{json.dumps(result['common_pitfalls'], ensure_ascii=False, indent=2)}")
            
            prompt_parts.append("\n\n".join(parts))
        
        return "\n\n" + "\n\n".join(prompt_parts) + "\n\n"
        
    except Exception as e:
        print(f"⚠️ 向量检索失败: {e}")
        return ""

def rewrite_question(
    item: Dict,
    schema_index: Dict[str, Dict],
    require_confirmation: bool = False,
    prompt_log_dir: Path = None,
    max_retries: int = 3
) -> str:
    """
    重述问题，使其更清晰、完整、无歧义
    
    Args:
        item: 包含question、knowledge、table_list的数据项
        schema_index: 表结构索引
        require_confirmation: 是否需要用户确认
        prompt_log_dir: prompt日志保存目录
        max_retries: 最大重试次数（用户可以补充信息后重新生成）
        
    Returns:
        重述后的问题
    """
    sql_id = item.get("sql_id", "unknown")
    original_question = item.get("question", "")
    knowledge = item.get("knowledge", "")
    table_list = item.get("table_list", [])
    
    logging.info(f"\n{'='*80}")
    logging.info(f"🔄 [{sql_id}] 开始问题重述")
    logging.info(f"   原始问题: {original_question}")
    
    # 构建表结构摘要（只构建一次）
    schema_summary = []
    for table_name in table_list:
        table_info = schema_index.get(table_name, {})
        table_desc = table_info.get("description", "")
        columns = table_info.get("columns", set())
        col_details_map = table_info.get("column_details", {})
        
        # 构建详细的列信息
        col_lines = []
        for col in sorted(columns):
            col_info = col_details_map.get(col, {})
            col_type = col_info.get("type", "")
            col_desc = col_info.get("description", "")
            top_values = col_info.get("top_values", [])
            
            col_line = f"    {col}({col_type}): {col_desc}"
            if top_values:
                col_line += f" [常见值: {', '.join(str(v) for v in top_values[:3])}]"
            col_lines.append(col_line)
        
        table_line = f"表 {table_name}: {table_desc}\n  字段:\n" + "\n".join(col_lines)
        schema_summary.append(table_line)
    
    schema_text = "\n".join(schema_summary)
    
    # 迭代式重述，允许用户补充信息
    user_feedback = ""  # 用户补充的上下文
    attempt = 0
    
    while attempt < max_retries:
        attempt += 1
        logging.info(f"   🔄 第 {attempt}/{max_retries} 次重述")
        
        # 构建重述prompt
        rewrite_prompt = f"""{QUESTION_REWRITE_PROMPT}

## 原始问题
{original_question}

## 业务知识
{knowledge}

## 相关表结构
{schema_text}"""

        if user_feedback:
            rewrite_prompt += f"""

## 用户补充说明
{user_feedback}

请结合用户的补充说明，重新理解并重述这个问题："""
        else:
            rewrite_prompt += "\n\n请基于以上信息，重述这个问题："
        
        # 保存完整的prompt日志（包括system message）
        if prompt_log_dir:
            full_prompt = f"=== SYSTEM MESSAGE ===\n你是一个专业的数据分析需求理解专家。\n\n=== USER MESSAGE ===\n{rewrite_prompt}"
            log_prompt(prompt_log_dir, sql_id, f"rewrite_attempt_{attempt}", full_prompt, attempt=attempt)
        
        try:
            client, model = _get_llm_client_and_model()
            logging.info(f"   调用模型进行问题重述...")
            
            system_msg = "你是一个专业的数据分析需求理解专家。"
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": rewrite_prompt}
                ],
                stream=False,
                temperature=0.1
            )
            
            rewritten_question = response.choices[0].message.content.strip()
            
            # 清理可能的前缀
            for prefix in ["重述：", "问题：", "重述后的问题：", "Question:", "Rewritten:"]:
                if rewritten_question.startswith(prefix):
                    rewritten_question = rewritten_question[len(prefix):].strip()
            
            logging.info(f"   ✅ 重述完成")
            logging.info(f"   重述后问题: {rewritten_question}")
            
            # 保存重述结果日志（模型响应）
            if prompt_log_dir:
                log_prompt(prompt_log_dir, sql_id, f"rewrite_output_{attempt}", rewritten_question, attempt=attempt)
            
            # 用户确认
            if require_confirmation:
                print(f"\n{'='*80}")
                print(f"📋 问题重述 [{sql_id}] - 第 {attempt}/{max_retries} 次")
                print(f"{'='*80}")
                print(f"\n原始问题:\n  {original_question}")
                if user_feedback:
                    print(f"\n补充说明:\n  {user_feedback}")
                print(f"\n重述后问题:\n  {rewritten_question}")
                print(f"\n{'='*80}")
                
                while True:
                    if attempt < max_retries:
                        user_input = input("\n请选择: y(接受) / n(补充信息并重新生成) / original(使用原问题): ").strip().lower()
                    else:
                        # 达到重试上限
                        user_input = input("\n请选择: y(接受) / original(使用原问题) [已达重试上限]: ").strip().lower()
                    
                    if user_input == 'y':
                        logging.info(f"   ✅ 用户确认接受重述")
                        return rewritten_question
                    elif user_input in ['n', 'no'] and attempt < max_retries:
                        print(f"\n请输入补充说明（帮助AI更好理解问题，输入空行结束）:")
                        feedback_lines = []
                        while True:
                            line = input()
                            if line == "":
                                break
                            feedback_lines.append(line)
                        if feedback_lines:
                            user_feedback = " ".join(feedback_lines)
                            logging.info(f"   📝 用户补充信息: {user_feedback}")
                            break  # 跳出内层循环，重新生成
                        else:
                            print("未输入补充信息，请重新选择")
                    elif user_input in ['original', 'o']:
                        logging.info(f"   ⚙️ 用户选择使用原问题")
                        return original_question
                    else:
                        if attempt < max_retries:
                            print("请输入 y(接受) / n(补充信息) / original(使用原问题)")
                        else:
                            print("请输入 y(接受) / original(使用原问题)")
            else:
                # 不需要确认，直接返回
                return rewritten_question
                
        except Exception as e:
            logging.error(f"   ❌ 问题重述失败: {e}")
            if attempt >= max_retries:
                logging.info(f"   ⚠️ 达到重试上限，回退到原始问题")
                return original_question
            else:
                logging.info(f"   ⚠️ 重试中...")
                continue
    
    # 达到最大重试次数，使用原问题
    logging.info(f"   ⚠️ 达到最大重试次数 ({max_retries})，使用原始问题")
    return original_question

def build_base_context(
    item: Dict,
    schema_index: Dict[str, Dict],
    few_shot_examples: Optional[List[Dict]] = None,
    vector_db: Optional[SQLVectorDatabase] = None
) -> str:
    """构建基础上下文(只构建一次,后续复用)"""
    sql_id = item.get("sql_id", "unknown")
    question = item.get("question", "") or ""
    table_list = item.get("table_list", []) or []
    knowledge = item.get("knowledge", "") or ""

    logging.info(f"📦 [{sql_id}] 开始构建base context")
    logging.info(f"   Question: {question[:100]}...")
    logging.info(f"   Tables: {table_list}")
    
    schema_snippets = _build_schema_snippets_for_tables(table_list, schema_index)
    logging.info(f"   Schema snippets: {len(schema_snippets)} 个表")
    
    special_prompt = gen_special_prompt(question=question,
                                        knowledge=knowledge,
                                        table_list=table_list,
                                        vector_db=vector_db,
                                        top_k=1)
    few_shot_prompt = ""
    if few_shot_examples:
        few_shot_prompt = "## Examples:\n" + build_few_shot_examples(few_shot_examples) + "\n\n"
        logging.info(f"   Few-shot examples: {len(few_shot_examples)} 个")

    base_context = (
        SYSTEM_PROMPT + "\n\n" +
        few_shot_prompt +
        "## Schema:\n" + "\n\n".join(schema_snippets) + "\n\n" +
        "## Knowledge:\n" + knowledge + "\n\n" +
        special_prompt + "\n\n" +
        "## Question:\n" + question + "\n\n"
    )
    
    logging.info(f"✅ [{sql_id}] Base context构建完成, 总长度: {len(base_context)} 字符")
    return base_context

def _attempt_generate_and_execute_sql(
    item: Dict,
    schema_index: Dict[str, Dict],
    few_shot_examples: Optional[List[Dict]] = None,
    vector_db: Optional[SQLVectorDatabase] = None,
    prompt_log_dir: Path = None
) -> Dict:
    """尝试生成并执行SQL,支持重试"""
    sql_id = item.get("sql_id", "unknown")
    logging.info(f"\n{'='*80}")
    logging.info(f"🚀 [{sql_id}] 开始SQL生成流程")
    logging.info(f"{'='*80}")
    
    base_context = build_base_context(item, schema_index, few_shot_examples, vector_db)
    
    # 记录base context
    if prompt_log_dir:
        log_prompt(prompt_log_dir, sql_id, "base_context", base_context, 0)
    
    qwen_client, qwen_model = _get_llm_client_and_model()
    
    max_attempts = 4
    actual_attempt = 0
    execution_history = ""  # 记录执行历史
    
    while actual_attempt < max_attempts:
        try:
            logging.info(f"\n🔄 [{sql_id}] 第 {actual_attempt + 1}/{max_attempts} 次尝试")
            
            # 构建完整prompt(复用base_context + 执行历史)
            if actual_attempt == 0:
                # 第一次:生成诊断SQL
                current_prompt = base_context + DIAGNOSTIC_PROMPT
                stage = "diagnostic"
            else:
                # 重试:附加执行历史
                current_prompt = base_context + execution_history + "\n\n请根据上述执行结果重新生成SQL:\n"
                stage = "final_sql"
            
            # 记录完整的prompt（包括system message）
            if prompt_log_dir:
                system_msg = "你是一个专业的 StarRocks SQL 生成专家。"
                full_prompt = f"=== SYSTEM MESSAGE ===\n{system_msg}\n\n=== USER MESSAGE ===\n{current_prompt}"
                log_prompt(prompt_log_dir, sql_id, stage, full_prompt, actual_attempt)
            
            logging.info(f"   Prompt长度: {len(current_prompt)} 字符")
            logging.info(f"   调用模型: {qwen_model}")
            
            # 调用 Qwen
            time.sleep(2)  # 减少等待时间
            system_msg = "你是一个专业的 StarRocks SQL 生成专家。"
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": current_prompt},
            ]
            qwen_response = qwen_client.chat.completions.create(
                model=qwen_model,
                messages=messages,
                stream=False,
                temperature=0.1
            )
            generated_text = qwen_response.choices[0].message.content.strip()
            logging.info(f"   ✅ 模型响应长度: {len(generated_text)} 字符")
            
            # 记录LLM的响应
            if prompt_log_dir:
                log_prompt(prompt_log_dir, sql_id, f"{stage}_response", generated_text, actual_attempt)
            
            # 判断是诊断SQL还是最终SQL
            if actual_attempt == 0 and 'select' in generated_text.lower():
                logging.info(f"   📊 [{sql_id}] 开始执行诊断SQL")
                # 执行诊断SQL
                sql_lines = [line.strip() for line in generated_text.split('\n') 
                           if line.strip() and 'select' in line.lower()]
                
                logging.info(f"   发现 {len(sql_lines)} 条诊断SQL")
                diagnostic_results = []
                for idx, sql in enumerate(sql_lines[:5], 1):  # 最多执行5条
                    logging.info(f"   执行诊断SQL {idx}: {sql[:80]}...")
                    exec_res = run_sql_and_get_feedback(f"diag_{uuid.uuid4().hex[:6]}", sql)
                    status = exec_res["status"]
                    if status == "success_non_empty":
                        logging.info(f"   ✅ 诊断SQL {idx} 成功: {exec_res['row_count']} 行")
                        sample_data = json.dumps(exec_res["result_data"][:5], ensure_ascii=False, indent=2)
                        diagnostic_results.append(f"Query: {sql}\nResult:\n{sample_data}")
                    elif status == "success_empty":
                        logging.info(f"   ⚠️ 诊断SQL {idx} 无数据")
                        diagnostic_results.append(f"Query: {sql}\nResult: Empty")
                    else:
                        logging.warning(f"   ❌ 诊断SQL {idx} 失败: {exec_res['error_message']}")
                        diagnostic_results.append(f"Query: {sql}\nError: {exec_res['error_message']}")
                
                # 更新执行历史
                execution_history += "\n\n## Diagnostic Results:\n" + "\n\n".join(diagnostic_results)
                execution_history += "\n\n" + FINAL_SQL_PROMPT
                
                # 记录诊断结果和接下来要用的完整prompt
                if prompt_log_dir:
                    # 记录诊断结果
                    log_prompt(prompt_log_dir, sql_id, "diagnostic_results", "\n\n".join(diagnostic_results), actual_attempt)
                    # 记录下一轮的完整prompt（base + diagnostic results + final prompt）
                    next_full_prompt = base_context + execution_history
                    system_msg = "你是一个专业的 StarRocks SQL 生成专家。"
                    next_prompt_with_system = f"=== SYSTEM MESSAGE ===\n{system_msg}\n\n=== USER MESSAGE ===\n{next_full_prompt}"
                    log_prompt(prompt_log_dir, sql_id, "final_sql_full_prompt", next_prompt_with_system, actual_attempt)
                
                logging.info(f"   ✅ 诊断阶段完成, 执行历史长度: {len(execution_history)} 字符")
                actual_attempt += 1
                continue
            
            # 执行最终SQL
            logging.info(f"   🔧 [{sql_id}] 执行最终SQL")
            logging.info(f"   SQL预览: {generated_text[:200]}...")
            
            # 记录最终生成的SQL
            if prompt_log_dir:
                log_prompt(prompt_log_dir, sql_id, "final_sql", generated_text, actual_attempt)
            
            exec_result = run_sql_and_get_feedback(sql_id, generated_text)
            
            # 记录执行结果
            if prompt_log_dir:
                execution_result_log = f"Status: {exec_result['status']}\n"
                execution_result_log += f"Row Count: {exec_result['row_count']}\n"
                execution_result_log += f"Fingerprint: {exec_result['result_fingerprint']}\n"
                if exec_result['error_message']:
                    execution_result_log += f"Error: {exec_result['error_message']}\n"
                if exec_result['row_count'] > 0:
                    execution_result_log += f"\nSample Data (first 3 rows):\n{json.dumps(exec_result['result_data'][:3], ensure_ascii=False, indent=2)}"
                log_prompt(prompt_log_dir, sql_id, "execution_result", execution_result_log, actual_attempt)
            
            # 包装结果
            result_package = {
                "sql": generated_text,
                "status": exec_result["status"],
                "error": exec_result["error_message"],
                "fingerprint": exec_result["result_fingerprint"],
                "row_count": exec_result["row_count"],
                "data": exec_result["result_data"]
            }

            if exec_result["status"] == "success_non_empty":
                logging.info(f"   ✅ [{sql_id}] SQL执行成功! 返回 {exec_result['row_count']} 行")
                logging.info(f"   指纹: {exec_result['result_fingerprint']}")
                return result_package
            elif exec_result["status"] == "success_empty":
                logging.warning(f"   ⚠️ [{sql_id}] SQL执行成功但无数据")
                execution_history += f"\n\n## Previous Attempt:\nSQL: {generated_text}\nResult: Empty, 请减少限制,过滤条件或者进行其他操作."
                
                # 记录重试prompt
                if prompt_log_dir:
                    log_prompt(prompt_log_dir, sql_id, "retry_empty", execution_history, actual_attempt)
                
                actual_attempt += 1
            else:
                logging.error(f"   ❌ [{sql_id}] SQL执行失败: {exec_result['error_message']}")
                execution_history += f"\n\n## Previous Attempt:\nSQL: {generated_text}\nError: {exec_result['error_message']}\nPlease fix the error."
                
                # 记录重试prompt
                if prompt_log_dir:
                    log_prompt(prompt_log_dir, sql_id, "retry_error", execution_history, actual_attempt)
                
                actual_attempt += 1
                
        except Exception as e:
            error_msg = str(e).lower()
            # 检查是否为网络相关错误
            network_errors = ['ssl', 'eof', 'connection', 'timeout', 'network', 'socket', 'overloaded','disconnected','exhausted','timed out']
            is_network_error = any(err_keyword in error_msg for err_keyword in network_errors)
            
            if is_network_error:
                logging.warning(f"   ⚠️ [{sql_id}] 网络异常(不消耗预算): {e}")
                logging.info(f"   😴 等待10秒后重试...")
                time.sleep(10)
                continue  # 不增加 actual_attempt,直接重试
            else:
                logging.error(f"   ❌ [{sql_id}] 其他异常: {e}")
                return {
                    "sql": f"-- 生成异常: {e}",
                    "status": "failed",
                    "error": str(e),
                    "fingerprint": "error",
                    "row_count": 0,
                    "data": []
                }
    
    logging.error(f"   ❌ [{sql_id}] 达到最大重试次数")
    return {
        "sql": generated_text if 'generated_text' in locals() else "-- Failed",
        "status": "failed",
        "error": "Max retries reached",
        "fingerprint": "error",
        "row_count": 0,
        "data": []
    }

def generate_sql_for_item(
    item: Dict,
    schema_index: Dict[str, Dict],
    golden_examples: Optional[List[Dict]] = None,
    vector_db: Optional[SQLVectorDatabase] = None,
    prompt_log_dir: Path = None
    ) -> Dict:
    """主入口:生成SQL"""
    few_shot_examples = []
    if golden_examples:
        question = item.get("question", "")
        few_shot_examples = get_top_k_similar_examples(question, golden_examples, k=1)
    
    return _attempt_generate_and_execute_sql(
        item=item,
        schema_index=schema_index,
        few_shot_examples=few_shot_examples,
        vector_db=vector_db,
        prompt_log_dir=prompt_log_dir
    )

def vote_for_best_result(results: List[Dict]) -> Dict:
    """
    输入：同一题目的多次运行结果列表
    输出：选出的最佳结果
    逻辑：
    1. 优先选 status='success_non_empty' 的
    2. 在 success 的结果中，按 fingerprint (执行结果) 进行投票
    3. 选出现次数最多的那个 fingerprint 对应的任意一个结果
    """
    # 优先非空成功结果
    success_results = [r for r in results if r['status'] == 'success_non_empty']
    if not success_results:
        # 其次选空成功结果
        success_results = [r for r in results if r['status'] == 'success_empty']
    
    if not success_results:
        # 如果都失败了，返回最后一个结果
        print(f"    ❌ 全部失败 ({len(results)} runs)")
        return results[-1]
    
    # 统计指纹出现次数
    fingerprints = [r['fingerprint'] for r in success_results]
    counts = Counter(fingerprints)
    
    # 找到票数最多的指纹
    most_common_fingerprint = counts.most_common(1)[0][0]
    vote_count = counts.most_common(1)[0][1]
    
    print(f"    投票结果: {len(success_results)} 次有效成功, 选中结果出现 {vote_count} 次")
    
    # 返回对应的第一个结果
    for r in success_results:
        if r['fingerprint'] == most_common_fingerprint:
            return r
            
    return success_results[0]

def remove_duplicate_keys(input_file, output_file):
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 递归去重（支持嵌套字典）
    def clean_sql_string(s):
        s = s.strip()
        s = re.sub(r'^```[sS][qQ][lL]\s*\n?', '', s)
        s = re.sub(r'^```\s*\n?', '', s)
        s = re.sub(r'\n?```\s*$', '', s)
        return s.strip()
    
    def deduplicate_dict(d):
        if isinstance(d, dict):
            has_sql = 'sql' in d
            new_d = {}
            for key, value in d.items():
                if key == 'generated_sql':
                    if has_sql:
                        continue
                    else:
                        key = 'sql'
                processed = deduplicate_dict(value)
                if key == 'sql' and isinstance(processed, str):
                    processed = clean_sql_string(processed)
                new_d[key] = processed
            return new_d
        elif isinstance(d, list):
            return [deduplicate_dict(item) for item in d]
        else:
            return d
    
    # 执行去重
    deduplicated_data = deduplicate_dict(data)
    
    # 写入新文件（indent=4保持格式化）
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(deduplicated_data, f, ensure_ascii=False, indent=4)
    
    print(f"去重完成！结果已保存到 {output_file}")

def main():
    # 初始化日志系统
    log_file, prompt_log_dir = setup_logging()
    logging.info("="*80)
    logging.info("🚀 SQL生成系统启动")
    logging.info(f"📝 日志文件: {log_file}")
    logging.info(f"📁 Prompt日志目录: {prompt_log_dir}")
    logging.info("="*80)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"数据集不存在: {DATASET_PATH}")
    if not os.path.exists(SCHEMA_PATH):
        raise FileNotFoundError(f"Schema 文件不存在: {SCHEMA_PATH}")
    
    logging.info("📚 开始加载向量数据库...")
    vector_db = SQLVectorDatabase()
    vector_db.load_database("./vectordatabase.pkl")
    logging.info("✅ 向量数据库加载成功")
    logging.info(f"📖 加载数据集: {DATASET_PATH}")
    dataset = load_json(DATASET_PATH)
    logging.info(f"   数据集大小: {len(dataset)} 条")
    
    logging.info(f"📖 加载Schema: {SCHEMA_PATH}")
    schema = load_json(SCHEMA_PATH)
    schema_index = build_schema_index(schema)
    logging.info(f"   Schema包含 {len(schema_index)} 个表")
    
    logging.info("🏆 提取Golden样本...")
    golden_examples = extract_golden_examples(DATASET_PATH)
    logging.info(f"   Golden样本数量: {len(golden_examples)} 个")

    # 准备容器存放所有轮次的结果
    # 结构: { "sql_id_1": [result_run1, result_run2, ...], ... }
    all_runs_collection = {} 
    NUM_RUNS = 1  # 定义运行轮数
    
    total_start_time = time.time()
    print(f"🚀 开始生成SQL (共 {NUM_RUNS} 轮)...")

    def _process_wrapper(item: Dict) -> Tuple[str, Dict]:
        """包装函数,用于并发执行"""
        try:
            res = generate_sql_for_item(item, schema_index, golden_examples=golden_examples, vector_db=vector_db, prompt_log_dir=prompt_log_dir)
            # 调试记录
            with debug_file_lock:
                with open("debug_last_generated_sql.sql", "a", encoding="utf-8") as f:
                    f.write(f"-- sql_id: {item.get('sql_id')}\n")
                    f.write(res.get("sql", "-- no sql"))
                    f.write("\n\n")
            return item.get('sql_id'), res
        except Exception as e:
            logging.error(f"❌ [{item.get('sql_id')}] 任务失败: {e}")
            return item.get('sql_id'), {"status": "error", "sql": f"-- 生成失败: {str(e)}", "fingerprint": "error"}

    # 筛选需要处理的数据
    items_to_process = []
    for idx, item in enumerate(dataset):
        sql_id = item.get('sql_id', '')
        # match = re.search(r'_(\d+)$', sql_id)
        # 这里控制筛选逻辑，例如只跑前10个或者特定ID
        # target_ids = ['sql_4','sql_26','sql_51','sql_68','sql_73','sql_78','sql_82','sql_119']
        # if sql_id in target_ids:
        items_to_process.append(item)
    
    logging.info(f"\n📋 共需处理 {len(items_to_process)} 个问题")
    if ENABLE_QUESTION_REWRITE:
        logging.info(f"   问题重述: 启用 (模式: {'需要用户确认' if REQUIRE_USER_CONFIRMATION else '自动重述'})")
    
    # 多轮执行（每个问题先重述再生成SQL）
    for run_i in range(NUM_RUNS):
        print(f"\n🚀 开始第 {run_i + 1} / {NUM_RUNS} 轮生成...")
        
        for idx, item in enumerate(items_to_process):
            sql_id = item.get('sql_id', '')
            logging.info(f"\n{'='*80}")
            logging.info(f"📝 [{idx+1}/{len(items_to_process)}] 处理问题: {sql_id}")
            logging.info(f"{'='*80}")
            
            # 创建副本
            item_copy = dict(item)
            
            # 1. 问题重述阶段（如果启用）
            if ENABLE_QUESTION_REWRITE:
                rewritten_question = rewrite_question(
                    item=item_copy,
                    schema_index=schema_index,
                    require_confirmation=REQUIRE_USER_CONFIRMATION,
                    prompt_log_dir=prompt_log_dir
                )
                # 保留原问题并替换为重述后的问题
                item_copy["original_question"] = item_copy.get("question", "")
                item_copy["question"] = rewritten_question
                logging.info(f"   ✅ 重述完成")
            
            # 2. SQL生成阶段
            try:
                result = generate_sql_for_item(
                    item_copy, 
                    schema_index, 
                    golden_examples=golden_examples, 
                    vector_db=vector_db, 
                    prompt_log_dir=prompt_log_dir
                )
                
                # 保存结果
                if sql_id not in all_runs_collection:
                    all_runs_collection[sql_id] = []
                all_runs_collection[sql_id].append(result)
                
                # 调试记录
                with debug_file_lock:
                    with open("debug_last_generated_sql.sql", "a", encoding="utf-8") as f:
                        f.write(f"-- sql_id: {sql_id}\n")
                        f.write(result.get("sql", "-- no sql"))
                        f.write("\n\n")
                
                logging.info(f"   ✅ SQL生成完成, 状态: {result.get('status', 'unknown')}")
            except Exception as e:
                logging.error(f"   ❌ SQL生成失败: {e}")
                error_result = {"status": "error", "sql": f"-- 生成失败: {str(e)}", "fingerprint": "error"}
                if sql_id not in all_runs_collection:
                    all_runs_collection[sql_id] = []
                all_runs_collection[sql_id].append(error_result)

    # 投票决策并输出
    logging.info("\n" + "="*80)
    logging.info("🗳️ 开始投票决策阶段")
    logging.info("="*80)
    output_items = []
    
    for item in items_to_process:
        sql_id = item['sql_id']
        candidates = all_runs_collection.get(sql_id, [])
        
        out = dict(item)
        if not candidates:
            logging.warning(f"⚠️ [{sql_id}] 无候选结果")
            generated_sql = "-- No result"
        else:
            logging.info(f"\n📊 [{sql_id}] 开始投票 (候选数: {len(candidates)})")
            best_res = vote_for_best_result(candidates)
            generated_sql = best_res['sql']
            logging.info(f"✅ [{sql_id}] 投票完成, 状态: {best_res['status']}")
        
        # 如果使用了问题重述，保留重述信息
        if ENABLE_QUESTION_REWRITE and "original_question" in out:
            # 将重述后的问题作为主问题，原问题保存为original_question
            pass  # 已经在item中更新
        
        out["generated_sql"] = generated_sql
        output_items.append(out)

    logging.info(f"\n💾 写入结果文件: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_items, f, ensure_ascii=False, indent=2)
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    logging.info("\n" + "="*80)
    logging.info("✅ 任务完成!")
    logging.info(f"📄 结果文件: {OUTPUT_PATH}")
    logging.info(f"⏱️ 总耗时: {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟)")
    logging.info(f"📊 处理数量: {len(output_items)} 条")
    logging.info(f"📝 日志文件: {log_file}")
    logging.info(f"📁 Prompt日志: {prompt_log_dir}")
    logging.info("="*80)

if __name__ == "__main__":
    main()
    input_path = r".\data\final_sql.json"
    output_path = r".\data\best_sql.json"
    remove_duplicate_keys(input_path, output_path)