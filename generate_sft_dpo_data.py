#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 gold 正样本自动生成 SFT 与 DPO 训练数据

功能：
1. 生成 repair 负样本（必报错的 SQL）
2. 生成 DPO 候选（采样多个 SQL 并执行打分）
3. 输出格式化的训练数据（JSONL）

依赖：复用 final_code.py 的现成函数
"""

import os
import json
import re
import argparse
import hashlib
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict, Counter
from dataclasses import dataclass
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# 从 final_code.py 导入必需的函数和常量
from final_code import (
    load_json,
    build_schema_index,
    build_base_context,
    run_sql_and_get_feedback,
    _get_llm_client_and_model,
    SYSTEM_PROMPT,
    FINAL_SQL_PROMPT
)

# SQL 关键字列表（用于过滤列名扰动）
SQL_KEYWORDS = {
    'SELECT', 'FROM', 'WHERE', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 
    'ON', 'GROUP', 'BY', 'HAVING', 'ORDER', 'LIMIT', 'OFFSET', 'UNION', 
    'AS', 'AND', 'OR', 'NOT', 'IN', 'EXISTS', 'CASE', 'WHEN', 'THEN', 
    'ELSE', 'END', 'WITH', 'DISTINCT', 'COUNT', 'SUM', 'AVG', 'MAX', 'MIN',
    'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP', 'TRUNCATE',
    'DATE', 'CAST', 'COALESCE', 'NULLIF', 'SUBSTR', 'CONCAT'
}


class SQLCache:
    """SQL 执行结果缓存（线程安全）"""
    def __init__(self):
        self.cache = {}
        self.lock = threading.Lock()
    
    def get_key(self, sql_text: str) -> str:
        """生成 SQL 的唯一标识"""
        return hashlib.md5(sql_text.strip().encode('utf-8')).hexdigest()
    
    def get(self, sql_text: str) -> Optional[Dict]:
        """获取缓存的执行结果"""
        key = self.get_key(sql_text)
        with self.lock:
            return self.cache.get(key)
    
    def set(self, sql_text: str, result: Dict):
        """缓存执行结果"""
        key = self.get_key(sql_text)
        with self.lock:
            self.cache[key] = result


class RepairSampleGenerator:
    """生成必报错的 repair 负样本"""
    
    def __init__(self, sql_cache: SQLCache):
        self.sql_cache = sql_cache
        self.stats = {
            'attempts': 0,
            'success': 0,
            'by_type': defaultdict(int)
        }
    
    def generate_bad_sql(self, gold_sql: str, table_list: List[str], method: str) -> Optional[str]:
        """
        生成必报错的 SQL
        
        Args:
            gold_sql: 正确的 SQL
            table_list: 表名列表
            method: 'column' | 'table' | 'dialect'
        
        Returns:
            bad_sql 或 None
        """
        if method == 'column':
            return self._perturb_column(gold_sql)
        elif method == 'table':
            return self._perturb_table(gold_sql, table_list)
        elif method == 'dialect':
            return self._inject_dialect_error(gold_sql)
        return None
    
    def _perturb_column(self, gold_sql: str) -> Optional[str]:
        """列名扰动：Unknown column"""
        # 提取疑似列名（简单正则）
        pattern = r'\b([a-z_][a-z0-9_]{2,})\b'
        matches = re.findall(pattern, gold_sql, re.IGNORECASE)
        
        # 过滤 SQL 关键字
        candidates = [m for m in matches if m.upper() not in SQL_KEYWORDS]
        if not candidates:
            return None
        
        # 随机选一个列名
        col = random.choice(candidates)
        
        # 扰动方式：添加后缀或删除字符
        if random.random() < 0.5:
            bad_col = col + "_x"
        else:
            bad_col = col[:-1] if len(col) > 3 else col + "z"
        
        # 替换（只替换第一次出现）
        bad_sql = gold_sql.replace(col, bad_col, 1)
        return bad_sql if bad_sql != gold_sql else None
    
    def _perturb_table(self, gold_sql: str, table_list: List[str]) -> Optional[str]:
        """表名扰动：Table not found"""
        if not table_list:
            return None
        
        # 随机选一个表
        table = random.choice(table_list)
        bad_table = f"not_exist_{table}"
        
        # 替换表名（使用词边界）
        bad_sql = re.sub(
            rf'\b{re.escape(table)}\b',
            bad_table,
            gold_sql,
            count=1
        )
        return bad_sql if bad_sql != gold_sql else None
    
    def _inject_dialect_error(self, gold_sql: str) -> Optional[str]:
        """方言注入：Syntax error"""
        # 在 FROM 后插入不兼容语法
        if 'FROM' in gold_sql.upper():
            # 插入 Hive 风格的 LATERAL VIEW
            match = re.search(r'\bFROM\s+(\w+)', gold_sql, re.IGNORECASE)
            if match:
                table_name = match.group(1)
                injection = f"FROM {table_name} LATERAL VIEW explode(array(1,2)) t AS x"
                bad_sql = gold_sql.replace(match.group(0), injection, 1)
                return bad_sql
        
        # 或在 SELECT 中插入不兼容函数
        if 'SELECT' in gold_sql.upper():
            match = re.search(r'SELECT\s+(.*?)\s+FROM', gold_sql, re.IGNORECASE | re.DOTALL)
            if match:
                select_part = match.group(1)
                bad_select = f"SELECT to_date(dtstatdate), {select_part} FROM"
                bad_sql = gold_sql.replace(match.group(0), bad_select, 1)
                return bad_sql
        
        return None
    
    def try_generate_repair_sample(
        self, 
        gold_sql: str, 
        table_list: List[str],
        sql_id: str,
        max_attempts: int = 5
    ) -> Optional[Tuple[str, str, str]]:
        """
        尝试生成一个 repair 样本
        
        Returns:
            (bad_sql, error_message, bad_type) 或 None
        """
        methods = ['column', 'table', 'dialect']
        
        for _ in range(max_attempts):
            self.stats['attempts'] += 1
            method = random.choice(methods)
            
            bad_sql = self.generate_bad_sql(gold_sql, table_list, method)
            if not bad_sql:
                continue
            
            # 检查缓存
            cached = self.sql_cache.get(bad_sql)
            if cached:
                result = cached
            else:
                # 执行 bad_sql
                result = run_sql_and_get_feedback(
                    sql_id=f"{sql_id}_repair_attempt",
                    sql_text=bad_sql,
                    save_to_file=False  # 不保存临时文件
                )
                self.sql_cache.set(bad_sql, result)
            
            # 检查是否报错
            if result.get('status') == 'error' and result.get('error_message'):
                self.stats['success'] += 1
                self.stats['by_type'][method] += 1
                
                bad_type_map = {
                    'column': 'unknown_column',
                    'table': 'table_not_found',
                    'dialect': 'dialect_error'
                }
                return bad_sql, result['error_message'], bad_type_map[method]
        
        return None


class DPOCandidateGenerator:
    """生成 DPO 候选并打分"""
    
    def __init__(self, sql_cache: SQLCache, client, model_name: str, temperature: float):
        self.sql_cache = sql_cache
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.stats = {
            'total_samples': 0,
            'total_candidates': 0,
            'pairs_generated': 0,
            'skip_reasons': defaultdict(int)
        }
    
    def sample_sql_candidates(
        self, 
        base_context: str, 
        k: int,
        sql_id: str,
        max_workers: int = 4
    ) -> List[str]:
        """
        并行采样 K 个候选 SQL
        
        Args:
            base_context: 基础上下文
            k: 采样数量
            sql_id: SQL ID
            max_workers: 最大并行线程数
        
        Returns:
            List[sql_text]
        """
        candidates = []
        prompt = base_context + FINAL_SQL_PROMPT
        
        def sample_one(idx):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "你是一个专业的 StarRocks SQL 生成专家。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=2048
                )
                
                sql_text = response.choices[0].message.content.strip()
                sql_text = self._extract_sql(sql_text)
                return sql_text
            except Exception as e:
                logging.warning(f"⚠️ [{sql_id}] 候选采样失败 ({idx+1}/{k}): {e}")
                return None
        
        # 并行采样
        with ThreadPoolExecutor(max_workers=min(max_workers, k)) as executor:
            futures = [executor.submit(sample_one, i) for i in range(k)]
            
            for future in as_completed(futures):
                sql_text = future.result()
                if sql_text:
                    candidates.append(sql_text)
                    self.stats['total_candidates'] += 1
        
        return candidates
    
    def _extract_sql(self, text: str) -> Optional[str]:
        """提取第一条 SQL 语句"""
        # 去除代码块标记
        text = re.sub(r'^```[sS][qQ][lL]\s*\n?', '', text)
        text = re.sub(r'^```\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)
        text = text.strip()
        
        # 查找第一个 SELECT 或 WITH
        match = re.search(r'(SELECT|WITH)\b.*?;', text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(0).strip()
        
        # 如果没有分号，取整个文本
        if re.search(r'\b(SELECT|WITH)\b', text, re.IGNORECASE):
            sql = text.strip()
            if not sql.endswith(';'):
                sql += ';'
            return sql
        
        return None
    
    def evaluate_candidates(
        self,
        candidates: List[str],
        gold_fingerprint: str,
        gold_status: str,
        sql_id: str
    ) -> List[Tuple[str, float]]:
        """
        评估候选 SQL 并打分
        
        Args:
            candidates: 候选 SQL 列表
            gold_fingerprint: gold 的指纹
            gold_status: gold 的状态
            sql_id: SQL ID
        
        Returns:
            List[(sql, reward)]
        """
        scored = []
        
        for idx, cand_sql in enumerate(candidates):
            # 检查缓存
            cached = self.sql_cache.get(cand_sql)
            if cached:
                result = cached
            else:
                # 执行候选 SQL
                result = run_sql_and_get_feedback(
                    sql_id=f"{sql_id}_cand{idx}",
                    sql_text=cand_sql,
                    save_to_file=False
                )
                self.sql_cache.set(cand_sql, result)
            
            # 计算 reward
            reward = self._compute_reward(
                result=result,
                gold_fingerprint=gold_fingerprint,
                gold_status=gold_status
            )
            
            scored.append((cand_sql, reward))
        
        return scored
    
    def _compute_reward(
        self, 
        result: Dict, 
        gold_fingerprint: str,
        gold_status: str
    ) -> float:
        """
        计算 reward
        
        规则：
        - error -> -1.0
        - 指纹匹配 -> +1.0
        - gold 为空且 cand 也为空 -> +1.0
        - gold 非空但 cand 为空 -> -0.4
        - 其他不匹配 -> -0.2
        """
        cand_status = result.get('status')
        cand_fingerprint = result.get('result_fingerprint', '')
        
        # 执行错误
        if cand_status == 'error':
            return -1.0
        
        # 指纹匹配
        if cand_fingerprint == gold_fingerprint:
            return +1.0
        
        # 都为空（即使指纹不同，只要都是空结果就算对）
        if gold_status == 'success_empty' and cand_status == 'success_empty':
            return +1.0
        
        # gold 非空，cand 为空
        if gold_status == 'success_non_empty' and cand_status == 'success_empty':
            return -0.4
        
        # 其他不匹配
        return -0.2
    
    def select_best_and_worst(
        self, 
        scored: List[Tuple[str, float]]
    ) -> Optional[Tuple[str, str, float, float]]:
        """
        选择最好和最差的候选
        
        Returns:
            (chosen_sql, rejected_sql, chosen_reward, rejected_reward) 或 None
        """
        if not scored:
            return None
        
        # 按 reward 排序
        scored = sorted(scored, key=lambda x: x[1], reverse=True)
        
        chosen_sql, chosen_reward = scored[0]
        rejected_sql, rejected_reward = scored[-1]
        
        # 检查是否有 margin
        if chosen_reward == rejected_reward:
            return None
        
        return chosen_sql, rejected_sql, chosen_reward, rejected_reward


def write_jsonl(path: Path, data: List[Dict]):
    """写入 JSONL 文件"""
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def read_jsonl(path: Path) -> List[Dict]:
    """读取 JSONL 文件"""
    if not path.exists():
        return []
    
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_input_data(input_path: str) -> List[Dict]:
    """加载输入数据（支持 JSON 和 JSONL）"""
    path = Path(input_path)
    
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    if input_path.endswith('.jsonl'):
        return read_jsonl(path)
    else:
        return load_json(input_path)


def main():
    parser = argparse.ArgumentParser(description='从 gold 样本生成 SFT 与 DPO 训练数据')
    
    parser.add_argument('--input', type=str, required=True, help='训练样本文件（JSON 或 JSONL）')
    parser.add_argument('--schema', type=str, required=True, help='Schema 文件')
    parser.add_argument('--outdir', type=str, required=True, help='输出目录')
    parser.add_argument('--repair_per_item', type=int, default=2, help='每条样本生成的 repair 负样本数')
    parser.add_argument('--k', type=int, default=6, help='DPO 候选采样数')
    parser.add_argument('--temperature', type=float, default=0.7, help='采样温度')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--workers', type=int, default=4, help='并行线程数（用于 DPO 候选采样）')
    parser.add_argument('--batch_save', type=int, default=5, help='每处理多少条保存一次（减少 IO 开销）')
    parser.add_argument('--skip_dpo', action='store_true', help='不生成 DPO')
    parser.add_argument('--resume', action='store_true', help='断点续跑')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # 创建输出目录
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # 输出文件路径
    gold_exec_meta_path = outdir / 'gold_exec_meta.jsonl'
    sft_final_path = outdir / 'sft_final.jsonl'
    sft_repair_path = outdir / 'sft_repair.jsonl'
    dpo_pairs_path = outdir / 'dpo_pairs.jsonl'
    stats_path = outdir / 'stats.json'
    
    # 断点续跑：读取已处理的 sql_id
    processed_ids = set()
    if args.resume:
        for path in [gold_exec_meta_path, sft_final_path]:
            if path.exists():
                data = read_jsonl(path)
                processed_ids.update(item.get('meta', {}).get('sql_id') for item in data)
        logging.info(f"✅ 断点续跑: 已处理 {len(processed_ids)} 条样本")
    
    # 加载数据
    logging.info(f"📂 加载输入数据: {args.input}")
    input_data = load_input_data(args.input)
    logging.info(f"✅ 加载 {len(input_data)} 条样本")
    
    logging.info(f"📂 加载 Schema: {args.schema}")
    schema_json = load_json(args.schema)
    schema_index = build_schema_index(schema_json)
    logging.info(f"✅ Schema 索引构建完成: {len(schema_index)} 个表")
    
    # 初始化缓存和生成器
    sql_cache = SQLCache()
    repair_gen = RepairSampleGenerator(sql_cache)
    
    # 尝试初始化 LLM 客户端
    client = None
    model_name = None
    skip_dpo_model_unavailable = False
    
    if not args.skip_dpo:
        try:
            client, model_name = _get_llm_client_and_model()
            logging.info(f"✅ LLM 客户端初始化成功: {model_name}")
        except Exception as e:
            logging.warning(f"⚠️ LLM 客户端初始化失败: {e}")
            logging.warning("⚠️ 将跳过 DPO 生成")
            skip_dpo_model_unavailable = True
    
    dpo_gen = None
    if client and not args.skip_dpo:
        dpo_gen = DPOCandidateGenerator(sql_cache, client, model_name, args.temperature)
    
    # 统计信息
    stats = {
        'total_gold': len(input_data),
        'gold_executable': 0,
        'gold_error': 0,
        'sft_final_count': 0,
        'sft_repair_count': 0,
        'dpo_pair_count': 0,
        'dpo_skip_reason_counts': defaultdict(int),
        'repair_attempts': 0,
        'repair_success': 0,
        'dpo_skipped_model_unavailable': skip_dpo_model_unavailable
    }
    
    # 输出数据
    gold_exec_meta = []
    sft_final = []
    sft_repair = []
    dpo_pairs = []
    
    # 处理每条样本
    for idx, item in enumerate(input_data):
        sql_id = item.get('sql_id', f'unknown_{idx}')
        
        # 跳过已处理
        if sql_id in processed_ids:
            logging.info(f"⏭️ [{idx+1}/{len(input_data)}] {sql_id} 已处理，跳过")
            continue
        
        logging.info(f"\n{'='*80}")
        logging.info(f"🚀 [{idx+1}/{len(input_data)}] 处理样本: {sql_id}")
        logging.info(f"{'='*80}")
        
        question = item.get('question', '').strip()
        table_list = item.get('table_list', [])
        knowledge = item.get('knowledge', '').strip()
        gold_sql = item.get('sql', '').strip()
        
        if not gold_sql:
            logging.warning(f"⚠️ [{sql_id}] 缺少 gold SQL，跳过")
            continue
        
        # 1. 执行 gold SQL
        logging.info(f"📊 [{sql_id}] 执行 gold SQL")
        gold_result = run_sql_and_get_feedback(
            sql_id=f"{sql_id}_gold",
            sql_text=gold_sql,
            save_to_file=False
        )
        sql_cache.set(gold_sql, gold_result)
        
        gold_status = gold_result.get('status')
        gold_fingerprint = gold_result.get('result_fingerprint', '')
        gold_row_count = gold_result.get('row_count', 0)
        gold_error = gold_result.get('error_message', '')
        
        # 记录 gold 执行元数据
        gold_exec_meta.append({
            'sql_id': sql_id,
            'gold_sql': gold_sql,
            'status': gold_status,
            'fingerprint': gold_fingerprint,
            'row_count': gold_row_count,
            'error_message': gold_error
        })
        
        # 如果 gold 执行失败，跳过该样本
        if gold_status == 'error':
            stats['gold_error'] += 1
            logging.warning(f"⚠️ [{sql_id}] Gold SQL 执行失败，跳过: {gold_error}")
            continue
        
        stats['gold_executable'] += 1
        logging.info(f"✅ [{sql_id}] Gold SQL 执行成功: {gold_status}, {gold_row_count} 行")
        
        # 2. 构建 base context（用于 SFT Final 和 DPO）
        base_context = build_base_context(
            item=item,
            schema_index=schema_index,
            few_shot_examples=None,
            vector_db=None
        )
        
        # 3. 生成 SFT Final
        sft_final.append({
            'messages': [
                {'role': 'system', 'content': '你是一个专业的 StarRocks SQL 生成专家。'},
                {'role': 'user', 'content': base_context + FINAL_SQL_PROMPT}
            ],
            'assistant': gold_sql,
            'meta': {
                'sql_id': sql_id,
                'table_list': table_list
            }
        })
        stats['sft_final_count'] += 1
        logging.info(f"✅ [{sql_id}] 生成 SFT Final")
        
        # 4. 生成 Repair 负样本
        logging.info(f"🔧 [{sql_id}] 生成 {args.repair_per_item} 个 repair 负样本")
        for r_idx in range(args.repair_per_item):
            repair_result = repair_gen.try_generate_repair_sample(
                gold_sql=gold_sql,
                table_list=table_list,
                sql_id=sql_id,
                max_attempts=5
            )
            
            if repair_result:
                bad_sql, error_msg, bad_type = repair_result
                
                # 构建 repair prompt
                repair_prompt = (
                    base_context + "\n\n" +
                    f"下面SQL执行报错：\n{error_msg}\n\n" +
                    f"SQL:\n{bad_sql}\n\n" +
                    "请修复并只输出一条可直接执行的StarRocks SQL："
                )
                
                sft_repair.append({
                    'messages': [
                        {'role': 'system', 'content': '你是一个专业的 StarRocks SQL 生成专家。'},
                        {'role': 'user', 'content': repair_prompt}
                    ],
                    'assistant': gold_sql,
                    'meta': {
                        'sql_id': sql_id,
                        'bad_type': bad_type,
                        'bad_sql': bad_sql,
                        'error_message': error_msg
                    }
                })
                stats['sft_repair_count'] += 1
                logging.info(f"  ✅ [{sql_id}] Repair #{r_idx+1}: {bad_type}")
            else:
                logging.warning(f"  ⚠️ [{sql_id}] Repair #{r_idx+1} 生成失败")
        
        # 5. 生成 DPO 候选
        if dpo_gen and not args.skip_dpo:
            logging.info(f"🎯 [{sql_id}] 采样 {args.k} 个 DPO 候选")
            
            candidates = dpo_gen.sample_sql_candidates(
                base_context=base_context,
                k=args.k,
                sql_id=sql_id,
                max_workers=args.workers
            )
            
            if not candidates:
                logging.warning(f"⚠️ [{sql_id}] 未采样到有效候选，跳过 DPO")
                stats['dpo_skip_reason_counts']['no_candidates'] += 1
                continue
            
            logging.info(f"  ✅ [{sql_id}] 成功采样 {len(candidates)} 个候选")
            
            # 评估候选
            scored = dpo_gen.evaluate_candidates(
                candidates=candidates,
                gold_fingerprint=gold_fingerprint,
                gold_status=gold_status,
                sql_id=sql_id
            )
            
            # 选择最好和最差
            pair = dpo_gen.select_best_and_worst(scored)
            
            if pair:
                chosen_sql, rejected_sql, chosen_reward, rejected_reward = pair
                
                dpo_pairs.append({
                    'messages': [
                        {'role': 'system', 'content': '你是一个专业的 StarRocks SQL 生成专家。'},
                        {'role': 'user', 'content': base_context + FINAL_SQL_PROMPT}
                    ],
                    'chosen': chosen_sql,
                    'rejected': rejected_sql,
                    'meta': {
                        'sql_id': sql_id,
                        'chosen_reward': chosen_reward,
                        'rejected_reward': rejected_reward
                    }
                })
                stats['dpo_pair_count'] += 1
                dpo_gen.stats['pairs_generated'] += 1
                logging.info(f"  ✅ [{sql_id}] DPO pair: chosen={chosen_reward:.2f}, rejected={rejected_reward:.2f}")
            else:
                logging.warning(f"  ⚠️ [{sql_id}] 无有效 margin，跳过 DPO")
                stats['dpo_skip_reason_counts']['no_margin'] += 1
        
        # 定期保存（减少 IO 开销）
        if (idx + 1) % args.batch_save == 0:
            write_jsonl(gold_exec_meta_path, gold_exec_meta)
            write_jsonl(sft_final_path, sft_final)
            write_jsonl(sft_repair_path, sft_repair)
            if dpo_pairs:
                write_jsonl(dpo_pairs_path, dpo_pairs)
            logging.info(f"💾 已保存中间结果 ({idx+1}/{len(input_data)})")
            
            # 显示进度和速度
            if idx > 0:
                progress = (idx + 1) / len(input_data) * 100
                logging.info(f"📈 进度: {progress:.1f}% | 已处理: {idx+1}/{len(input_data)}")
    
    # 最终保存
    logging.info(f"\n{'='*80}")
    logging.info("💾 保存最终结果")
    logging.info(f"{'='*80}")
    
    write_jsonl(gold_exec_meta_path, gold_exec_meta)
    write_jsonl(sft_final_path, sft_final)
    write_jsonl(sft_repair_path, sft_repair)
    if dpo_pairs:
        write_jsonl(dpo_pairs_path, dpo_pairs)
    
    # 更新统计
    stats['repair_attempts'] = repair_gen.stats['attempts']
    stats['repair_success'] = repair_gen.stats['success']
    
    if dpo_gen:
        stats['dpo_total_candidates'] = dpo_gen.stats['total_candidates']
    
    # 转换 defaultdict 为 dict
    stats['dpo_skip_reason_counts'] = dict(stats['dpo_skip_reason_counts'])
    
    # 保存统计
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # 打印摘要
    print(f"\n{'='*80}")
    print("📊 生成完成！统计摘要：")
    print(f"{'='*80}")
    print(f"总样本数: {stats['total_gold']}")
    print(f"可执行样本: {stats['gold_executable']}")
    print(f"执行失败: {stats['gold_error']}")
    print(f"\nSFT Final: {stats['sft_final_count']}")
    print(f"SFT Repair: {stats['sft_repair_count']}")
    print(f"  - 尝试次数: {stats['repair_attempts']}")
    print(f"  - 成功次数: {stats['repair_success']}")
    print(f"  - 成功率: {stats['repair_success']/max(stats['repair_attempts'],1)*100:.1f}%")
    
    if not args.skip_dpo and not skip_dpo_model_unavailable:
        print(f"\nDPO Pairs: {stats['dpo_pair_count']}")
        print(f"  - 总候选数: {stats.get('dpo_total_candidates', 0)}")
        print(f"  - 跳过原因:")
        for reason, count in stats['dpo_skip_reason_counts'].items():
            print(f"    - {reason}: {count}")
    else:
        print(f"\nDPO: 已跳过")
        if skip_dpo_model_unavailable:
            print(f"  原因: 模型不可用")
    
    print(f"\n输出目录: {outdir}")
    print(f"  - gold_exec_meta.jsonl: {len(gold_exec_meta)} 条")
    print(f"  - sft_final.jsonl: {len(sft_final)} 条")
    print(f"  - sft_repair.jsonl: {len(sft_repair)} 条")
    if dpo_pairs:
        print(f"  - dpo_pairs.jsonl: {len(dpo_pairs)} 条")
    print(f"  - stats.json")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
