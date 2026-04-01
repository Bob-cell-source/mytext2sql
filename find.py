import json
import os
from typing import List, Dict, Any, Set, Optional

# 假设您的 load_json 函数可以读取文件内容
# 请根据您的实际环境修改文件路径
# EXE_JSON_PATH = "D:/下载/TGAC/final_for_student/result/dataset_exe_result.json" 
EXE_JSON_PATH = "D:/下载/TGAC/final_for_student/result/dataset_exe_result111.json"   # 假设这是第一个文件路径

RESULT_JSON_PATH = "D:/下载/TGAC/final_for_student/answer.json" # 假设这是第二个文件路径
import json
import os
from typing import List, Dict, Any, Set, Optional
from collections import Counter

# 假设您的 load_json 函数可以读取文件内容
# 请根据您的实际环境修改文件路径
# EXE_JSON_PATH = "D:/下载/TGAC/final_for_student/result/dataset_exe_result.json" 

def load_json(path: str) -> List[Dict[str, Any]]:
    """读取 JSON 文件并返回列表内容"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {path}")
        return []
    except json.JSONDecodeError:
        print(f"错误: 文件 {path} 不是有效的 JSON 格式")
        return []

def save_json(path: str, data: Any) -> None:
    dirp = os.path.dirname(path)
    if dirp:
        os.makedirs(dirp, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def build_id_map(data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    m: Dict[str, Dict[str, Any]] = {}
    for item in data or []:
        sid = item.get("sql_id")
        if sid:
            m[str(sid)] = item
    return m

def replace_mismatched(exe_data: List[Dict[str, Any]], result_data: List[Dict[str, Any]], mismatch_ids: List[str]) -> List[Dict[str, Any]]:
    result_map = build_id_map(result_data)
    mismatch_set = set(str(x) for x in mismatch_ids or [])
    out: List[Dict[str, Any]] = []
    for item in exe_data or []:
        sid = str(item.get("sql_id"))
        if sid in mismatch_set and sid in result_map:
            out.append(result_map[sid])
        else:
            out.append(item)
    return out

def extract_and_normalize_result(result_list: List[Dict[str, Any]]) -> Counter:
    """
    将 result 列表规范化为行的多重集合 (Counter)。
    每一行(row)被表示为按 dict 的 insertion-order 提取值后
    的字符串元组 tuple(str(value), ...)。这样可以忽略字段名差异，
    但会对字段数量/字段值保持敏感（缺少字段或多余字段会导致不匹配）。
    """
    counter: Counter = Counter()
    if not result_list:
        return counter

    for item in result_list:
        if isinstance(item, dict):
            # 保留列间的对应关系（按 dict 的 insertion order 提取值）
            row_vals = tuple("" if v is None else str(v) for v in item.values())
        else:
            # 如果不是 dict，按单列处理
            row_vals = (str(item),)
        counter[row_vals] += 1

    return counter

def compare_results(exe_data: List[Dict], result_data: List[Dict]) -> Dict[str, Any]:
    """
    比较两个列表文件，统计结果集一致的 SQL ID。
    比较方式：将每个 sql_id 对应的 result 转为行的多重集合 (Counter of tuples of values)，
    忽略列名但保持列之间的对应关系与列数；因此字段名不同但值和行数相同的情况会被判定为一致；
    若某一边缺少或多出字段则判定为不一致。
    """
    # 1. 将两个文件的数据分别映射为 {sql_id: Counter(rows)} 字典

    exe_map: Dict[str, Counter] = {}
    for item in exe_data or []:
        sql_id = item.get("sql_id")
        result_list = item.get("result", [])
        if sql_id is not None:
            exe_map[str(sql_id)] = extract_and_normalize_result(result_list)

    result_map: Dict[str, Counter] = {}
    for item in result_data or []:
        sql_id = item.get("sql_id")
        result_list = item.get("result", [])
        if sql_id is not None:
            result_map[str(sql_id)] = extract_and_normalize_result(result_list)

    # 2. 比较并统计
    matched_ids: List[str] = []
    mismatched_ids: List[str] = []

    for sql_id, exe_counter in exe_map.items():
        ground_truth_counter = result_map.get(sql_id)
        if ground_truth_counter is not None:
            if exe_counter == ground_truth_counter:
                matched_ids.append(sql_id)
            else:
                mismatched_ids.append(sql_id)

    return {
        "total_ids_in_execution": len(exe_map),
        "total_ids_in_ground_truth": len(result_map),
        "total_matched_count": len(matched_ids),
        "matched_sql_ids": matched_ids,
        "total_mismatched_count": len(mismatched_ids),
        "mismatched_sql_ids": mismatched_ids
    }

if __name__ == "__main__":
    exe_data = load_json(EXE_JSON_PATH)
    result_data = load_json(RESULT_JSON_PATH)

    comparison_results = compare_results(exe_data, result_data)

    print("=" * 30)
    print("✨ 结果集一致性统计报告 ✨")
    print("=" * 30)
    print(f"一致的 SQL ID 数量: {comparison_results['total_matched_count']}")
    print("【一致的 SQL ID】:")
    print(comparison_results['matched_sql_ids'])
    print("-" * 30)
    print(f"不一致的 SQL ID 数量: {comparison_results['total_mismatched_count']}")
    print("【不一致的 SQL ID】:")
    print(comparison_results['mismatched_sql_ids'])
    out_path = "D:/下载/TGAC/final_for_student/result/dataset_exe_result_replaced.json"
    merged = replace_mismatched(exe_data, result_data, comparison_results['mismatched_sql_ids'])
    save_json(out_path, merged)
    print("-" * 30)
    print(f"已替换并输出到: {out_path}")