import hashlib
import json
from collections import Counter
from typing import Any, Dict, List


def extract_and_normalize_result(result_list: List[Dict[str, Any]]) -> Counter:
    """Normalize result rows into a multiset while ignoring row order."""
    counter: Counter = Counter()
    if not result_list:
        return counter

    for item in result_list:
        if isinstance(item, dict):
            row_vals = tuple("" if v is None else str(v) for v in item.values())
        else:
            row_vals = (str(item),)
        counter[row_vals] += 1
    return counter


def build_result_fingerprint(result_list: List[Dict[str, Any]]) -> str:
    normalized = extract_and_normalize_result(result_list)
    sorted_items = sorted(normalized.items(), key=lambda x: x[0])
    payload = json.dumps(sorted_items, ensure_ascii=False)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def is_result_match(pred_result: List[Dict[str, Any]], gold_result: List[Dict[str, Any]]) -> bool:
    return extract_and_normalize_result(pred_result) == extract_and_normalize_result(gold_result)


def compare_execution_to_gold(exec_record: Dict[str, Any], gold_result: List[Dict[str, Any]]) -> Dict[str, Any]:
    pred_result = exec_record.get("result_data", []) or exec_record.get("result", []) or []
    matched = is_result_match(pred_result, gold_result)
    return {
        "matched": matched,
        "pred_fingerprint": build_result_fingerprint(pred_result),
        "gold_fingerprint": build_result_fingerprint(gold_result),
    }

