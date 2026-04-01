import re
from typing import List, Dict, Any, Set, Optional

# --- 存储历史正确ID的集合（模拟持久化存储）---
# 在真实的生产环境中，这个集合应该从 JSON 文件中读取和写入。
MASTER_CORRECT_IDS: Set[int] = set()
[]
import json

# 定义 answer.json 的文件路径
answer_file_path = r"d:\下载\TGAC\final_for_student\answer.json"
previous_correct_list = []

try:
    # 打开并加载 JSON 文件
    with open(answer_file_path, 'r', encoding='utf-8') as f:
        answer_data = json.load(f)
    
    # 从 answer.json 的 key (例如 "sql_1") 中提取数字 ID
    for i in answer_data:
        try:
            # 假设格式是 "sql_数字"
            numeric_part = int(i['sql_id'].split('_')[-1])
            previous_correct_list.append(numeric_part)
        except (ValueError, IndexError):
            # 如果 key 的格式不符合预期，打印一个警告
            print(f"警告: 无法从 '{sql_id_str}' 中解析出数字 ID。")
        
    
    # 为了与之前的行为保持一致，可以对列表进行排序
    previous_correct_list.sort()

except FileNotFoundError:
    print(f"错误: 文件 '{answer_file_path}' 未找到。")
except json.JSONDecodeError:
    print(f"错误: 解析 JSON 文件 '{answer_file_path}' 失败。")
print(len(previous_correct_list))
MASTER_CORRECT_IDS.update(previous_correct_list)
class SQLTracker:
    def __init__(self, initial_ids: Optional[Set[int]] = None):
        """
        初始化追踪器，加载历史正确ID集合。
        """
        global MASTER_CORRECT_IDS
        # 使用全局集合作为持久化存储
        self.master_correct_ids: Set[int] = initial_ids if initial_ids is not None else MASTER_CORRECT_IDS
    
    @staticmethod
    def parse_input(input_string: str) -> Set[int]:
        """
        解析输入字符串，提取状态为 '1' 的 SQL ID。
        规则：以 'sql_' 分割，每个块的最后一个数字是状态，前面是 ID。
        """
        # Split the string by 'sql_' and ignore the first empty element
        chunks = input_string.split("sql_")[1:]
        current_correct_ids = set()

        for chunk in chunks:
            if not chunk:
                continue
            
            # The last digit is the status (0 or 1)
            status = chunk[-1]
            
            # The rest is the numeric ID
            id_str = chunk[:-1]
            
            if status == '1':
                try:
                    # Convert to integer and add to the set
                    current_correct_ids.add(int(id_str))
                except ValueError:
                    # 忽略无效的 ID 字符串
                    pass
        return current_correct_ids

    def update_results(self, new_result_string: str) -> Dict[str, Any]:
        """
        处理新的结果字符串，计算增量，并更新主列表。
        """
        # 1. 提取当前批次中所有正确的ID
        current_correct_ids = self.parse_input(new_result_string)
        
        # 2. 计算增量（相对于 MASTER_CORRECT_IDS）：新列表中有，但主列表没有的 ID
        newly_added_ids = current_correct_ids - self.master_correct_ids

        # 计算新丢失的ID：主列表有，但新列表没有的ID
        newly_missing_ids = self.master_correct_ids - current_correct_ids
        
        # 3. 更新主列表
        self.master_correct_ids.update(newly_added_ids)
        
        # 4. 返回结果
        return {
            "processed_correct_ids_count": len(current_correct_ids),
            "newly_added_ids_count": len(newly_added_ids),
            "newly_added_ids_list": sorted(list(newly_added_ids)),
            "newly_missing_ids_count": len(newly_missing_ids),
            "newly_missing_ids_list": sorted(list(newly_missing_ids)),
            "master_correct_ids_count": len(self.master_correct_ids),
            "master_correct_ids_list": sorted(list(self.master_correct_ids))
        }

# --- 示例使用 ---

# 初始化追踪器
tracker = SQLTracker()

# 第一次运行数据（您的输入）
initial_data =input()
# 第一次处理：所有结果都是增量
result_1 = tracker.update_results(initial_data)
print(result_1)