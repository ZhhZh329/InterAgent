"""
Evaluator的所有Prompt模板
分离Prompt提高代码可读性和可维护性
"""


def build_mlebench_output_parsing_prompt(stdout: str, stderr: str, benchmark_name: str) -> str:
    """构建MLEBench输出解析的prompt

    Args:
        stdout: 标准输出
        stderr: 标准错误
        benchmark_name: benchmark名称

    Returns:
        完整的prompt字符串
    """
    return f"""你是一个专业的日志解析助手。请从以下{benchmark_name}评估命令的输出中提取所有关键信息，并以纯JSON格式返回。

**输出内容：**

STDERR:
{stderr}

STDOUT:
{stdout}

**任务要求：**
1. 提取evaluation report中的所有字段（如果存在）：
   - competition_id
   - score
   - gold_medal, silver_medal, bronze_medal, any_medal, above_median
   - valid_submission
   - is_lower_better
   - 各种threshold值

2. 仔细查看STDERR中是否有错误信息，常见错误包括：
   - "Invalid submission: Expected the submission to have the same 'id' values" → ID值不匹配
   - "行数不匹配" / "row count mismatch"
   - "列名错误" / "column name mismatch"
   - "文件找不到" / "file not found"
   - 其他格式错误

3. **关键判断**：分析失败原因并设置正确的failure_type和requires_code_regeneration：

   需要重新生成代码的情况（requires_code_regeneration=true）：
   - "file_format_error": ID值不匹配、列名错误、行数不匹配、数据格式错误
   - "code_logic_error": valid_submission=false 且有错误信息

   不需要重新生成代码的情况（requires_code_regeneration=false）：
   - "file_not_found": 找不到提交文件
   - "command_error": 评估命令本身执行错误

**返回格式要求：**
必须返回一个纯JSON对象，不要有任何markdown标记或其他文本。JSON格式如下：

{{
    "evaluation_success": true/false,
    "score": <分数值或null>,
    "valid_submission": true/false,
    "gold_medal": true/false,
    "silver_medal": true/false,
    "bronze_medal": true/false,
    "any_medal": true/false,
    "above_median": true/false,
    "is_lower_better": true/false,
    "competition_id": "<竞赛ID>",
    "error_message": "<完整的错误信息>",
    "failure_type": "file_format_error|file_not_found|command_error|code_logic_error|unknown",
    "full_report": {{<完整的JSON report>}},
    "requires_code_regeneration": true/false
}}

**判断规则（非常重要）**：
1. 如果STDERR包含"Invalid submission"、"Expected the submission to have the same 'id'"、"列名"、"行数"等，设置：
   - failure_type: "file_format_error"
   - requires_code_regeneration: true

2. 如果valid_submission=false且score=null，设置：
   - failure_type: "code_logic_error"
   - requires_code_regeneration: true

3. 如果valid_submission=true且score不为null，设置：
   - evaluation_success: true
   - requires_code_regeneration: false

只返回JSON，不要有其他文字。
"""
