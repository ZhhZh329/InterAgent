"""
CodingAgent的所有Prompt模板
分离Prompt提高代码可读性和可维护性
"""


def build_package_installation_prompt(error_message: str) -> str:
    """构建包安装分析的prompt

    Args:
        error_message: ModuleNotFoundError错误信息

    Returns:
        完整的prompt字符串
    """
    return f"""你是一个Python包管理专家。请分析以下ModuleNotFoundError错误信息，确定需要安装的正确包名。

**错误信息：**
{error_message}

**任务要求：**
1. 识别缺失的模块名（import语句中的名称）
2. 确定对应的PyPI包名（可能与模块名不同）
3. 返回纯JSON格式，格式如下：

{{
    "missing_module": "<缺失的模块名>",
    "package_name": "<需要安装的包名>",
    "reasoning": "<为什么是这个包名的简短说明>"
}}

**常见映射示例：**
- import cv2 → opencv-python
- import PIL → Pillow
- import sklearn → scikit-learn
- import yaml → PyYAML
- import dotenv → python-dotenv

只返回JSON，不要有其他文字。"""


def build_data_loader_code_generation_prompt(
    research_topic: str,
    data_structure_analysis: str,
    dataset_path: str
) -> str:
    """构建数据加载代码生成的prompt

    Args:
        research_topic: 研究主题
        data_structure_analysis: 数据结构分析
        dataset_path: 数据集路径

    Returns:
        完整的prompt字符串
    """
    return f"""你是一个专业的Python数据工程师，请生成一个数据加载脚本。

**研究主题**: {research_topic}

**数据集路径**: {dataset_path}

**数据结构分析**:
{data_structure_analysis}

**要求**:
1. 创建load_data()函数，返回(train_data, test_data)元组
2. 支持USE_SMALL_SAMPLE环境变量（用于快速测试）
3. 完整的错误处理
4. 日志记录
5. 在__main__中测试并打印数据概览

**输出格式**:
只输出Python代码，使用```python代码块包裹。"""


def build_research_code_generation_prompt(
    research_topic: str,
    research_goal: str,
    init_analysis: str,
    data_structure_info: str,
    submission_file_name: str,
    device_info: dict
) -> str:
    """构建研究代码生成的prompt

    Args:
        research_topic: 研究主题
        research_goal: 研究目标
        init_analysis: 初步分析结果
        data_structure_info: 数据结构信息
        submission_file_name: 提交文件名
        device_info: 设备信息

    Returns:
        完整的prompt字符串
    """
    device_type = device_info.get('device_type', 'cpu')
    device_name = device_info.get('device_name', 'CPU')

    return f"""你是一个专业的机器学习工程师，请根据以下信息生成完整的research.py代码。

**研究主题**: {research_topic}

**研究目标**: {research_goal}

**初步分析结果**:
{init_analysis}

**数据结构信息**:
{data_structure_info}

**提交文件名**: {submission_file_name}

**计算设备**: {device_type} ({device_name})

**代码要求**:

1. **必须包含的部分**:
   - 导入必要的库
   - 从load_research_data.py导入数据加载函数
   - 完整的数据预处理pipeline
   - 模型训练代码
   - 预测和结果保存
   - 环境变量支持：USE_SMALL_SAMPLE和QUICK_TEST

2. **严禁使用任何Dummy/占位机制**:
   * 绝对禁止使用sklearn.dummy.DummyClassifier、DummyRegressor等任何dummy模型
   * 绝对禁止使用random.choice()、np.random.random()等生成随机预测值
   * 绝对禁止使用固定常数（如全0、全1、均值）作为预测结果
   * 绝对禁止使用简单规则（if-else规则、阈值判断）代替真实ML模型
   * 绝对禁止在训练失败时fallback到dummy机制，训练失败就应该抛出异常
   * 必须使用真正的机器学习模型进行训练和预测，没有例外

3. **严禁使用K折交叉验证**：
   * 由于运行时间限制，禁止使用K折交叉验证（KFold, StratifiedKFold等）
   * 禁止使用cross_val_score、cross_validate等交叉验证函数
   * 如需验证集，使用简单的train_test_split一次性划分即可
   * 直接在训练集上训练模型，不要重复训练K次

4. **设备优化**:
   - 根据可用设备（{device_type}）选择合适的算法
   - CPU环境：避免使用深度学习，优先sklearn模型
   - GPU环境：可以使用PyTorch/TensorFlow

5. **环境变量支持**:
   ```python
   USE_SMALL_SAMPLE = os.getenv('USE_SMALL_SAMPLE', '0') == '1'
   QUICK_TEST = os.getenv('QUICK_TEST', '0') == '1'

   if USE_SMALL_SAMPLE:
       # 只使用少量数据进行快速测试
       ...

   if QUICK_TEST:
       # 使用更简单快速的模型配置
       ...
   ```

6. **输出要求**:
   - 生成{submission_file_name}文件
   - 格式必须符合提交要求
   - 包含所有测试集样本的预测结果

7. **代码规范**:
   - 完整的注释
   - 清晰的日志输出
   - 合理的异常处理
   - 可读性强的变量命名

**输出格式**:
只输出完整的research.py代码，使用```python代码块包裹。不要包含额外的解释文字。"""


def build_debug_prompt(error_message: str, error_context: str, code_content: str) -> str:
    """构建代码调试的prompt

    Args:
        error_message: 错误信息
        error_context: 错误上下文
        code_content: 当前代码内容

    Returns:
        完整的prompt字符串
    """
    return f"""你是一个专业的Python调试专家，请修复以下代码中的错误。

**错误信息**:
{error_message}

**错误上下文**:
{error_context}

**当前代码**:
```python
{code_content}
```

**调试要求**:
1. 仔细分析错误原因
2. 提供修复后的完整代码
3. 保持原有功能不变
4. 添加必要的错误处理
5. 确保代码可以正常运行

**严禁使用任何Dummy/占位机制**:
* 绝对禁止使用sklearn.dummy.DummyClassifier、DummyRegressor等任何dummy模型
* 绝对禁止使用random.choice()、np.random.random()等生成随机预测值
* 绝对禁止使用固定常数（如全0、全1、均值）作为预测结果
* 绝对禁止使用简单规则（if-else规则、阈值判断）代替真实ML模型
* 绝对禁止在训练失败时fallback到dummy机制，训练失败就应该抛出异常
* 必须使用真正的机器学习模型进行训练和预测，没有例外

**严禁数据加载容错机制**:
* 绝对禁止创建dummy数据、示例数据或模拟数据
* 绝对禁止在找不到数据文件时使用假数据
* 如果数据文件不存在，必须抛出FileNotFoundError，不要尝试创建替代数据
* 必须加载真实的数据文件，没有例外

**输出格式**:
只输出修复后的完整代码，使用```python代码块包裹。"""


def build_optimization_prompt(
    code_content: str,
    performance_issue: str,
    optimization_goals: str
) -> str:
    """构建性能优化的prompt

    Args:
        code_content: 当前代码内容
        performance_issue: 性能问题描述
        optimization_goals: 优化目标

    Returns:
        完整的prompt字符串
    """
    return f"""你是一个专业的Python性能优化专家，请优化以下代码的性能。

**当前代码**:
```python
{code_content}
```

**性能问题**:
{performance_issue}

**优化目标**:
{optimization_goals}

**优化建议**:
1. 算法优化：使用更高效的算法和数据结构
2. 向量化：使用NumPy/Pandas的向量化操作替代循环
3. 并行化：合理使用n_jobs等并行参数
4. 内存优化：减少内存占用，使用生成器等
5. 模型简化：在保证效果的前提下简化模型

**严格要求**:
- 保持功能完整性
- 确保代码可读性
- 添加性能改进说明注释

**输出格式**:
只输出优化后的完整代码，使用```python代码块包裹。"""


# ===== 为未来扩展准备的prompt模板占位符 =====

def build_code_improvement_with_memory_prompt(
    code_content: str,
    test_results: str,
    memory_context: list
) -> str:
    """构建基于记忆的代码改进prompt（Stage 3使用）

    Args:
        code_content: 当前代码
        test_results: 测试结果
        memory_context: 历史成功案例的记忆上下文

    Returns:
        完整的prompt字符串
    """
    # TODO: Stage 3实现
    return f"""基于历史成功经验改进代码：

**当前代码**:
```python
{code_content}
```

**测试结果**:
{test_results}

**历史成功案例参考**:
{memory_context}

请参考历史成功案例，改进当前代码。"""
