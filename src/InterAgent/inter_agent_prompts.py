"""
InterAgent的所有Prompt模板
分离Prompt提高代码可读性和可维护性
"""


def build_structure_analysis_prompt(structure_summary: str) -> str:
    """构建文件结构分析的prompt"""
    return f"""
请分析以下数据集的文件结构，判断数据集的组织方式和文件加载需求：

文件结构：
{structure_summary}

请按照以下格式分析并返回结果：

训练集文件: [识别哪些是训练数据文件]
测试集文件: [识别哪些是测试数据文件]
验证集文件: [识别哪些是验证数据文件，如果有的话]
额外文件: [识别是否有需要额外加载的文件，如图片、文本等]
数据格式: [判断数据格式，如CSV、图片、文本等]
加载建议: [给出数据加载的建议]

要求：
1. 基于文件名和结构判断数据集的组织方式
2. 识别是否需要加载额外的文件（如图片文件夹）
3. 给出合理的数据加载策略建议
4. 用中文回答，简洁明了
"""


def build_init_analysis_prompt(
    research_topic: str,
    research_goal: str,
    data_structure_info: str,
    train_files: str,
    test_files: str,
    data_format: str
) -> str:
    """构建初步分析的prompt"""
    return f"""
作为一名经验丰富的机器学习研究员，你需要基于数据结构和任务信息，给出技术路线分析和实现方案建议。

**研究主题**:
{research_topic}

**研究目标**:
{research_goal}

**数据加载测试结果**:
{data_structure_info}

**文件结构分析**:
训练集文件: {train_files}
测试集文件: {test_files}
数据格式: {data_format}

**数据集规模评估**:
请根据上述信息评估数据集规模，并据此选择合适的技术方案：
- **小数据集**（<10万行）：可以使用任何sklearn模型，内存占用不是问题
- **中等数据集**（10万-100万行）：需要注意内存使用，可使用sklearn但要优化参数
- **大数据集**（>100万行）：**强烈建议使用查找表(lookup table)或简单规则方法**，避免训练大型sklearn模型导致内存溢出
- 注意：如果训练数据文件大小>100MB，通常意味着是大数据集

请基于上述信息，提供技术路线分析和方案建议（不要写具体实现代码）：

## 1. 任务类型识别
- 这是什么类型的机器学习任务？（分类/回归/序列预测/聚类等）
- 任务的难点和挑战是什么？
- 评估指标应该是什么？

## 2. 可行技术路线（列举多个方案）
请列举2-3个可行的技术路线，从简单到复杂：

**路线A（最简单快速）**：
- 技术栈：（如sklearn、传统ML算法）
- 模型选择：（具体算法，如RandomForest、LogisticRegression等）
- 特征工程：（需要提取什么特征）
- 优势：（为什么简单快速）
- 预期效果：（baseline性能预期）

**路线B（中等复杂度）**：
- 技术栈：（如XGBoost、LightGBM等）
- 模型选择：
- 特征工程：
- 优势：
- 预期效果：

**路线C（更复杂/sota方向）**：
- 技术栈：（如深度学习、transformers等）
- 模型选择：
- 设计要点：
- 优势：
- 预期效果：

## 3. MVP极速方案推荐（最重要）
**根据数据集规模推荐第一版baseline方案**：

**首先评估数据规模**：
- 查看文件大小（训练集>100MB通常是大数据集）
- 估算总行数（从测试结果中获取）

**小数据集推荐**（<10万行）：
- 技术栈：sklearn + pandas
- 模型：RandomForest/XGBoost等传统ML模型
- 可以正常使用model.fit()训练

**大数据集推荐**（>100万行或文件>100MB）：
- **强烈推荐**：内存高效的方法（避免将全部数据加载到内存进行训练）
- **严禁**：训练需要大量内存的sklearn模型（RandomForest/XGBoost等）
- **原因**：大数据集训练sklearn模型会导致内存溢出（OOM killed, exit code -9）
- **通用策略**：根据任务特点选择合适的轻量级方法（如记忆化、增量学习、规则提取等）

**推荐方案详细说明**：
- **推荐技术栈**：（根据数据规模明确指定）
- **推荐方法**：（具体方法类型：内存高效方法/sklearn模型/深度学习）
- **核心理由**：（为什么这个方案最快最可靠，是否考虑了内存限制）
- **内存估算**：（评估方案是否会导致内存问题）
- **性能加速考虑**：（根据所选方法，分析如何避免运行缓慢）
  * 评估数据规模和计算复杂度
  * 识别性能瓶颈（串行处理、Python循环、内存拷贝等）
  * 提出针对性的加速策略（并行、向量化、GPU等）
  * 说明为什么这个策略适合当前方案
- **关键步骤**：
  1. 数据预处理：（如何处理数据）
  2. 主要方法：（使用何种技术实现）
  3. Fallback策略：（对于边界情况如何处理）
  4. 预测输出：（输出格式）
- **预计开发时间**：（15分钟内可完成）
- **成功率评估**：（调试成功的可能性，是否会OOM）

## 4. 数据处理注意事项
- 数据中可能存在的问题（缺失值、异常值等）
- 需要注意的数据格式问题
- 训练集和测试集的一致性检查

请给出清晰的技术路线分析，重点推荐最快最可行的MVP方案。
"""


def build_data_loader_generation_prompt(
    dataset_path: str,
    structure_summary: str,
    train_files: str,
    test_files: str,
    data_format: str,
    loading_suggestions: str
) -> str:
    """构建数据加载代码生成的prompt

    Args:
        dataset_path: 数据集路径
        structure_summary: 文件结构摘要
        train_files: 训练集文件信息
        test_files: 测试集文件信息
        data_format: 数据格式
        loading_suggestions: 加载建议

    Returns:
        完整的prompt字符串
    """
    return f"""
**重要：只能生成一个文件load_research_data.py，不要生成任何其他文件！**

创建一个数据加载脚本 load_research_data.py，要求：

数据集路径: {dataset_path}

**小样本支持**: 脚本需要支持环境变量USE_SMALL_SAMPLE=1时只加载少量数据用于快速测试

文件结构分析（完整路径示例）:
{structure_summary}

训练集文件: {train_files}
测试集文件: {test_files}
数据格式: {data_format}
加载建议: {loading_suggestions}

**数据组合要求：**
如果数据集中同时包含结构化数据（CSV/JSON/JSONL等）和非结构化数据（图片、音频、文本文件等），
需要智能地将它们组合起来：

1. **识别唯一标识符**：分析结构化数据的字段，找出可能的唯一标识符或文件引用字段
2. **数据匹配**：根据标识符将结构化数据与对应的文件数据匹配
3. **统一格式**：返回统一的数据格式，包含所有相关信息

例如：
- 如果表格数据中有文件路径/名称字段，匹配对应的外部文件
- 如果元数据中有标识符字段，匹配对应的数据文件
- 如果数据中有样本编号，匹配对应的数据文件

**严格要求：**
1. 只生成一个文件load_research_data.py，绝对不要生成其他任何文件！
2. 完整的import语句
3. 一个load_research_data()函数，返回(train_data, test_data)
4. 根据文件结构分析，编写适当的文件解析函数
5. **数据组合函数**：如果有多种数据类型，编写函数将它们智能组合
6. **唯一标识符识别**：分析并找出数据间的关联字段
7. 错误处理和日志记录
8. 在文件末尾添加if __name__ == "__main__":测试代码
9. 测试代码要打印前5个训练样本的基本信息，展示组合后的数据结构

**绝对禁止的行为：**
1. **禁止创建dummy数据、示例数据或模拟数据**：必须加载真实的数据文件
2. **禁止任何形式的容错机制**：如果数据文件不存在或加载失败，必须抛出异常，不要返回示例数据
3. **禁止使用fallback到假数据**：找不到文件时必须报错，不要生成假数据继续运行
4. **数据文件路径必须准确**：仔细检查文件结构分析中提供的实际文件名和路径
5. **如果文件不存在必须立即失败**：不要尝试创建替代数据，直接抛出FileNotFoundError

**数据组合示例：**
```python
# 示例：组合表格标注数据和外部文件
def combine_data(metadata, external_file_dir):
    combined_data = []
    for item in metadata:
        # 根据元数据中的路径信息找到对应文件
        file_path = os.path.join(external_file_dir, item['file_reference'])
        if os.path.exists(file_path):
            combined_data.append({{
                'file_path': file_path,
                'metadata': item,
                # 其他相关字段根据实际数据结构添加
            }})
    return combined_data
```

注意：
- 绝对只生成一个文件，文件名必须是load_research_data.py
- 不要生成README、配置文件或任何其他文件
- 代码要能直接运行
- 使用pandas、numpy等适当的库处理数据
- 根据实际文件格式编写对应的解析代码
- 对于特殊文件格式，根据文件结构分析中的内容示例来理解格式
- **重点**：智能识别和组合不同类型的数据

**数据完整性要求（非常重要）**：
- 在加载数据时，**绝对不能**使用dropna()或过滤掉NaN/None/空值的行
- 必须保留数据集中的所有行，包括含有缺失值的行
- 对于缺失值，应该保持原样（保留NaN）或用明确的默认值填充，但不能删除行
- 确保load_research_data()返回的数据行数与原始文件完全一致
- 这是为了保证后续生成的预测结果能够与测试集完全对应

请按照以下格式输出：
```python
# load_research_data.py的完整代码
```

**再次强调：只输出单个Python文件(load_research_data.py)，包含完整的数据加载和组合逻辑！**
"""


# ===== 为未来Stage 2-4准备的prompt模板占位符 =====

def build_memory_save_prompt(experiment_data: dict) -> str:
    """构建实验记录保存的prompt（Stage 2使用）

    Args:
        experiment_data: 实验数据字典

    Returns:
        完整的prompt字符串
    """
    # TODO: Stage 2实现
    return f"""
请分析以下实验结果并提取关键信息用于保存到记忆系统：

实验数据：
{experiment_data}

请提取：
1. 关键技术栈和算法
2. 主要参数配置
3. 成功/失败的原因
4. 可复用的经验
"""


def build_memory_query_prompt(task_description: str, dataset_name: str) -> str:
    """构建记忆查询prompt（Stage 2-3使用）

    Args:
        task_description: 任务描述
        dataset_name: 数据集名称

    Returns:
        完整的prompt字符串
    """
    # TODO: Stage 2实现
    return f"""
请从历史实验记录中找到与以下任务相似的成功案例：

任务描述：{task_description}
数据集：{dataset_name}

请返回：
1. 最相似的5个成功案例
2. 每个案例的关键成功因素
3. 可以借鉴的具体技术方案
"""


def build_strategic_planning_prompt(
    task: str,
    l1_history: str,
    l2_similar: str,
    l4_meta: str
) -> str:
    """构建策略规划prompt（Stage 4使用）

    Args:
        task: 当前任务描述
        l1_history: L1记忆（直接历史）
        l2_similar: L2记忆（相似案例）
        l4_meta: L4记忆（元认知策略）

    Returns:
        完整的prompt字符串
    """
    # TODO: Stage 4实现
    return f"""
基于多层记忆系统，为以下任务制定最优策略：

当前任务：{task}

L1直接历史：
{l1_history}

L2相似案例：
{l2_similar}

L4元认知策略：
{l4_meta}

请制定：
1. 最优技术路线
2. 风险预判和应对
3. 资源分配策略
4. 预期成功率
"""
