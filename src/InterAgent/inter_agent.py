import os
import asyncio
import logging
import subprocess
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
from src.utils.openai_client import get_client
from src.InterAgent.coding_agent import CodingAgent
from src.mle_bench.evaluator_agent import MLEBenchEvaluator


class InterAgent:
    """主要的研究代理，负责协调整个研究任务流程"""

    def __init__(self, model_name: Optional[str] = None, device_info: Optional[Dict] = None):
        self.client = get_client(model_name)
        self.coding_agent = CodingAgent(model_name)
        self.logger = logging.getLogger(__name__)
        self.workspace_root = Path("workspaces")
        self.device_info = device_info or {'device_type': 'cpu', 'device_name': 'CPU'}

        # 初始化评估器字典
        self.evaluators = {
            'mlebench': MLEBenchEvaluator()
        }
    
    async def run_research_task(
        self,
        research_topic: str,
        research_goal: str,
        dataset_path: str,
        benchmark_name: str = "mlebench",
        dataset_name: str = "unknown",
        debug: bool = False
    ) -> Dict[str, str]:
        """
        主入口函数，执行研究任务
        
        Args:
            research_topic: 研究主题描述
            research_goal: 研究目标
            dataset_path: 数据集路径
            benchmark_name: benchmark名称
            dataset_name: 数据集名称
            debug: 是否开启调试模式
        
        Returns:
            Dict包含任务执行结果
        """
        # 创建工作目录
        workspace_dir = self._create_workspace(benchmark_name, dataset_name)
        
        self.logger.info("开始执行研究任务")
        self.logger.info(f"研究主题: {research_topic}")
        self.logger.info(f"研究目标: {research_goal}")
        self.logger.info(f"数据集路径: {dataset_path}")
        self.logger.info(f"工作目录: {workspace_dir}")
        
        try:
            # 首先探测文件结构
            structure_info = await self.file_structure_probing(dataset_path)
            self.logger.info(f"文件结构分析完成: {structure_info['status']}")
            
            # 生成数据加载代码
            data_loader_result = await self._generate_data_loader(
                structure_info, dataset_path, workspace_dir
            )
            self.logger.info(f"数据加载代码生成: {data_loader_result['status']}")
            
            # 执行数据加载测试（无论debug模式如何都会执行）
            test_result = await self._test_data_loader(workspace_dir)
            self.logger.info(f"数据加载测试: {test_result['status']}")
            
            # 如果数据加载失败，尝试debug修复
            if test_result['status'] == 'error' and test_result.get('detailed_error'):
                self.logger.info("数据加载失败，尝试自动修复...")
                debug_result = await self._debug_data_loader_with_retry(
                    workspace_dir, test_result['detailed_error']
                )
                if debug_result['status'] == 'success':
                    # 重新测试修复后的代码
                    test_result = await self._test_data_loader(workspace_dir)
                    self.logger.info(f"修复后数据加载测试: {test_result['status']}")
            
            # 初步分析：基于数据结构和研究任务进行分析
            init_analysis_result = None
            if test_result.get('data_structure_info'):
                init_analysis_result = await self.init_analyzing(
                    research_topic=research_topic,
                    research_goal=research_goal,
                    dataset_path=dataset_path,
                    data_structure_info=test_result['data_structure_info'],
                    structure_analysis=structure_info.get('analysis_result', {})
                )
                self.logger.info(f"初步分析: {init_analysis_result['status']}")
            
            # 生成研究代码：基于初步分析结果生成research.py
            research_code_result = None
            if init_analysis_result and init_analysis_result['status'] == 'success':
                # 获取提交文件名
                submission_file_name = self._get_submission_file_name(structure_info, benchmark_name)
                
                research_code_result = await self._generate_research_code(
                    research_topic=research_topic,
                    research_goal=research_goal,
                    init_analysis=init_analysis_result['analysis_content'],
                    data_structure_info=test_result['data_structure_info'],
                    submission_file_name=submission_file_name,
                    workspace_dir=workspace_dir
                )
                self.logger.info(f"研究代码生成: {research_code_result['status']}")
                
                # 无论生成成功与否，都要测试研究代码执行
                if research_code_result['status'] == 'success':
                    # 测试生成的研究代码
                    research_test_result = await self._test_research_code(workspace_dir)
                    self.logger.info(f"研究代码测试: {research_test_result['status']}")
                    
                    # 如果测试失败，启动debug循环
                    if research_test_result['status'] == 'error' and research_test_result.get('detailed_error'):
                        self.logger.info("研究代码执行失败，启动自动debug循环...")
                        max_retries = 3
                        retry_count = 0
                        install_attempts = 0  # 依赖安装尝试次数
                        max_install_attempts = 3

                        while retry_count < max_retries and research_test_result['status'] == 'error':
                            # 检查是否是ModuleNotFoundError
                            error_message = research_test_result.get('detailed_error', '')
                            is_module_error = 'ModuleNotFoundError' in error_message or 'No module named' in error_message

                            if is_module_error and install_attempts < max_install_attempts:
                                # 尝试安装缺失的包
                                install_attempts += 1
                                self.logger.info(f"检测到ModuleNotFoundError，尝试自动安装依赖 ({install_attempts}/{max_install_attempts})...")
                                self.logger.error("=" * 80)
                                self.logger.error("缺失依赖错误:")
                                self.logger.error(error_message)
                                self.logger.error("=" * 80)

                                # 调用CodingAgent安装依赖
                                install_result = await self.coding_agent.install_missing_package(
                                    error_message=error_message,
                                    project_root=str(self.workspace_root.parent)  # 项目根目录
                                )

                                if install_result['status'] == 'success':
                                    self.logger.info(f"成功安装: {install_result.get('package_name')}")
                                    # 直接重新测试，不计入debug次数
                                    research_test_result = await self._test_research_code(workspace_dir)
                                    self.logger.info(f"安装依赖后重新测试: {research_test_result['status']}")

                                    if research_test_result['status'] == 'success':
                                        self.logger.info("安装依赖后代码执行成功！")
                                        break
                                else:
                                    self.logger.error(f"依赖安装失败: {install_result.get('error')}")
                                    # 如果达到最大安装尝试次数，进入正常debug流程
                                    if install_attempts >= max_install_attempts:
                                        self.logger.error(f"已达到最大依赖安装尝试次数({max_install_attempts})，进入代码调试流程")

                            # 不是依赖问题或已经尝试安装多次，进入正常debug流程
                            if not is_module_error or install_attempts >= max_install_attempts:
                                retry_count += 1
                                self.logger.info(f"Debug尝试 {retry_count}/{max_retries}...")
                                # 打印详细的错误信息
                                self.logger.error("=" * 80)
                                self.logger.error("错误详情:")
                                self.logger.error(error_message)
                                self.logger.error("=" * 80)

                                debug_result = await self._debug_research_code_with_retry(
                                    workspace_dir, research_test_result['detailed_error']
                                )

                                if debug_result['status'] == 'success':
                                    # 重新测试修复后的代码
                                    research_test_result = await self._test_research_code(workspace_dir)
                                    self.logger.info(f"修复后研究代码测试: {research_test_result['status']}")

                                    if research_test_result['status'] == 'success':
                                        self.logger.info("研究代码debug成功，代码可正常运行")
                                        break
                                else:
                                    self.logger.error(f"Debug失败: {debug_result.get('message', '')}")

                        if research_test_result['status'] == 'error':
                            self.logger.error(f"经过{max_retries}次debug尝试和{install_attempts}次依赖安装后，研究代码仍然失败")

                    # 如果小样本测试成功，运行完整数据集
                    full_run_result = None
                    if research_test_result['status'] == 'success':
                        self.logger.info("=" * 80)
                        self.logger.info("小样本测试通过，开始运行完整数据集...")
                        self.logger.info("=" * 80)
                        full_run_result = await self._run_full_research_code(workspace_dir)
                        self.logger.info(f"完整数据集运行: {full_run_result['status']}")

                        if full_run_result['status'] == 'success':
                            self.logger.info("完整数据集训练成功！")
                            self.logger.info(f"生成的submission文件: {full_run_result.get('submission_files', [])}")

                            # 评估结果
                            evaluation_result = await self._evaluate_submission(
                                workspace_dir,
                                dataset_name,
                                benchmark_name
                            )
                            full_run_result['evaluation'] = evaluation_result

                            if evaluation_result['status'] == 'success':
                                self.logger.info("=" * 80)
                                self.logger.info("=== 评估结果 ===")
                                self.logger.info(f"分数: {evaluation_result.get('score', 'N/A')}")
                                self.logger.info(f"金牌: {'是' if evaluation_result.get('gold_medal') else '否'}")
                                self.logger.info(f"银牌: {'是' if evaluation_result.get('silver_medal') else '否'}")
                                self.logger.info(f"铜牌: {'是' if evaluation_result.get('bronze_medal') else '否'}")
                                self.logger.info(f"超过中位数: {'是' if evaluation_result.get('above_median') else '否'}")
                                self.logger.info(f"提交有效: {'是' if evaluation_result.get('valid_submission') else '否'}")
                                self.logger.info(evaluation_result.get('message', ''))
                                self.logger.info("=" * 80)
                            else:
                                self.logger.error("=" * 80)
                                self.logger.error("=== 评估失败 ===")
                                self.logger.error(f"错误: {evaluation_result.get('error_message', 'unknown')}")
                                self.logger.error(f"失败类型: {evaluation_result.get('failure_type', 'unknown')}")
                                self.logger.error("=" * 80)

                                # 智能重试逻辑
                                max_retry_attempts = 3  # 最多重试3次（包括重新生成和重新评估）
                                retry_count = 0
                                regeneration_count = 0

                                while retry_count < max_retry_attempts and evaluation_result['status'] == 'error':
                                    retry_count += 1
                                    requires_regeneration = evaluation_result.get('requires_code_regeneration', False)

                                    self.logger.info(f"DEBUG: evaluation_result keys: {evaluation_result.keys()}")
                                    self.logger.info(f"DEBUG: requires_code_regeneration = {requires_regeneration}")
                                    self.logger.info(f"DEBUG: failure_type = {evaluation_result.get('failure_type')}")

                                    if requires_regeneration:
                                        # 需要重新生成代码
                                        regeneration_count += 1
                                        self.logger.info("=" * 80)
                                        self.logger.info(f"=== 第 {regeneration_count} 次重新生成research.py（原因：{evaluation_result.get('failure_type')}）===")
                                        self.logger.info("=" * 80)

                                        # 完全重新生成research.py
                                        research_code_result = await self._generate_research_code(
                                            research_topic=research_topic,
                                            research_goal=research_goal,
                                            init_analysis=init_analysis_result['analysis_content'],
                                            data_structure_info=test_result['data_structure_info'],
                                            submission_file_name=submission_file_name,
                                            workspace_dir=workspace_dir
                                        )

                                        if research_code_result['status'] == 'success':
                                            # 测试新代码
                                            research_test_result = await self._test_research_code(workspace_dir)

                                            if research_test_result['status'] == 'success':
                                                # 运行完整数据集
                                                full_run_result = await self._run_full_research_code(workspace_dir)

                                                if full_run_result['status'] == 'success':
                                                    # 重新评估
                                                    evaluation_result = await self._evaluate_submission(
                                                        workspace_dir,
                                                        dataset_name,
                                                        benchmark_name
                                                    )
                                                    full_run_result['evaluation'] = evaluation_result

                                                    if evaluation_result['status'] == 'success':
                                                        self.logger.info("=" * 80)
                                                        self.logger.info(f"=== 第 {regeneration_count} 次重新生成后评估成功 ===")
                                                        self.logger.info(f"分数: {evaluation_result.get('score', 'N/A')}")
                                                        self.logger.info(evaluation_result.get('message', ''))
                                                        self.logger.info("=" * 80)
                                                        break
                                                    else:
                                                        self.logger.error(f"第 {regeneration_count} 次重新生成后评估仍然失败")
                                                else:
                                                    self.logger.error(f"第 {regeneration_count} 次重新生成的代码完整运行失败")
                                                    break
                                            else:
                                                self.logger.error(f"第 {regeneration_count} 次重新生成的代码测试失败")
                                                break
                                        else:
                                            self.logger.error(f"第 {regeneration_count} 次重新生成research.py失败")
                                            break
                                    else:
                                        # 不需要重新生成代码，只需重新评估（可能是临时问题或文件查找问题）
                                        self.logger.info("=" * 80)
                                        self.logger.info(f"=== 第 {retry_count} 次重新评估（不重新生成代码）===")
                                        self.logger.info("=" * 80)

                                        evaluation_result = await self._evaluate_submission(
                                            workspace_dir,
                                            dataset_name,
                                            benchmark_name
                                        )
                                        full_run_result['evaluation'] = evaluation_result

                                        if evaluation_result['status'] == 'success':
                                            self.logger.info("=" * 80)
                                            self.logger.info(f"=== 重新评估成功 ===")
                                            self.logger.info(f"分数: {evaluation_result.get('score', 'N/A')}")
                                            self.logger.info(evaluation_result.get('message', ''))
                                            self.logger.info("=" * 80)
                                            break
                                        else:
                                            self.logger.error(f"重新评估仍然失败")
                                            # 如果重新评估还是失败且仍然不需要重新生成，可能是其他问题
                                            if not evaluation_result.get('requires_code_regeneration', False):
                                                self.logger.error("评估器持续失败且不需要重新生成代码，可能是配置或环境问题")
                                                break

                                if retry_count >= max_retry_attempts and evaluation_result['status'] == 'error':
                                    self.logger.error(f"已达到最大重试次数 ({max_retry_attempts})，评估仍然失败")

                                # 更新research_code_result以包含重试信息
                                research_code_result['regeneration_count'] = regeneration_count
                                research_code_result['total_retry_count'] = retry_count
                        else:
                            # 完整数据集运行失败，启动重试循环
                            self.logger.error(f"完整数据集运行失败: {full_run_result.get('message', '')}")

                            max_full_run_retries = 3
                            full_run_retry_count = 0
                            full_regeneration_count = 0

                            while full_run_retry_count < max_full_run_retries and full_run_result['status'] == 'error':
                                full_run_retry_count += 1
                                full_regeneration_count += 1

                                # 分析失败原因
                                failure_message = full_run_result.get('message', '')
                                error_output = full_run_result.get('stderr', '') + '\n' + full_run_result.get('stdout', '')

                                # 构建优化提示
                                optimization_hint = ""
                                if '超时' in failure_message or 'timeout' in failure_message.lower():
                                    optimization_hint = """
代码执行超时（2小时限制）。请优化代码以提高运行速度：
1. 减少特征数量（如TF-IDF的max_features从5000降到1000-2000）
2. 使用更简单的模型（如LogisticRegression而非复杂的集成模型）
3. 减少训练数据量（如使用stratified sampling取子集）
4. 移除耗时的特征工程步骤
5. 避免使用循环遍历数据，使用向量化操作
6. 确保使用n_jobs=-1等并行参数加速训练
"""
                                elif 'memory' in failure_message.lower() or 'oom' in failure_message.lower():
                                    optimization_hint = """
内存不足错误。请优化内存使用：
1. 减少TF-IDF特征数量（max_features降低到1000以下）
2. 使用sparse matrix而非dense matrix
3. 分批处理数据而非一次性加载
4. 使用内存高效的模型（LinearSVC, SGDClassifier等）
5. 及时释放不需要的中间变量
"""
                                else:
                                    optimization_hint = f"""
完整数据集运行失败，错误信息：
{error_output[-1000:] if len(error_output) > 1000 else error_output}

请分析错误并优化代码，确保能在2小时内完成训练。
"""

                                self.logger.info("=" * 80)
                                self.logger.info(f"=== 第 {full_regeneration_count} 次重新生成research.py（完整数据集运行失败）===")
                                self.logger.info(f"失败原因: {failure_message}")
                                self.logger.info("=" * 80)

                                # 重新生成代码，带上优化提示
                                optimized_init_analysis = init_analysis_result['analysis_content'] + "\n\n**重要优化要求**：" + optimization_hint

                                research_code_result = await self._generate_research_code(
                                    research_topic=research_topic,
                                    research_goal=research_goal,
                                    init_analysis=optimized_init_analysis,
                                    data_structure_info=test_result['data_structure_info'],
                                    submission_file_name=submission_file_name,
                                    workspace_dir=workspace_dir
                                )

                                if research_code_result['status'] == 'success':
                                    # 测试新代码
                                    research_test_result = await self._test_research_code(workspace_dir)

                                    if research_test_result['status'] == 'success':
                                        # 重新运行完整数据集
                                        full_run_result = await self._run_full_research_code(workspace_dir)

                                        if full_run_result['status'] == 'success':
                                            self.logger.info("=" * 80)
                                            self.logger.info(f"=== 第 {full_regeneration_count} 次优化后完整运行成功 ===")
                                            self.logger.info("=" * 80)

                                            # 评估结果
                                            evaluation_result = await self._evaluate_submission(
                                                workspace_dir,
                                                dataset_name,
                                                benchmark_name
                                            )
                                            full_run_result['evaluation'] = evaluation_result

                                            if evaluation_result['status'] == 'success':
                                                self.logger.info("=" * 80)
                                                self.logger.info("=== 评估结果 ===")
                                                self.logger.info(f"分数: {evaluation_result.get('score', 'N/A')}")
                                                self.logger.info(f"金牌: {'是' if evaluation_result.get('gold_medal') else '否'}")
                                                self.logger.info(f"银牌: {'是' if evaluation_result.get('silver_medal') else '否'}")
                                                self.logger.info(f"铜牌: {'是' if evaluation_result.get('bronze_medal') else '否'}")
                                                self.logger.info(f"超过中位数: {'是' if evaluation_result.get('above_median') else '否'}")
                                                self.logger.info(f"提交有效: {'是' if evaluation_result.get('valid_submission') else '否'}")
                                                self.logger.info(evaluation_result.get('message', ''))
                                                self.logger.info("=" * 80)
                                            break
                                        else:
                                            self.logger.error(f"第 {full_regeneration_count} 次优化后完整运行仍然失败: {full_run_result.get('message', '')}")
                                    else:
                                        self.logger.error(f"第 {full_regeneration_count} 次优化后的代码测试失败")
                                        # 测试失败也要继续重试，不要break
                                else:
                                    self.logger.error(f"第 {full_regeneration_count} 次重新生成research.py失败")
                                    break

                            if full_run_retry_count >= max_full_run_retries and full_run_result['status'] == 'error':
                                self.logger.error(f"已达到最大完整运行重试次数 ({max_full_run_retries})，完整运行仍然失败")

                    # 更新research_code_result以包含测试结果和完整运行结果
                    research_code_result['test_result'] = research_test_result
                    research_code_result['full_run_result'] = full_run_result
                    if research_test_result['status'] == 'success':
                        if full_run_result and full_run_result['status'] == 'success':
                            research_code_result['status'] = 'success'
                            research_code_result['message'] = '研究代码生成、测试和完整运行全部成功'
                        else:
                            research_code_result['status'] = 'partial_success'
                            research_code_result['message'] = '研究代码测试成功但完整运行失败'
                    else:
                        research_code_result['status'] = 'error'
                        research_code_result['message'] = '研究代码生成成功但执行失败'
            
            return {
                "status": "success",
                "workspace_dir": str(workspace_dir),
                "research_topic": research_topic,
                "research_goal": research_goal,
                "dataset_path": dataset_path,
                "structure_info": structure_info,
                "data_loader_result": data_loader_result,
                "test_result": test_result,  # 替换debug_result
                "init_analysis_result": init_analysis_result,  # 新增初步分析结果
                "research_code_result": research_code_result,  # 新增研究代码结果
                "message": "研究任务启动成功"
            }
            
        except Exception as e:
            self.logger.error(f"研究任务执行失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "message": f"研究任务执行失败: {str(e)}"
            }
    
    async def file_structure_probing(self, dataset_path: str) -> Dict[str, str]:
        """
        文件结构探测，分析数据集结构并判断文件加载需求
        
        Args:
            dataset_path: 数据集路径
        
        Returns:
            Dict包含结构分析结果
        """
        self.logger.info(f"开始探测文件结构: {dataset_path}")
        
        try:
            # 获取数据集目录结构信息
            structure_summary = self._get_directory_structure(dataset_path)
            
            # 构建分析prompt
            prompt = self._build_structure_analysis_prompt(structure_summary)
            
            # 调用LLM分析文件结构
            messages = [{"role": "user", "content": prompt}]
            response = await self.client.chat(messages, max_tokens=2000)
            
            # 调试：记录LLM响应
            self.logger.info(f"LLM文件结构分析响应: {response}")
            
            # 解析LLM响应
            analysis_result = self._parse_structure_analysis(response)
            
            self.logger.info("文件结构探测完成")
            
            return {
                "status": "success",
                "dataset_path": dataset_path,
                "structure_summary": structure_summary,
                "analysis_result": analysis_result,
                "message": "文件结构分析完成"
            }
            
        except Exception as e:
            self.logger.error(f"文件结构探测失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "message": f"文件结构探测失败: {str(e)}"
            }
    
    def _get_directory_structure(self, dataset_path: str, max_files_per_dir: int = 10) -> str:
        """
        获取目录结构摘要，递归遍历所有目录层级，显示完整路径帮助LLM理解文件格式
        
        Args:
            dataset_path: 数据集路径
            max_files_per_dir: 每个目录最大文件数限制
        
        Returns:
            目录结构的字符串描述
        """
        path = Path(dataset_path)
        if not path.exists():
            return f"路径不存在: {dataset_path}"
        
        structure_lines = []
        structure_lines.append(f"数据集根目录: {dataset_path}")
        structure_lines.append("")
        structure_lines.append("文件结构分析:")
        
        def _get_relative_path(file_path: Path) -> str:
            """获取相对于数据集根目录的路径"""
            try:
                return f"./{file_path.relative_to(path)}"
            except ValueError:
                return str(file_path)
        
        def _explore_directory(dir_path: Path, level: int = 0, max_level: int = 4):
            """递归探索目录结构"""
            if level > max_level:  # 限制最大层级防止过深
                return
            
            try:
                items = list(dir_path.iterdir())
                
                # 分离文件和目录
                files = [item for item in items if item.is_file()]
                dirs = [item for item in items if item.is_dir()]
                
                # 显示文件（限制数量），包含完整路径
                for i, file_item in enumerate(files[:max_files_per_dir]):
                    file_size = file_item.stat().st_size
                    relative_path = _get_relative_path(file_item)
                    structure_lines.append(f"{'  ' * level}📄 {relative_path} ({self._format_file_size(file_size)})")
                    
                    # 如果是特殊文件格式，显示文件内容示例
                    if file_item.suffix.lower() in ['.xyz', '.csv', '.xlsx', '.json', '.jsonl', '.txt', '.md', '.tsv']:
                        try:
                            with open(file_item, 'r', encoding='utf-8', errors='ignore') as f:
                                first_lines = f.readlines()[:3]
                            structure_lines.append(f"{'  ' * level}    [内容示例]:")
                            for line in first_lines:
                                structure_lines.append(f"{'  ' * level}    {line.strip()}")
                            structure_lines.append(f"{'  ' * level}    ...")
                        except:
                            pass
                
                if len(files) > max_files_per_dir:
                    remaining_files = len(files) - max_files_per_dir
                    structure_lines.append(f"{'  ' * level}    ... 还有{remaining_files}个文件")
                
                # 显示目录（限制数量），包含完整路径
                for i, dir_item in enumerate(dirs[:max_files_per_dir]):
                    relative_path = _get_relative_path(dir_item)
                    structure_lines.append(f"{'  ' * level}📁 {relative_path}/")
                    
                    # 深入探索目录
                    if i < 5:  # 探索前5个目录
                        _explore_directory(dir_item, level + 1, max_level)
                    elif i == 5 and len(dirs) > 5:
                        # 对于第6个目录，显示其内容作为示例
                        try:
                            sub_items = list(dir_item.iterdir())
                            structure_lines.append(f"{'  ' * (level+1)}[示例目录内容]:")
                            for j, sample_item in enumerate(sub_items[:3]):
                                if sample_item.is_file():
                                    file_size = sample_item.stat().st_size
                                    sample_relative_path = _get_relative_path(sample_item)
                                    structure_lines.append(f"{'  ' * (level+1)}📄 {sample_relative_path} ({self._format_file_size(file_size)})")
                                    
                                    # 显示文件内容示例
                                    if sample_item.suffix.lower() in ['.xyz', '.csv', '.xlsx', '.json', '.jsonl', '.txt', '.tsv']:
                                        try:
                                            with open(sample_item, 'r', encoding='utf-8', errors='ignore') as f:
                                                first_lines = f.readlines()[:3]
                                            structure_lines.append(f"{'  ' * (level+1)}    [内容示例]:")
                                            for line in first_lines:
                                                structure_lines.append(f"{'  ' * (level+1)}    {line.strip()}")
                                        except:
                                            pass
                                else:
                                    sample_relative_path = _get_relative_path(sample_item)
                                    structure_lines.append(f"{'  ' * (level+1)}📁 {sample_relative_path}/")
                            if len(sub_items) > 3:
                                structure_lines.append(f"{'  ' * (level+1)}    ... 等共{len(sub_items)}个项目")
                        except:
                            pass
                
                if len(dirs) > max_files_per_dir:
                    remaining_dirs = len(dirs) - max_files_per_dir
                    structure_lines.append(f"{'  ' * level}    ... 还有{remaining_dirs}个目录")
                    
            except Exception as e:
                structure_lines.append(f"{'  ' * level}(无法读取目录内容: {str(e)})")
        
        try:
            _explore_directory(path)
        except Exception as e:
            structure_lines.append(f"读取目录失败: {str(e)}")
        
        return "\n".join(structure_lines)
    
    def _format_file_size(self, size_bytes: int) -> str:
        """格式化文件大小显示"""
        if size_bytes < 1024:
            return f"{size_bytes}B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes/1024:.1f}KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes/(1024*1024):.1f}MB"
        else:
            return f"{size_bytes/(1024*1024*1024):.1f}GB"
    
    def _build_structure_analysis_prompt(self, structure_summary: str) -> str:
        """构建文件结构分析的prompt"""
        prompt = f"""
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
        return prompt
    
    def _parse_structure_analysis(self, response: str) -> Dict[str, str]:
        """解析LLM的文件结构分析响应"""
        analysis = {
            "train_files": "",
            "test_files": "",
            "validation_files": "",
            "additional_files": "",
            "data_format": "",
            "loading_suggestions": ""
        }
        
        lines = response.strip().split('\n')
        current_key = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            
            # 处理markdown格式的标题
            if "训练集文件" in line and (":" in line or "**" in line):
                if current_key and current_content:
                    analysis[current_key] = "\n".join(current_content).strip()
                current_key = "train_files"
                current_content = []
                # 提取冒号后的内容
                if ":" in line:
                    content = line.split(":", 1)[-1].strip()
                    if content:
                        current_content.append(content)
                        
            elif "测试集文件" in line and (":" in line or "**" in line):
                if current_key and current_content:
                    analysis[current_key] = "\n".join(current_content).strip()
                current_key = "test_files"
                current_content = []
                if ":" in line:
                    content = line.split(":", 1)[-1].strip()
                    if content:
                        current_content.append(content)
                        
            elif "验证集文件" in line and (":" in line or "**" in line):
                if current_key and current_content:
                    analysis[current_key] = "\n".join(current_content).strip()
                current_key = "validation_files"
                current_content = []
                if ":" in line:
                    content = line.split(":", 1)[-1].strip()
                    if content:
                        current_content.append(content)
                        
            elif "额外文件" in line and (":" in line or "**" in line):
                if current_key and current_content:
                    analysis[current_key] = "\n".join(current_content).strip()
                current_key = "additional_files"
                current_content = []
                if ":" in line:
                    content = line.split(":", 1)[-1].strip()
                    if content:
                        current_content.append(content)
                        
            elif "数据格式" in line and (":" in line or "**" in line):
                if current_key and current_content:
                    analysis[current_key] = "\n".join(current_content).strip()
                current_key = "data_format"
                current_content = []
                if ":" in line:
                    content = line.split(":", 1)[-1].strip()
                    if content:
                        current_content.append(content)
                        
            elif "加载建议" in line and (":" in line or "**" in line):
                if current_key and current_content:
                    analysis[current_key] = "\n".join(current_content).strip()
                current_key = "loading_suggestions"
                current_content = []
                if ":" in line:
                    content = line.split(":", 1)[-1].strip()
                    if content:
                        current_content.append(content)
                        
            elif current_key and line and not line.startswith("**") and line != "":
                # 添加内容行到当前字段
                current_content.append(line)
        
        # 处理最后一个字段
        if current_key and current_content:
            analysis[current_key] = "\n".join(current_content).strip()
        
        return analysis
    
    def _extract_python_code(self, response: str) -> str:
        """从LLM响应中提取Python代码"""
        lines = response.split('\n')
        code_lines = []
        in_code_block = False
        
        for line in lines:
            if line.strip().startswith('```python') or line.strip().startswith('```Python'):
                in_code_block = True
                continue
            elif line.strip().startswith('```') and in_code_block:
                break
            elif in_code_block:
                code_lines.append(line)
        
        if not code_lines:
            # 如果没有找到代码块，尝试查找其他格式
            for line in lines:
                if line.strip().startswith('```'):
                    in_code_block = not in_code_block
                    continue
                elif in_code_block:
                    code_lines.append(line)
        
        if not code_lines:
            # 如果还是没找到，返回整个响应的可能代码部分
            self.logger.warning("未找到标准代码块，尝试提取整个响应")
            return response.strip()
        
        return '\n'.join(code_lines)
    
    async def _debug_data_loader_with_retry(self, workspace_dir: Path, error_message: str) -> Dict[str, str]:
        """
        使用CodingAgent调试修复数据加载代码
        
        Args:
            workspace_dir: 工作目录
            error_message: 错误信息
            
        Returns:
            调试结果
        """
        try:
            loader_file = workspace_dir / "load_research_data.py"
            if not loader_file.exists():
                return {
                    "status": "error",
                    "message": "数据加载文件不存在"
                }
            
            self.logger.info("开始调试数据加载代码...")
            
            # 调用CodingAgent的debug_code功能
            debug_result = await self.coding_agent.debug_code(
                code_path=str(loader_file),
                error_message=error_message,
                error_context="数据加载过程中出现错误，可能需要处理数据类型、缺失值或文件路径问题"
            )
            
            if debug_result['status'] == 'success':
                self.logger.info("数据加载代码调试修复成功")
                return {
                    "status": "success",
                    "message": "代码修复成功"
                }
            else:
                self.logger.error(f"数据加载代码调试失败: {debug_result.get('error', '')}")
                return {
                    "status": "error",
                    "message": f"代码调试失败: {debug_result.get('error', '')}"
                }
                
        except Exception as e:
            self.logger.error(f"调试过程出错: {str(e)}")
            return {
                "status": "error",
                "message": f"调试过程出错: {str(e)}"
            }
    
    def _get_submission_file_name(self, structure_info: Dict[str, str], benchmark_name: str = "mlebench") -> str:
        """
        从文件结构信息中获取提交文件名

        Args:
            structure_info: 文件结构信息
            benchmark_name: benchmark名称

        Returns:
            提交文件名
        """
        structure_summary = structure_info.get('structure_summary', '')
        analysis_result = structure_info.get('analysis_result', {})

        # 从结构摘要中查找submission相关文件
        if 'sample_submission' in structure_summary.lower() or 'submission' in structure_summary.lower():
            lines = structure_summary.split('\n')
            for line in lines:
                if 'submission' in line.lower():
                    # 支持多种格式
                    for ext in ['.csv', '.xlsx', '.json', '.jsonl', '.txt']:
                        if ext in line.lower():
                            parts = line.split()
                            for part in parts:
                                if 'submission' in part.lower() and ext in part.lower():
                                    return part.split('/')[-1]  # 只要文件名部分

        # 从额外文件中查找
        additional_files = analysis_result.get('additional_files', '')
        if 'submission' in additional_files.lower():
            # 检查多种格式
            for ext in ['.csv', '.xlsx', '.json', '.jsonl', '.txt']:
                if f'submission{ext}' in additional_files.lower():
                    # 提取完整文件名
                    import re
                    pattern = rf'\S*submission\S*{ext}'
                    match = re.search(pattern, additional_files.lower())
                    if match:
                        return match.group(0)

        # 根据benchmark类型返回默认提交文件名
        benchmark_submission_files = {
            "mlebench": "sample_submission.csv",
            "kaggle": "submission.csv",
            "default": "submission.csv"
        }

        return benchmark_submission_files.get(benchmark_name.lower(), benchmark_submission_files["default"])
    
    async def _generate_research_code(
        self,
        research_topic: str,
        research_goal: str,
        init_analysis: str,
        data_structure_info: str,
        submission_file_name: str,
        workspace_dir: Path
    ) -> Dict[str, str]:
        """
        生成研究代码
        
        Args:
            research_topic: 研究主题
            research_goal: 研究目标
            init_analysis: 初步分析结果
            data_structure_info: 数据结构信息
            submission_file_name: 提交文件名
            workspace_dir: 工作目录
            
        Returns:
            生成结果
        """
        try:
            result = await self.coding_agent.generate_research_code(
                research_topic=research_topic,
                research_goal=research_goal,
                init_analysis=init_analysis,
                data_structure_info=data_structure_info,
                submission_file_name=submission_file_name,
                workspace_dir=str(workspace_dir),
                device_info=self.device_info
            )
            return result
            
        except Exception as e:
            self.logger.error(f"生成研究代码时出错: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "message": f"生成研究代码失败: {str(e)}"
            }
    
    def _find_and_validate_output_files(self, workspace_dir: Path, benchmark_name: str = "mlebench") -> tuple:
        """
        查找并验证输出文件

        Args:
            workspace_dir: 工作目录
            benchmark_name: benchmark名称

        Returns:
            (valid_files, all_files): 有效文件列表和所有找到的文件列表
        """
        import pandas as pd

        # 根据benchmark类型决定检查哪些格式
        benchmark_file_patterns = {
            "mlebench": ["*.csv"],  # mlebench主要使用CSV格式
            "kaggle": ["*.csv"],
            "default": ["*.csv", "*.xlsx", "*.json", "*.jsonl", "*.txt"]  # 其他benchmark支持更多格式
        }

        file_patterns = benchmark_file_patterns.get(benchmark_name.lower(), benchmark_file_patterns["default"])

        all_files = []
        for pattern in file_patterns:
            all_files.extend(workspace_dir.glob(pattern))

        valid_files = []
        for file_path in all_files:
            try:
                # 根据文件类型验证
                if file_path.suffix.lower() == '.csv':
                    df = pd.read_csv(file_path)
                    if len(df) > 0 and len(df.columns) >= 1:
                        valid_files.append(file_path)
                        self.logger.info(f"有效文件: {file_path.name}, 形状: {df.shape}")
                    else:
                        self.logger.warning(f"文件为空或无效: {file_path.name}")
                elif file_path.suffix.lower() in ['.json', '.jsonl']:
                    # JSON文件基本检查
                    if file_path.stat().st_size > 0:
                        valid_files.append(file_path)
                        self.logger.info(f"有效文件: {file_path.name}, 大小: {file_path.stat().st_size}")
                elif file_path.suffix.lower() in ['.txt', '.xlsx']:
                    # 其他格式只检查是否非空
                    if file_path.stat().st_size > 0:
                        valid_files.append(file_path)
                        self.logger.info(f"有效文件: {file_path.name}, 大小: {file_path.stat().st_size}")
            except Exception as e:
                self.logger.warning(f"无法读取文件 {file_path.name}: {e}")

        return valid_files, all_files

    async def _test_research_code(self, workspace_dir: Path, benchmark_name: str = "mlebench") -> Dict[str, str]:
        """
        测试研究代码

        Args:
            workspace_dir: 工作目录
            benchmark_name: benchmark名称

        Returns:
            测试结果
        """
        try:
            research_file = workspace_dir / "research.py"
            if not research_file.exists():
                return {
                    "status": "error",
                    "error": "研究代码文件不存在",
                    "message": "research.py文件未找到",
                    "detailed_error": "research.py文件未找到"
                }
            
            self.logger.info("执行研究代码测试（小样本快速验证模式）...")
            
            # 执行Python脚本
            result = subprocess.run(
                ["python", "-u", "research.py"],
                cwd=str(workspace_dir),
                capture_output=True,
                text=True,
                timeout=600,  # 10分钟超时
                env={**os.environ, "PYTHONUNBUFFERED": "1", "USE_SMALL_SAMPLE": "1", "QUICK_TEST": "1"}
            )
            
            # 合并输出
            full_output = ""
            if result.stdout:
                full_output += result.stdout
            if result.stderr:
                full_output += "\n=== STDERR ===\n" + result.stderr
            
            if result.returncode == 0:
                # 检查是否成功生成了有效的.csv文件
                submission_files = list(workspace_dir.glob("*.csv"))
                
                if submission_files:
                    # 验证submission文件的有效性
                    valid_files = []
                    for csv_file in submission_files:
                        try:
                            import pandas as pd
                            df = pd.read_csv(csv_file)
                            # 检查文件不是空的，且有实际数据行
                            if len(df) > 0 and not df.empty:
                                # 检查是否至少有1列数据
                                if len(df.columns) >= 1:
                                    valid_files.append(csv_file)
                                    self.logger.info(f"有效submission文件: {csv_file.name}, 形状: {df.shape}")
                                else:
                                    self.logger.warning(f"submission文件列数不足: {csv_file.name}, 列数: {len(df.columns)}")
                            else:
                                self.logger.warning(f"submission文件为空: {csv_file.name}")
                        except Exception as e:
                            self.logger.warning(f"无法读取submission文件 {csv_file.name}: {e}")
                    
                    if valid_files:
                        self.logger.info(f"研究代码执行成功，生成了有效submission文件: {[str(f.name) for f in valid_files]}")
                        return {
                            "status": "success",
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                            "output_info": full_output,
                            "submission_files": [str(f) for f in valid_files],
                            "message": "研究代码执行完成并生成了有效submission文件"
                        }
                    else:
                        self.logger.warning("生成的.csv文件无效（空文件或格式错误）")
                        error_details = f"Exit code: {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
                        return {
                            "status": "error",
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                            "returncode": result.returncode,
                            "output_info": full_output,
                            "detailed_error": f"生成的.csv文件无效（空文件或格式错误）。完整输出:\n{error_details}",
                            "message": "研究代码执行失败：生成的submission文件无效"
                        }
                else:
                    self.logger.warning("研究代码执行完成但未生成.csv文件")
                    # 将完整输出作为错误信息，方便debug
                    error_details = f"Exit code: {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
                    return {
                        "status": "error",
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "returncode": result.returncode,
                        "output_info": full_output,
                        "detailed_error": f"代码执行完成但未生成.csv文件。完整输出:\n{error_details}",
                        "message": "研究代码执行失败：未生成.csv文件"
                    }
            else:
                self.logger.error(f"研究代码执行失败，退出码: {result.returncode}")
                # 返回完整的错误信息，包括stdout和stderr
                error_details = f"Exit code: {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
                return {
                    "status": "error",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "output_info": full_output,
                    "detailed_error": error_details,
                    "message": "研究代码执行失败"
                }
                
        except subprocess.TimeoutExpired:
            self.logger.error("研究代码执行超时")
            return {
                "status": "error",
                "error": "执行超时",
                "message": "研究代码执行超时",
                "detailed_error": "执行超时"
            }
        except Exception as e:
            self.logger.error(f"研究代码测试出错: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "message": f"研究代码测试失败: {str(e)}",
                "detailed_error": str(e)
            }

    async def _run_full_research_code(self, workspace_dir: Path) -> Dict[str, str]:
        """
        运行完整数据集的研究代码

        Args:
            workspace_dir: 工作目录

        Returns:
            运行结果
        """
        try:
            research_file = workspace_dir / "research.py"
            if not research_file.exists():
                return {
                    "status": "error",
                    "error": "研究代码文件不存在",
                    "message": "research.py文件未找到"
                }

            self.logger.info("执行完整数据集训练（移除环境变量限制）...")

            # 执行Python脚本，不设置USE_SMALL_SAMPLE和QUICK_TEST
            env = os.environ.copy()
            env.pop("USE_SMALL_SAMPLE", None)
            env.pop("QUICK_TEST", None)
            env["PYTHONUNBUFFERED"] = "1"

            result = subprocess.run(
                ["python", "-u", "research.py"],
                cwd=str(workspace_dir),
                capture_output=True,
                text=True,
                timeout=7200,  # 2小时超时（完整数据集可能需要更长时间）
                env=env
            )

            # 合并输出
            full_output = ""
            if result.stdout:
                full_output += result.stdout
            if result.stderr:
                full_output += "\n=== STDERR ===\n" + result.stderr

            if result.returncode == 0:
                # 检查是否成功生成了有效的.csv文件
                submission_files = list(workspace_dir.glob("*.csv"))

                if submission_files:
                    # 验证submission文件的有效性
                    valid_files = []
                    for csv_file in submission_files:
                        try:
                            import pandas as pd
                            df = pd.read_csv(csv_file)
                            # 检查文件不是空的，且有实际数据行
                            if len(df) > 0 and not df.empty:
                                # 检查是否有合理的列
                                if len(df.columns) >= 1:
                                    valid_files.append(csv_file)
                                    self.logger.info(f"完整运行生成有效submission: {csv_file.name}, 形状: {df.shape}")
                                else:
                                    self.logger.warning(f"submission文件列数不足: {csv_file.name}")
                            else:
                                self.logger.warning(f"submission文件为空: {csv_file.name}")
                        except Exception as e:
                            self.logger.warning(f"无法读取submission文件 {csv_file.name}: {e}")

                    if valid_files:
                        self.logger.info(f"完整数据集训练成功，生成submission: {[str(f.name) for f in valid_files]}")
                        return {
                            "status": "success",
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                            "submission_files": [str(f.name) for f in valid_files],
                            "message": "完整数据集训练成功"
                        }
                    else:
                        self.logger.warning("完整运行生成的.csv文件无效")
                        return {
                            "status": "error",
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                            "returncode": result.returncode,
                            "message": "完整运行生成的submission文件无效"
                        }
                else:
                    self.logger.warning("完整运行未生成.csv文件")
                    return {
                        "status": "error",
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "returncode": result.returncode,
                        "message": "完整运行未生成.csv文件"
                    }
            else:
                self.logger.error(f"完整数据集运行失败，退出码: {result.returncode}")
                return {
                    "status": "error",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "message": f"完整数据集运行失败，退出码: {result.returncode}"
                }

        except subprocess.TimeoutExpired:
            self.logger.error("完整数据集运行超时（2小时）")
            return {
                "status": "error",
                "error": "执行超时",
                "message": "完整数据集运行超时"
            }
        except Exception as e:
            self.logger.error(f"完整数据集运行出错: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "message": f"完整数据集运行失败: {str(e)}"
            }

    async def _debug_research_code_with_retry(self, workspace_dir: Path, error_message: str) -> Dict[str, str]:
        """
        使用CodingAgent调试修复研究代码
        
        Args:
            workspace_dir: 工作目录
            error_message: 错误信息
            
        Returns:
            调试结果
        """
        try:
            research_file = workspace_dir / "research.py"
            if not research_file.exists():
                return {
                    "status": "error",
                    "message": "研究代码文件不存在"
                }
            
            self.logger.info("开始调试研究代码...")
            
            # 调用CodingAgent的debug_code功能
            debug_result = await self.coding_agent.debug_code(
                code_path=str(research_file),
                error_message=error_message,
                error_context="研究代码执行过程中出现错误，可能需要处理模型训练、数据处理或文件输出问题"
            )
            
            if debug_result['status'] == 'success':
                self.logger.info("研究代码调试修复成功")
                return {
                    "status": "success",
                    "message": "代码修复成功"
                }
            else:
                self.logger.error(f"研究代码调试失败: {debug_result.get('error', '')}")
                return {
                    "status": "error",
                    "message": f"代码调试失败: {debug_result.get('error', '')}"
                }
                
        except Exception as e:
            self.logger.error(f"调试过程出错: {str(e)}")
            return {
                "status": "error",
                "message": f"调试过程出错: {str(e)}"
            }

    async def _evaluate_submission(
        self,
        workspace_dir: Path,
        dataset_name: str,
        benchmark_name: str
    ) -> Dict:
        """
        评估提交结果

        Args:
            workspace_dir: 工作目录
            dataset_name: 数据集名称
            benchmark_name: benchmark名称

        Returns:
            评估结果
        """
        try:
            self.logger.info(f"开始评估提交结果: benchmark={benchmark_name}, dataset={dataset_name}")

            # 获取对应的评估器
            evaluator = self.evaluators.get(benchmark_name.lower())

            if not evaluator:
                self.logger.warning(f"未找到benchmark '{benchmark_name}' 的评估器，跳过评估")
                return {
                    "status": "skipped",
                    "message": f"未找到benchmark '{benchmark_name}' 的评估器",
                    "valid_submission": None
                }

            # 调用评估器
            result = await evaluator.evaluate(workspace_dir, dataset_name, benchmark_name)

            return result

        except Exception as e:
            self.logger.error(f"评估过程出错: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "message": f"评估失败: {str(e)}",
                "valid_submission": False
            }

    async def init_analyzing(
        self,
        research_topic: str,
        research_goal: str,
        dataset_path: str,
        data_structure_info: str,
        structure_analysis: Dict[str, str]
    ) -> Dict[str, str]:
        """
        初步分析：基于数据结构和研究任务进行初步分析
        
        Args:
            research_topic: 研究主题
            research_goal: 研究目标
            dataset_path: 数据集路径
            data_structure_info: 数据结构信息（从data loader输出获取）
            structure_analysis: 文件结构分析结果
        
        Returns:
            初步分析结果
        """
        self.logger.info("开始进行初步数据和任务分析...")
        
        try:
            # 构建初步分析prompt
            prompt = f"""
            作为一名经验丰富的机器学习研究员，你需要基于数据结构和任务信息，给出技术路线分析和实现方案建议。

            **研究主题**:
            {research_topic}

            **研究目标**:
            {research_goal}

            **数据加载测试结果**:
            {data_structure_info}

            **文件结构分析**:
            训练集文件: {structure_analysis.get('train_files', '')}
            测试集文件: {structure_analysis.get('test_files', '')}
            数据格式: {structure_analysis.get('data_format', '')}

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

            # 调用LLM进行初步分析
            messages = [{"role": "user", "content": prompt}]
            response = await self.client.chat(messages, max_tokens=3000)
            
            self.logger.info("=== 初步分析结果 ===")
            self.logger.info(response)
            
            return {
                "status": "success",
                "analysis_content": response,
                "message": "初步分析完成"
            }
            
        except Exception as e:
            self.logger.error(f"初步分析失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "message": f"初步分析失败: {str(e)}"
            }
    
    def _create_workspace(self, benchmark_name: str, dataset_name: str) -> Path:
        """创建工作目录"""
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        exp_id = f"exp_id_{timestamp}"
        
        # 创建工作目录路径
        workspace_dir = self.workspace_root / benchmark_name / dataset_name / exp_id
        workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"创建工作目录: {workspace_dir}")
        return workspace_dir
    
    async def _generate_data_loader(
        self, 
        structure_info: Dict[str, str], 
        dataset_path: str, 
        workspace_dir: Path
    ) -> Dict[str, str]:
        """生成数据加载代码"""
        try:
            # 构建数据加载代码生成需求
            analysis_result = structure_info.get('analysis_result', {})
            structure_summary = structure_info.get('structure_summary', '')
            
            requirements = f"""
            **重要：只能生成一个文件load_research_data.py，不要生成任何其他文件！**
            
            创建一个数据加载脚本 load_research_data.py，要求：

            数据集路径: {dataset_path}
            
            **小样本支持**: 脚本需要支持环境变量USE_SMALL_SAMPLE=1时只加载少量数据用于快速测试
            
            文件结构分析（完整路径示例）:
            {structure_summary}
            
            训练集文件: {analysis_result.get('train_files', '')}
            测试集文件: {analysis_result.get('test_files', '')}
            数据格式: {analysis_result.get('data_format', '')}
            加载建议: {analysis_result.get('loading_suggestions', '')}

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
            
            # 直接调用LLM生成代码，不通过CodingAgent的项目结构
            messages = [{"role": "user", "content": requirements}]
            response = await self.client.chat(messages, max_tokens=4000)
            
            # 提取Python代码
            python_code = self._extract_python_code(response)
            
            # 直接保存到workspace_dir
            target_file = workspace_dir / "load_research_data.py"
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(python_code)
            
            result = {
                "status": "success",
                "generated_files": [str(target_file)],
                "message": "数据加载代码生成成功"
            }
            
            self.logger.info("数据加载代码生成成功")
            return result
                
        except Exception as e:
            self.logger.error(f"生成数据加载代码时出错: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "message": f"生成数据加载代码失败: {str(e)}"
            }
    
    async def _test_data_loader(self, workspace_dir: Path) -> Dict[str, str]:
        """执行数据加载测试并返回数据结构信息（无论debug模式如何都会执行）"""
        try:
            loader_file = workspace_dir / "load_research_data.py"
            if not loader_file.exists():
                return {
                    "status": "error",
                    "error": "数据加载文件不存在",
                    "message": "load_research_data.py文件未找到",
                    "data_structure_info": "",
                    "detailed_error": ""
                }
            
            self.logger.info("执行数据加载测试（小样本模式）...")
            
            # 执行Python脚本，确保捕获所有输出
            result = subprocess.run(
                ["python", "-u", "load_research_data.py"],  # -u确保输出不缓冲
                cwd=str(workspace_dir),
                capture_output=True,
                text=True,
                timeout=120,  # 增加超时时间
                env={**os.environ, "PYTHONUNBUFFERED": "1", "USE_SMALL_SAMPLE": "1"}  # 添加小样本环境变量
            )
            
            # 合并stdout和stderr作为完整输出
            full_output = ""
            if result.stdout:
                full_output += result.stdout
            if result.stderr:
                full_output += "\n=== STDERR ===\n" + result.stderr
            
            if result.returncode == 0:
                self.logger.info("数据加载测试成功")
                return {
                    "status": "success",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "data_structure_info": result.stdout,
                    "detailed_error": "",
                    "message": "数据加载测试完成"
                }
            else:
                self.logger.error(f"数据加载测试失败，退出码: {result.returncode}")
                self.logger.error(f"错误输出: {result.stderr}")
                return {
                    "status": "error",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "data_structure_info": full_output,  # 包含所有输出信息
                    "detailed_error": result.stderr,  # 详细错误信息
                    "message": "数据加载测试失败"
                }
                
        except subprocess.TimeoutExpired:
            self.logger.error("数据加载测试超时")
            return {
                "status": "error",
                "error": "执行超时",
                "message": "数据加载测试超时",
                "data_structure_info": "",
                "detailed_error": "执行超时"
            }
        except Exception as e:
            self.logger.error(f"数据加载测试出错: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "message": f"数据加载测试失败: {str(e)}",
                "data_structure_info": "",
                "detailed_error": str(e)
            }


# 便捷函数
async def run_research_pipeline(
    research_topic: str,
    research_goal: str,
    dataset_path: str
) -> Dict[str, str]:
    """执行研究流水线的便捷函数"""
    agent = InterAgent()
    return await agent.run_research_task(research_topic, research_goal, dataset_path)