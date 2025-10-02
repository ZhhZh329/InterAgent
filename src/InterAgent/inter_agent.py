import os
import asyncio
import logging
import subprocess
from typing import Dict, List, Optional, TypedDict
from pathlib import Path
from datetime import datetime
from src.utils.openai_client import get_client
from src.InterAgent.coding_agent import CodingAgent
from src.mle_bench.evaluator_agent import MLEBenchEvaluator
from src.InterAgent.inter_agent_prompts import (
    build_structure_analysis_prompt,
    build_init_analysis_prompt,
    build_data_loader_generation_prompt
)


# LangGraph State定义
class ResearchState(TypedDict, total=False):
    """研究任务的完整状态"""
    # 输入参数
    research_topic: str
    research_goal: str
    dataset_path: str
    benchmark_name: str
    dataset_name: str
    debug: bool
    workspace_dir: Path

    # 文件结构相关
    structure_info: Dict

    # 数据加载相关
    data_loader_result: Dict
    test_result: Dict

    # 分析相关
    init_analysis_result: Dict

    # 研究代码相关
    research_code_result: Dict
    research_test_result: Dict
    submission_file_name: str

    # 完整运行相关
    full_run_result: Dict

    # 评估相关
    evaluation_result: Dict

    # 重试计数器
    install_attempts: int
    debug_retry_count: int
    full_run_retry_count: int
    evaluation_retry_count: int
    regeneration_count: int

    # 最终结果
    status: str
    message: str
    error: Optional[str]


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

    # ===== LangGraph Node Functions =====
    # These are wrapper nodes that call existing methods

    async def node_setup(self, state: ResearchState) -> ResearchState:
        """节点: 初始化工作目录"""
        workspace_dir = self._create_workspace(state['benchmark_name'], state['dataset_name'])
        self.logger.info(f"工作目录: {workspace_dir}")
        return {'workspace_dir': workspace_dir}

    async def node_probe_structure(self, state: ResearchState) -> ResearchState:
        """节点: 探测文件结构"""
        structure_info = await self.file_structure_probing(state['dataset_path'])
        self.logger.info(f"文件结构分析完成: {structure_info['status']}")
        return {'structure_info': structure_info}

    async def node_generate_data_loader(self, state: ResearchState) -> ResearchState:
        """节点: 生成数据加载代码"""
        data_loader_result = await self._generate_data_loader(
            state['structure_info'],
            state['dataset_path'],
            state['workspace_dir']
        )
        self.logger.info(f"数据加载代码生成: {data_loader_result['status']}")
        return {'data_loader_result': data_loader_result}

    async def node_test_data_loader(self, state: ResearchState) -> ResearchState:
        """节点: 测试数据加载"""
        test_result = await self._test_data_loader(state['workspace_dir'])
        self.logger.info(f"数据加载测试: {test_result['status']}")
        return {'test_result': test_result}

    async def node_debug_data_loader(self, state: ResearchState) -> ResearchState:
        """节点: Debug数据加载代码"""
        debug_result = await self._debug_data_loader_with_retry(
            state['workspace_dir'],
            state['test_result'].get('detailed_error', '')
        )
        if debug_result['status'] == 'success':
            # 重新测试
            test_result = await self._test_data_loader(state['workspace_dir'])
            self.logger.info(f"修复后数据加载测试: {test_result['status']}")
            return {'test_result': test_result}
        return {}

    async def node_init_analysis(self, state: ResearchState) -> ResearchState:
        """节点: 初步分析"""
        init_analysis_result = await self.init_analyzing(
            research_topic=state['research_topic'],
            research_goal=state['research_goal'],
            dataset_path=state['dataset_path'],
            data_structure_info=state['test_result'].get('data_structure_info', ''),
            structure_analysis=state['structure_info'].get('analysis_result', {})
        )
        self.logger.info(f"初步分析: {init_analysis_result['status']}")
        return {'init_analysis_result': init_analysis_result}

    async def node_generate_research_code(self, state: ResearchState) -> ResearchState:
        """节点: 生成研究代码"""
        submission_file_name = self._get_submission_file_name(
            state['structure_info'],
            state['benchmark_name']
        )

        research_code_result = await self._generate_research_code(
            research_topic=state['research_topic'],
            research_goal=state['research_goal'],
            init_analysis=state['init_analysis_result']['analysis_content'],
            data_structure_info=state['test_result']['data_structure_info'],
            submission_file_name=submission_file_name,
            workspace_dir=state['workspace_dir']
        )
        self.logger.info(f"研究代码生成: {research_code_result['status']}")

        return {
            'submission_file_name': submission_file_name,
            'research_code_result': research_code_result
        }

    async def node_test_research_code(self, state: ResearchState) -> ResearchState:
        """节点: 测试研究代码（小样本）"""
        research_test_result = await self._test_research_code(state['workspace_dir'])
        self.logger.info(f"研究代码测试: {research_test_result['status']}")
        return {'research_test_result': research_test_result}

    async def node_auto_fix_code(self, state: ResearchState) -> ResearchState:
        """节点: 自动修复代码问题（依赖安装+debug+重新生成）"""
        self.logger.info("研究代码执行失败，启动自动debug循环...")
        fix_result = await self._auto_fix_code_issues(
            workspace_dir=state['workspace_dir'],
            research_test_result=state['research_test_result'],
            research_topic=state['research_topic'],
            research_goal=state['research_goal'],
            init_analysis_result=state['init_analysis_result'],
            test_result=state['test_result'],
            submission_file_name=state['submission_file_name'],
            current_regeneration_count=state.get('regeneration_count', 0)
        )

        return {
            'research_test_result': fix_result['final_test_result'],
            'install_attempts': fix_result['install_attempts'],
            'debug_retry_count': fix_result['debug_retry_count'],
            'regeneration_count': fix_result['regeneration_count']
        }

    async def node_run_full_dataset(self, state: ResearchState) -> ResearchState:
        """节点: 运行完整数据集"""
        self.logger.info("=" * 80)
        self.logger.info("小样本测试通过，开始运行完整数据集...")
        self.logger.info("=" * 80)
        full_run_result = await self._run_full_research_code(state['workspace_dir'])
        self.logger.info(f"完整数据集运行: {full_run_result['status']}")
        return {'full_run_result': full_run_result}

    async def node_optimize_full_run(self, state: ResearchState) -> ResearchState:
        """节点: 优化并重试完整数据集运行（与debug失败共享regeneration_count）"""
        self.logger.error(f"完整数据集运行失败: {state['full_run_result'].get('message', '')}")
        optimize_result = await self._optimize_and_retry_full_run(
            workspace_dir=state['workspace_dir'],
            full_run_result=state['full_run_result'],
            research_topic=state['research_topic'],
            research_goal=state['research_goal'],
            init_analysis_result=state['init_analysis_result'],
            test_result=state['test_result'],
            submission_file_name=state['submission_file_name'],
            dataset_name=state['dataset_name'],
            benchmark_name=state['benchmark_name'],
            current_regeneration_count=state.get('regeneration_count', 0)
        )

        return {
            'full_run_result': optimize_result['final_full_run_result'],
            'regeneration_count': optimize_result['regeneration_count']
        }

    async def node_evaluate(self, state: ResearchState) -> ResearchState:
        """节点: 评估提交结果"""
        self.logger.info("完整数据集训练成功！")
        self.logger.info(f"生成的submission文件: {state['full_run_result'].get('submission_files', [])}")

        evaluation_result = await self._evaluate_submission(
            state['workspace_dir'],
            state['dataset_name'],
            state['benchmark_name']
        )

        # 更新full_run_result中的evaluation
        full_run_result = state['full_run_result'].copy()
        full_run_result['evaluation'] = evaluation_result

        # 实时输出评估结果，方便用户查看执行进度
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

        return {
            'evaluation_result': evaluation_result,
            'full_run_result': full_run_result
        }

    # ===== Routing Functions =====
    # These determine which node to go to next

    def route_after_data_loader_test(self, state: ResearchState) -> str:
        """路由: 数据加载测试后"""
        if state['test_result']['status'] == 'error' and state['test_result'].get('detailed_error'):
            return "debug_data_loader"
        return "init_analysis"

    def route_after_data_loader_debug(self, state: ResearchState) -> str:
        """路由: 数据加载debug后"""
        # 无论debug成功或失败，都进入init_analysis（使用现有的test_result）
        return "init_analysis"

    def route_after_init_analysis(self, state: ResearchState) -> str:
        """路由: 初步分析后"""
        if state.get('init_analysis_result') and state['init_analysis_result']['status'] == 'success':
            return "generate_research_code"
        return "end"

    def route_after_research_code_test(self, state: ResearchState) -> str:
        """路由: 研究代码测试后"""
        if state['research_test_result']['status'] == 'error' and state['research_test_result'].get('detailed_error'):
            return "auto_fix_code"
        return "run_full_dataset"

    def route_after_auto_fix(self, state: ResearchState) -> str:
        """路由: 自动修复后"""
        if state['research_test_result']['status'] == 'success':
            return "run_full_dataset"
        return "end"

    def route_after_full_run(self, state: ResearchState) -> str:
        """路由: 完整运行后"""
        if state['full_run_result']['status'] == 'success':
            return "evaluate"
        return "optimize_full_run"

    def route_after_optimize(self, state: ResearchState) -> str:
        """路由: 优化重试后"""
        if state['full_run_result']['status'] == 'success':
            return "evaluate"
        return "end"

    def route_after_evaluate(self, state: ResearchState) -> str:
        """路由: 评估后"""
        # 评估完成后结束（evaluation retry logic暂时简化，待未来扩展）
        return "end"

    def build_research_graph(self):
        """构建LangGraph研究流程图"""
        from langgraph.graph import StateGraph, END

        # 创建StateGraph
        graph = StateGraph(ResearchState)

        # 添加所有节点
        graph.add_node("setup", self.node_setup)
        graph.add_node("probe_structure", self.node_probe_structure)
        graph.add_node("generate_data_loader", self.node_generate_data_loader)
        graph.add_node("test_data_loader", self.node_test_data_loader)
        graph.add_node("debug_data_loader", self.node_debug_data_loader)
        graph.add_node("init_analysis", self.node_init_analysis)
        graph.add_node("generate_research_code", self.node_generate_research_code)
        graph.add_node("test_research_code", self.node_test_research_code)
        graph.add_node("auto_fix_code", self.node_auto_fix_code)
        graph.add_node("run_full_dataset", self.node_run_full_dataset)
        graph.add_node("optimize_full_run", self.node_optimize_full_run)
        graph.add_node("evaluate", self.node_evaluate)

        # 设置入口点
        graph.set_entry_point("setup")

        # 添加边（定义流程）
        graph.add_edge("setup", "probe_structure")
        graph.add_edge("probe_structure", "generate_data_loader")
        graph.add_edge("generate_data_loader", "test_data_loader")

        # 条件边: 数据加载测试后
        graph.add_conditional_edges(
            "test_data_loader",
            self.route_after_data_loader_test,
            {
                "debug_data_loader": "debug_data_loader",
                "init_analysis": "init_analysis"
            }
        )

        # 条件边: 数据加载debug后
        graph.add_conditional_edges(
            "debug_data_loader",
            self.route_after_data_loader_debug,
            {
                "init_analysis": "init_analysis"
            }
        )

        # 条件边: 初步分析后
        graph.add_conditional_edges(
            "init_analysis",
            self.route_after_init_analysis,
            {
                "generate_research_code": "generate_research_code",
                "end": END
            }
        )

        # 直接边
        graph.add_edge("generate_research_code", "test_research_code")

        # 条件边: 研究代码测试后
        graph.add_conditional_edges(
            "test_research_code",
            self.route_after_research_code_test,
            {
                "auto_fix_code": "auto_fix_code",
                "run_full_dataset": "run_full_dataset"
            }
        )

        # 条件边: 自动修复后
        graph.add_conditional_edges(
            "auto_fix_code",
            self.route_after_auto_fix,
            {
                "run_full_dataset": "run_full_dataset",
                "end": END
            }
        )

        # 条件边: 完整运行后
        graph.add_conditional_edges(
            "run_full_dataset",
            self.route_after_full_run,
            {
                "evaluate": "evaluate",
                "optimize_full_run": "optimize_full_run"
            }
        )

        # 条件边: 优化后
        graph.add_conditional_edges(
            "optimize_full_run",
            self.route_after_optimize,
            {
                "evaluate": "evaluate",
                "end": END
            }
        )

        # 条件边: 评估后
        graph.add_conditional_edges(
            "evaluate",
            self.route_after_evaluate,
            {
                "end": END
            }
        )

        # 编译graph
        return graph.compile()

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
            
            # 实时输出LLM响应，方便查看执行进度
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
        return build_structure_analysis_prompt(structure_summary)
    
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
    
    async def _auto_fix_code_issues(
        self,
        workspace_dir: Path,
        research_test_result: Dict,
        research_topic: str,
        research_goal: str,
        init_analysis_result: Dict,
        test_result: Dict,
        submission_file_name: str,
        current_regeneration_count: int = 0,
        max_debug_retries: int = 3,
        max_install_attempts: int = 3,
        max_regeneration: int = 3
    ) -> Dict:
        """
        自动修复代码问题的完整循环逻辑（提取自run_research_task）
        包括依赖安装、代码debug、以及debug失败后的重新生成

        Args:
            workspace_dir: 工作目录
            research_test_result: 研究代码测试结果
            research_topic: 研究主题
            research_goal: 研究目标
            init_analysis_result: 初步分析结果
            test_result: 测试结果（包含data_structure_info）
            submission_file_name: 提交文件名
            current_regeneration_count: 当前已重新生成次数（与完整运行失败共享）
            max_debug_retries: 每次代码的最大debug尝试次数
            max_install_attempts: 最大依赖安装尝试次数
            max_regeneration: 全局最大重新生成次数

        Returns:
            Dict containing:
                - final_test_result: 最终测试结果
                - install_attempts: 依赖安装尝试次数
                - debug_retry_count: debug重试次数
                - regeneration_count: 重新生成次数
        """
        retry_count = 0
        install_attempts = 0
        regeneration_count = current_regeneration_count

        while retry_count < max_debug_retries and research_test_result['status'] == 'error':
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
                self.logger.info(f"Debug尝试 {retry_count}/{max_debug_retries}...")
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

        # Debug失败后，如果还有重新生成机会，则重新生成代码
        if research_test_result['status'] == 'error':
            self.logger.error(f"经过{max_debug_retries}次debug尝试和{install_attempts}次依赖安装后，研究代码仍然失败")

            # 检查是否还能重新生成
            if regeneration_count < max_regeneration:
                regeneration_count += 1
                self.logger.info("=" * 80)
                self.logger.info(f"=== 第 {regeneration_count} 次重新生成research.py（debug失败）===")
                self.logger.info(f"失败原因: {research_test_result.get('message', '')}")
                self.logger.info("=" * 80)

                # 构建错误提示
                error_output = research_test_result.get('detailed_error', '')
                regeneration_hint = f"""
**前一版本代码存在问题，需要重新生成**：
前一版本代码经过{max_debug_retries}次debug尝试仍然失败，可能代码逻辑从根本上有问题。

错误信息：
{error_output[-1000:] if len(error_output) > 1000 else error_output}

请重新分析任务需求，生成全新的、更可靠的代码实现。
"""

                # 重新生成代码，带上错误提示
                regenerated_init_analysis = init_analysis_result['analysis_content'] + "\n\n" + regeneration_hint

                research_code_result = await self._generate_research_code(
                    research_topic=research_topic,
                    research_goal=research_goal,
                    init_analysis=regenerated_init_analysis,
                    data_structure_info=test_result['data_structure_info'],
                    submission_file_name=submission_file_name,
                    workspace_dir=workspace_dir
                )

                if research_code_result['status'] == 'success':
                    # 测试新生成的代码
                    research_test_result = await self._test_research_code(workspace_dir)
                    self.logger.info(f"重新生成后研究代码测试: {research_test_result['status']}")

                    if research_test_result['status'] == 'success':
                        self.logger.info("=" * 80)
                        self.logger.info(f"=== 第 {regeneration_count} 次重新生成后测试成功 ===")
                        self.logger.info("=" * 80)
                    else:
                        # 新生成的代码仍然失败，递归调用进行debug或再次重新生成
                        self.logger.warning("重新生成的代码仍然失败，继续尝试修复...")
                        return await self._auto_fix_code_issues(
                            workspace_dir=workspace_dir,
                            research_test_result=research_test_result,
                            research_topic=research_topic,
                            research_goal=research_goal,
                            init_analysis_result=init_analysis_result,
                            test_result=test_result,
                            submission_file_name=submission_file_name,
                            current_regeneration_count=regeneration_count,
                            max_debug_retries=max_debug_retries,
                            max_install_attempts=max_install_attempts,
                            max_regeneration=max_regeneration
                        )
                else:
                    self.logger.error(f"重新生成代码失败: {research_code_result.get('message', '')}")
            else:
                self.logger.error(f"已达到最大重新生成次数({max_regeneration})，无法继续修复")

        return {
            'final_test_result': research_test_result,
            'install_attempts': install_attempts,
            'debug_retry_count': retry_count,
            'regeneration_count': regeneration_count
        }

    async def _optimize_and_retry_full_run(
        self,
        workspace_dir: Path,
        full_run_result: Dict,
        research_topic: str,
        research_goal: str,
        init_analysis_result: Dict,
        test_result: Dict,
        submission_file_name: str,
        dataset_name: str,
        benchmark_name: str,
        current_regeneration_count: int = 0,
        max_regeneration: int = 3
    ) -> Dict:
        """
        完整数据集运行失败后的优化重试循环（提取自run_research_task）

        Args:
            workspace_dir: 工作目录
            full_run_result: 完整数据集运行结果
            research_topic: 研究主题
            research_goal: 研究目标
            init_analysis_result: 初步分析结果
            test_result: 测试结果
            submission_file_name: 提交文件名
            dataset_name: 数据集名称
            benchmark_name: benchmark名称
            current_regeneration_count: 当前已重新生成次数（与debug失败共享）
            max_regeneration: 全局最大重新生成次数

        Returns:
            Dict containing:
                - final_full_run_result: 最终完整运行结果
                - regeneration_count: 重新生成次数
        """
        regeneration_count = current_regeneration_count

        while regeneration_count < max_regeneration and full_run_result['status'] == 'error':
            regeneration_count += 1

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
            self.logger.info(f"=== 第 {regeneration_count} 次重新生成research.py（完整数据集运行失败）===")
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
                        self.logger.info(f"=== 第 {regeneration_count} 次优化后完整运行成功 ===")
                        self.logger.info("=" * 80)

                        # 评估结果
                        evaluation_result = await self._evaluate_submission(
                            workspace_dir,
                            dataset_name,
                            benchmark_name
                        )
                        full_run_result['evaluation'] = evaluation_result

                        # 实时输出评估结果
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
                        self.logger.error(f"第 {regeneration_count} 次优化后完整运行仍然失败: {full_run_result.get('message', '')}")
                else:
                    self.logger.error(f"第 {regeneration_count} 次优化后的代码测试失败")
                    # 测试失败也要继续重试，不要break
            else:
                self.logger.error(f"第 {regeneration_count} 次重新生成research.py失败")
                break

        if regeneration_count >= max_regeneration and full_run_result['status'] == 'error':
            self.logger.error(f"已达到最大重新生成次数 ({max_regeneration})，完整运行仍然失败")

        return {
            'final_full_run_result': full_run_result,
            'regeneration_count': regeneration_count
        }

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
    
    def _execute_research_code(
        self,
        workspace_dir: Path,
        timeout: int,
        use_small_sample: bool = False
    ) -> subprocess.CompletedProcess:
        """执行research.py代码

        Args:
            workspace_dir: 工作目录
            timeout: 超时时间（秒）
            use_small_sample: 是否使用小样本模式

        Returns:
            subprocess.CompletedProcess对象
        """
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        if use_small_sample:
            env["USE_SMALL_SAMPLE"] = "1"
            env["QUICK_TEST"] = "1"
        else:
            # 完整运行时，移除小样本环境变量
            env.pop("USE_SMALL_SAMPLE", None)
            env.pop("QUICK_TEST", None)

        result = subprocess.run(
            ["python", "-u", "research.py"],
            cwd=str(workspace_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env
        )

        return result

    def _validate_csv_files(self, workspace_dir: Path) -> tuple:
        """验证workspace中的CSV文件

        Args:
            workspace_dir: 工作目录

        Returns:
            (valid_files, all_files): 有效文件列表和所有文件列表
        """
        import pandas as pd

        submission_files = list(workspace_dir.glob("*.csv"))
        valid_files = []

        for csv_file in submission_files:
            try:
                df = pd.read_csv(csv_file)
                # 检查文件不是空的，且有实际数据行
                if len(df) > 0 and not df.empty:
                    # 检查是否至少有1列数据
                    if len(df.columns) >= 1:
                        valid_files.append(csv_file)
                        self.logger.info(f"有效submission文件: {csv_file.name}, 形状: {df.shape}")
                    else:
                        self.logger.warning(f"submission文件列数不足: {csv_file.name}")
                else:
                    self.logger.warning(f"submission文件为空: {csv_file.name}")
            except Exception as e:
                self.logger.warning(f"无法读取submission文件 {csv_file.name}: {e}")

        return valid_files, submission_files

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
            prompt = build_init_analysis_prompt(
                research_topic=research_topic,
                research_goal=research_goal,
                data_structure_info=data_structure_info,
                train_files=structure_analysis.get('train_files', ''),
                test_files=structure_analysis.get('test_files', ''),
                data_format=structure_analysis.get('data_format', '')
            )

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

            requirements = build_data_loader_generation_prompt(
                dataset_path=dataset_path,
                structure_summary=structure_summary,
                train_files=analysis_result.get('train_files', ''),
                test_files=analysis_result.get('test_files', ''),
                data_format=analysis_result.get('data_format', ''),
                loading_suggestions=analysis_result.get('loading_suggestions', '')
            )

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
        主入口函数（LangGraph版本），执行研究任务

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
        self.logger.info("开始执行研究任务（LangGraph版本）")
        self.logger.info(f"研究主题: {research_topic}")
        self.logger.info(f"研究目标: {research_goal}")
        self.logger.info(f"数据集路径: {dataset_path}")

        try:
            # 构建LangGraph
            workflow = self.build_research_graph()

            # 初始化状态
            initial_state: ResearchState = {
                'research_topic': research_topic,
                'research_goal': research_goal,
                'dataset_path': dataset_path,
                'benchmark_name': benchmark_name,
                'dataset_name': dataset_name,
                'debug': debug
            }

            # 运行workflow
            final_state = await workflow.ainvoke(initial_state)

            # 判断最终状态
            status = "success"
            message = "研究任务完成"

            # 检查关键步骤是否成功
            research_test_result = final_state.get('research_test_result', {})
            full_run_result = final_state.get('full_run_result', {})
            evaluation_result = final_state.get('evaluation_result', {})

            # 1. 如果研究代码测试失败（经过debug后仍然失败）
            if research_test_result.get('status') == 'error':
                status = "error"
                message = f"研究代码测试失败: {research_test_result.get('message', 'unknown error')}"

            # 2. 如果完整数据集运行失败（经过优化后仍然失败）
            elif full_run_result.get('status') == 'error':
                status = "error"
                message = f"完整数据集运行失败: {full_run_result.get('message', 'unknown error')}"

            # 3. 如果有评估结果但评估失败
            elif evaluation_result and evaluation_result.get('status') == 'error':
                status = "error"
                message = f"评估失败: {evaluation_result.get('message', 'unknown error')}"

            # 4. 如果评估成功，使用评估结果的消息
            elif evaluation_result and evaluation_result.get('status') == 'success':
                status = "success"
                message = evaluation_result.get('message', '研究任务完成')

            # 构建返回结果
            return {
                "status": status,
                "workspace_dir": str(final_state.get('workspace_dir', '')),
                "research_topic": research_topic,
                "research_goal": research_goal,
                "dataset_path": dataset_path,
                "structure_info": final_state.get('structure_info', {}),
                "data_loader_result": final_state.get('data_loader_result', {}),
                "test_result": final_state.get('test_result', {}),
                "init_analysis_result": final_state.get('init_analysis_result', {}),
                "research_code_result": final_state.get('research_code_result', {}),
                "research_test_result": research_test_result,
                "full_run_result": full_run_result,
                "evaluation_result": evaluation_result,
                "message": message
            }

        except Exception as e:
            self.logger.error(f"研究任务执行失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "message": f"研究任务执行失败: {str(e)}"
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