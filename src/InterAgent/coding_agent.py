import os
import asyncio
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
from src.utils.openai_client import get_client
from src.InterAgent.coding_agent_prompts import (
    build_package_installation_prompt,
    build_research_code_generation_prompt,
    build_debug_prompt,
    build_optimization_prompt
)


class CodingAgent:
    """编程代理，负责代码生成、调试和性能优化"""
    
    def __init__(self, model_name: Optional[str] = None):
        self.client = get_client(model_name)
        self.logger = logging.getLogger(__name__)
    
    async def generate_load_data_code(
        self,
        requirements: str,
        output_dir: str,
        project_name: str = "generated_project",
        language: str = "python",
        framework: Optional[str] = None
    ) -> Dict[str, str]:
        """
        生成数据加载代码
        
        Args:
            requirements: 需求描述
            output_dir: 输出目录
            project_name: 项目名称
            language: 编程语言
            framework: 框架选择（如scikit-learn, pytorch等）
        
        Returns:
            Dict包含生成的文件路径和状态信息
        """
        self.logger.info(f"开始生成数据加载代码: {project_name}")
        
        # 创建输出目录
        project_path = Path(output_dir) / project_name
        project_path.mkdir(parents=True, exist_ok=True)
        
        # 构建代码生成prompt
        prompt = self._build_generation_prompt(requirements, language, framework)
        
        try:
            # 调用LLM生成代码
            messages = [{"role": "user", "content": prompt}]
            response = await self.client.chat(messages)
            
            # 解析并保存生成的代码文件
            generated_files = await self._parse_and_save_code(response, project_path)
            
            self.logger.info(f"数据加载代码生成完成，共生成 {len(generated_files)} 个文件")
            
            return {
                "status": "success",
                "project_path": str(project_path),
                "generated_files": generated_files,
                "message": f"成功生成数据加载代码 {project_name}"
            }
            
        except Exception as e:
            self.logger.error(f"数据加载代码生成失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "message": f"数据加载代码生成失败: {str(e)}"
            }
    
    async def generate_research_code(
        self,
        research_topic: str,
        research_goal: str,
        init_analysis: str,
        data_structure_info: str,
        submission_file_name: str,
        workspace_dir: str,
        device_info: Optional[Dict] = None,
        data_loader_info: Optional[str] = None
    ) -> Dict[str, str]:
        """
        生成研究代码(research.py)，实现完整的ML pipeline

        Args:
            research_topic: 研究主题
            research_goal: 研究目标
            init_analysis: 初步分析结果
            data_structure_info: 数据结构信息
            submission_file_name: 提交文件名
            workspace_dir: 工作目录
            device_info: 设备信息（GPU/CPU配置）
            data_loader_info: load_research_data函数的接口信息

        Returns:
            Dict包含生成的文件路径和状态信息
        """
        self.logger.info("开始生成研究代码...")

        try:
            # 构建研究代码生成prompt
            prompt = self._build_research_generation_prompt(
                research_topic, research_goal, init_analysis,
                data_structure_info, submission_file_name, device_info,
                data_loader_info
            )
            
            # 调用LLM生成代码
            messages = [{"role": "user", "content": prompt}]
            response = await self.client.chat(messages, max_tokens=6000)
            
            # 提取代码并保存到research.py
            research_code = self._extract_code_from_response(response)
            research_file = Path(workspace_dir) / "research.py"
            
            with open(research_file, 'w', encoding='utf-8') as f:
                f.write(research_code)
            
            self.logger.info(f"研究代码生成完成: {research_file}")
            
            return {
                "status": "success",
                "generated_file": str(research_file),
                "message": "研究代码生成成功"
            }
            
        except Exception as e:
            self.logger.error(f"研究代码生成失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "message": f"研究代码生成失败: {str(e)}"
            }
    
    async def install_missing_package(
        self,
        error_message: str,
        project_root: str = "/Users/zhzhou/Desktop/InterAgentV2"
    ) -> Dict[str, str]:
        """
        根据ModuleNotFoundError自动安装缺失的包

        Args:
            error_message: 错误信息
            project_root: 项目根目录

        Returns:
            Dict包含安装结果和状态信息
        """
        self.logger.info("检测到ModuleNotFoundError，开始分析缺失的包...")

        try:
            # 使用LLM分析需要安装的包
            prompt = build_package_installation_prompt(error_message)
            messages = [{"role": "user", "content": prompt}]
            response = await self.client.chat(messages, max_tokens=500)

            # 解析LLM返回的包名
            package_info = self._parse_json_response(response)

            if not package_info or 'package_name' not in package_info:
                return {
                    "status": "error",
                    "error": "无法解析LLM返回的包名信息",
                    "message": f"LLM返回: {response}"
                }

            package_name = package_info['package_name']
            missing_module = package_info.get('missing_module', 'unknown')
            reasoning = package_info.get('reasoning', '')

            self.logger.info(f"缺失模块: {missing_module}")
            self.logger.info(f"需要安装的包: {package_name}")
            self.logger.info(f"原因: {reasoning}")

            # 使用uv add安装包
            import subprocess
            cmd = ["uv", "add", package_name]

            self.logger.info(f"执行安装命令: {' '.join(cmd)}")
            self.logger.info(f"工作目录: {project_root}")

            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )

            if result.returncode == 0:
                self.logger.info(f"成功安装包: {package_name}")
                return {
                    "status": "success",
                    "package_name": package_name,
                    "missing_module": missing_module,
                    "message": f"成功安装 {package_name}",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                self.logger.error(f"安装包失败: {package_name}")
                self.logger.error(f"STDOUT: {result.stdout}")
                self.logger.error(f"STDERR: {result.stderr}")
                return {
                    "status": "error",
                    "package_name": package_name,
                    "error": f"uv add 失败，退出码: {result.returncode}",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }

        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "error": "安装超时（超过5分钟）",
                "message": "包安装时间过长"
            }
        except Exception as e:
            self.logger.error(f"安装包时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "status": "error",
                "error": str(e),
                "message": f"安装失败: {str(e)}"
            }

    async def debug_code(
        self,
        code_path: str,
        error_message: str,
        error_context: Optional[str] = None
    ) -> Dict[str, str]:
        """
        根据报错信息调试代码
        
        Args:
            code_path: 代码文件路径
            error_message: 错误信息
            error_context: 错误上下文（可选）
        
        Returns:
            Dict包含修复后的代码和状态信息
        """
        self.logger.info(f"开始调试代码: {code_path}")
        
        try:
            # 读取原始代码
            with open(code_path, 'r', encoding='utf-8') as f:
                original_code = f.read()
            
            # 构建调试prompt
            prompt = self._build_debug_prompt(original_code, error_message, error_context)
            
            # 调用LLM进行调试
            messages = [{"role": "user", "content": prompt}]
            response = await self.client.chat(messages)
            
            # 解析修复后的代码
            fixed_code = self._extract_code_from_response(response)

            # 备份原文件
            backup_path = self._backup_file(code_path)

            # 保存修复后的代码
            with open(code_path, 'w', encoding='utf-8') as f:
                f.write(fixed_code)
            
            self.logger.info(f"代码调试完成: {code_path}")
            
            return {
                "status": "success",
                "original_code": original_code,
                "fixed_code": fixed_code,
                "backup_path": backup_path,
                "message": "代码调试完成"
            }
            
        except Exception as e:
            self.logger.error(f"代码调试失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "message": f"代码调试失败: {str(e)}"
            }
    
    async def optimize_performance(
        self,
        code_path: str,
        optimization_strategy: str,
        target_metrics: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        根据提升策略优化代码性能
        
        Args:
            code_path: 代码文件路径
            optimization_strategy: 优化策略描述
            target_metrics: 目标优化指标列表
        
        Returns:
            Dict包含优化后的代码和状态信息
        """
        self.logger.info(f"开始优化代码性能: {code_path}")
        
        try:
            # 读取原始代码
            with open(code_path, 'r', encoding='utf-8') as f:
                original_code = f.read()
            
            # 构建优化prompt
            prompt = self._build_optimization_prompt(
                original_code, optimization_strategy, target_metrics
            )
            
            # 调用LLM进行优化
            messages = [{"role": "user", "content": prompt}]
            response = await self.client.chat(messages)
            
            # 解析优化后的代码
            optimized_code = self._extract_code_from_response(response)

            # 备份原文件
            backup_path = self._backup_file(code_path)

            # 保存优化后的代码
            with open(code_path, 'w', encoding='utf-8') as f:
                f.write(optimized_code)
            
            self.logger.info(f"代码性能优化完成: {code_path}")
            
            return {
                "status": "success",
                "original_code": original_code,
                "optimized_code": optimized_code,
                "backup_path": backup_path,
                "optimization_strategy": optimization_strategy,
                "message": "代码性能优化完成"
            }
            
        except Exception as e:
            self.logger.error(f"代码优化失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "message": f"代码优化失败: {str(e)}"
            }
    
    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """从LLM响应中解析JSON，支持多种格式

        Args:
            response: LLM响应文本

        Returns:
            解析后的JSON对象，失败返回None
        """
        import json
        import re

        # 方式1: 直接解析
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        # 方式2: 提取```json...```代码块
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 方式3: 查找第一个完整的JSON对象
        brace_start = response.find('{')
        if brace_start != -1:
            brace_count = 0
            for i in range(brace_start, len(response)):
                if response[i] == '{':
                    brace_count += 1
                elif response[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            return json.loads(response[brace_start:i+1])
                        except json.JSONDecodeError:
                            pass
                        break

        return None

    def _backup_file(self, file_path: str) -> str:
        """备份文件

        Args:
            file_path: 要备份的文件路径

        Returns:
            备份文件路径
        """
        backup_path = f"{file_path}.backup"
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        return backup_path

    def _build_generation_prompt(
        self,
        requirements: str,
        language: str,
        framework: Optional[str]
    ) -> str:
        """构建代码生成的prompt"""
        framework_info = f"使用{framework}框架，" if framework else ""
        
        prompt = f"""
        请根据以下需求，用{language}语言{framework_info}生成完整的项目代码：

        需求描述：
        {requirements}

        请按照以下要求生成代码：
        1. 生成完整可运行的项目结构
        2. 包含必要的依赖文件（如requirements.txt）
        3. 代码要有良好的注释和文档
        4. 遵循最佳实践和代码规范
        5. 确保代码的可读性和可维护性

        请按照以下格式输出：
        ```文件名
        文件内容
        ```

        每个文件用上述格式分开，确保输出的代码完整且可以直接运行。
        """
        return prompt
    
    def _build_research_generation_prompt(
        self,
        research_topic: str,
        research_goal: str,
        init_analysis: str,
        data_structure_info: str,
        submission_file_name: str,
        device_info: Optional[Dict] = None,
        data_loader_info: Optional[str] = None
    ) -> str:
        """构建研究代码生成的prompt"""
        if device_info is None:
            device_info = {'device_type': 'cpu', 'device_name': 'CPU'}

        return build_research_code_generation_prompt(
            research_topic=research_topic,
            research_goal=research_goal,
            init_analysis=init_analysis,
            data_structure_info=data_structure_info,
            submission_file_name=submission_file_name,
            device_info=device_info,
            data_loader_info=data_loader_info
        )

    def _build_debug_prompt(
        self,
        code: str,
        error_message: str,
        error_context: Optional[str]
    ) -> str:
        """构建调试的prompt"""
        context_info = error_context if error_context else ""
        return build_debug_prompt(
            error_message=error_message,
            error_context=context_info,
            code_content=code
        )
    
    def _build_optimization_prompt(
        self,
        code: str,
        strategy: str,
        target_metrics: Optional[List[str]]
    ) -> str:
        """构建性能优化的prompt"""
        metrics_info = f"目标指标: {', '.join(target_metrics)}" if target_metrics else ""
        optimization_goals = f"{strategy}\n{metrics_info}".strip()

        return build_optimization_prompt(
            code_content=code,
            performance_issue="需要性能优化",
            optimization_goals=optimization_goals
        )
    
    async def _parse_and_save_code(self, response: str, project_path: Path) -> List[str]:
        """解析LLM响应并保存代码文件"""
        generated_files = []
        
        # 简单的代码块解析
        lines = response.split('\n')
        current_file = None
        current_content = []
        in_code_block = False
        
        for line in lines:
            if line.startswith('```') and not in_code_block:
                # 开始代码块
                if current_file and current_content:
                    # 保存前一个文件
                    file_path = project_path / current_file
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(current_content))
                    generated_files.append(str(file_path))
                
                # 提取文件名
                current_file = line[3:].strip()
                if not current_file:
                    current_file = "main.py"  # 默认文件名
                current_content = []
                in_code_block = True
                
            elif line.startswith('```') and in_code_block:
                # 结束代码块
                in_code_block = False
                
            elif in_code_block:
                # 代码内容
                current_content.append(line)
        
        # 保存最后一个文件
        if current_file and current_content:
            file_path = project_path / current_file
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(current_content))
            generated_files.append(str(file_path))
        
        return generated_files
    
    def _extract_code_from_response(self, response: str) -> str:
        """从LLM响应中提取代码"""
        # 查找代码块
        lines = response.split('\n')
        code_lines = []
        in_code_block = False
        
        for line in lines:
            if line.startswith('```'):
                if in_code_block:
                    break  # 结束代码块
                else:
                    in_code_block = True  # 开始代码块
                    continue
            
            if in_code_block:
                code_lines.append(line)
        
        if not code_lines:
            # 如果没有找到代码块，返回整个响应
            return response.strip()
        
        return '\n'.join(code_lines)


# 便捷函数
async def generate_load_data_project(
    requirements: str,
    output_dir: str,
    project_name: str = "generated_project",
    language: str = "python",
    framework: Optional[str] = None
) -> Dict[str, str]:
    """生成数据加载代码项目的便捷函数"""
    agent = CodingAgent()
    return await agent.generate_load_data_code(requirements, output_dir, project_name, language, framework)


async def debug_code_file(
    code_path: str,
    error_message: str,
    error_context: Optional[str] = None
) -> Dict[str, str]:
    """调试代码文件的便捷函数"""
    agent = CodingAgent()
    return await agent.debug_code(code_path, error_message, error_context)


async def optimize_code_performance(
    code_path: str,
    optimization_strategy: str,
    target_metrics: Optional[List[str]] = None
) -> Dict[str, str]:
    """优化代码性能的便捷函数"""
    agent = CodingAgent()
    return await agent.optimize_performance(code_path, optimization_strategy, target_metrics)