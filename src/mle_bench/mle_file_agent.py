import os
import asyncio
import logging
import zipfile
from typing import Dict, List, Optional
from pathlib import Path
from src.utils.openai_client import get_client


class MLEFileAgent:
    """MLE Bench文件代理，负责解析dataset信息并提取研究目标"""
    
    def __init__(self):
        self.client = get_client()
        # 使用.cache目录下的数据路径
        self.cache_path = Path(__file__).parent.parent.parent / ".cache" / "mle-bench" / "data"
    
    def find_dataset_path(self, dataset_name: str) -> Optional[str]:
        """根据dataset名称找到对应的路径"""
        dataset_path = self.cache_path / dataset_name / "prepared" / "public"
        if dataset_path.exists():
            return str(dataset_path)
        return None
    
    async def analyze_dataset_info(self, dataset_name: str) -> Dict[str, str]:
        """分析dataset信息，提取research topic和goal"""
        dataset_path = self.find_dataset_path(dataset_name)
        if not dataset_path:
            raise ValueError(f"数据集 {dataset_name} 未找到")
        
        # 读取description.md文件，从cache目录下读取
        cache_dataset_path = self.cache_path / dataset_name
        description_file = cache_dataset_path / "prepared" / "public" / "description.md"
        if not description_file.exists():
            raise ValueError(f"数据集 {dataset_name} 的描述文件未找到: {description_file}")
        
        with open(description_file, 'r', encoding='utf-8') as f:
            description_content = f.read()
        
        # 使用LLM分析内容
        prompt = f"""
        请分析以下数据集描述，提取关键信息：

        描述内容：
        {description_content}

        请按照以下格式返回分析结果：
        Task Description: [详细描述这个竞赛/任务要解决的具体问题]
        Optimization Metric: [提取出需要优化的具体指标，如"RMSLE"、"F1 Score"、"AUC"等]

        要求：
        1. Task Description要详细描述任务背景、目标和方法
        2. Optimization Metric只需要提取出评估指标的名称，如果有多个指标请列出
        3. 用中文回答，保持准确和简洁
        """
        
        messages = [{"role": "user", "content": prompt}]
        response = await self.client.chat(messages)
        
        # 解析LLM响应
        task_description = ""
        optimization_metric = ""
        
        # 使用更灵活的解析方法
        if "**Task Description:**" in response and "**Optimization Metric:**" in response:
            # 处理带格式的响应
            task_start = response.find("**Task Description:**") + len("**Task Description:**")
            metric_start = response.find("**Optimization Metric:**")
            task_description = response[task_start:metric_start].strip()
            
            metric_start = response.find("**Optimization Metric:**") + len("**Optimization Metric:**")
            optimization_metric = response[metric_start:].strip()
        else:
            # 处理无格式的响应
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("Task Description:"):
                    task_description = line.replace("Task Description:", "").strip()
                elif line.startswith("Optimization Metric:"):
                    optimization_metric = line.replace("Optimization Metric:", "").strip()
        
        # 根据benchmark类型添加固定目标描述
        benchmark_type = self._detect_benchmark_type()
        fixed_goal = self._get_fixed_goal(benchmark_type)
        
        # 组合最终的research_goal
        research_goal = f"优化{optimization_metric}指标"
        if fixed_goal:
            research_goal += f"，{fixed_goal}"
        
        return {
            "research_topic": task_description,
            "research_goal": research_goal,
            "dataset_path": dataset_path
        }
    
    def _detect_benchmark_type(self) -> str:
        """检测benchmark类型"""
        # 根据路径判断是mlebench还是paperbench
        if "mle-bench" in str(self.cache_path):
            return "mlebench"
        elif "paperbench" in str(self.cache_path):
            return "paperbench"
        else:
            return "other"
    
    def _get_fixed_goal(self, benchmark_type: str) -> str:
        """根据benchmark类型获取固定目标描述"""
        if benchmark_type == "mlebench":
            return "在该Kaggle竞赛中取得金牌"
        elif benchmark_type == "paperbench":
            return "复现论文中的实验结果并达到SOTA性能"
        else:
            return ""
    
    def extract_zip_files(self, dataset_path: str) -> None:
        """解压数据集路径下的zip文件"""
        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            return
        
        # 查找所有zip文件
        zip_files = list(dataset_dir.glob("*.zip"))
        
        for zip_file in zip_files:
            # 提取文件名（不含扩展名）作为目标文件夹名
            target_dir = dataset_dir / zip_file.stem
            
            # 如果目标文件夹已存在，跳过
            if target_dir.exists():
                logging.info(f"目录已存在，跳过解压: {target_dir}")
                continue
            
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    # 创建目标目录
                    target_dir.mkdir(exist_ok=True)
                    # 解压到目标目录
                    zip_ref.extractall(target_dir)
                    logging.info(f"成功解压 {zip_file} 到 {target_dir}")
                    
            except Exception as e:
                logging.error(f"解压文件失败 {zip_file}: {str(e)}")

    async def get_dataset_info(self, dataset_name: str) -> Dict[str, str]:
        """获取完整的dataset信息"""
        try:
            logging.info(f"开始分析数据集: {dataset_name}")
            
            # 获取数据集路径并解压zip文件
            dataset_path = self.find_dataset_path(dataset_name)
            if dataset_path:
                self.extract_zip_files(dataset_path)
            
            info = await self.analyze_dataset_info(dataset_name)
            logging.info(f"数据集分析完成: {dataset_name}")
            return {
                "dataset_name": dataset_name,
                "research_topic": info["research_topic"],
                "research_goal": info["research_goal"],
                "dataset_path": info["dataset_path"]
            }
        except Exception as e:
            logging.error(f"分析数据集失败 {dataset_name}: {str(e)}")
            raise ValueError(f"分析数据集失败 {dataset_name}: {str(e)}")


# 便捷函数
async def get_mle_dataset_info(dataset_name: str) -> Dict[str, str]:
    """获取MLE数据集信息的便捷函数"""
    agent = MLEFileAgent()
    return await agent.get_dataset_info(dataset_name)