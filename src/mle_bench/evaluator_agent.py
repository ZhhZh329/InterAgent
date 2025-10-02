"""
EvaluatorAgent: 通用评估代理基类

负责评估研究任务的结果，支持不同的benchmark评估方式
"""

import os
import json
import logging
import subprocess
import re
from abc import ABC, abstractmethod
from typing import Dict, Optional, List
from pathlib import Path
from src.utils.openai_client import OpenAIClient
from src.mle_bench.evaluator_prompts import build_mlebench_output_parsing_prompt


class EvaluatorAgent(ABC):
    """评估代理基类"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def evaluate(
        self,
        workspace_dir: Path,
        dataset_name: str,
        benchmark_name: str
    ) -> Dict:
        """
        评估研究任务结果

        Args:
            workspace_dir: 工作目录
            dataset_name: 数据集名称
            benchmark_name: benchmark名称

        Returns:
            评估结果字典，包含：
            - status: success/error
            - score: 评估分数
            - metrics: 其他评估指标
            - medal_info: 奖牌信息（如果适用）
            - valid_submission: 提交是否有效
            - message: 消息
        """
        pass

    def _find_submission_files(self, workspace_dir: Path) -> List[Path]:
        """
        在工作目录中查找可能的提交文件

        Args:
            workspace_dir: 工作目录

        Returns:
            找到的提交文件列表
        """
        submission_files = []

        # 常见的提交文件名模式
        patterns = [
            "submission.csv",
            "sample_submission.csv",
            "*submission*.csv",
            "*.csv"
        ]

        for pattern in patterns:
            files = list(workspace_dir.glob(pattern))
            for f in files:
                if f not in submission_files:
                    submission_files.append(f)

        return submission_files


class MLEBenchEvaluator(EvaluatorAgent):
    """MLEBench专用评估器"""

    def __init__(self):
        super().__init__()
        self.mlebench_command = "mlebench"
        self.llm_client = OpenAIClient()

    async def evaluate(
        self,
        workspace_dir: Path,
        dataset_name: str,
        benchmark_name: str
    ) -> Dict:
        """
        使用mlebench命令评估提交文件

        Args:
            workspace_dir: 工作目录
            dataset_name: 数据集名称
            benchmark_name: benchmark名称（应该是"mlebench"）

        Returns:
            评估结果
        """
        try:
            # 查找提交文件
            submission_files = self._find_submission_files(workspace_dir)

            if not submission_files:
                return {
                    "status": "error",
                    "error": "未找到提交文件",
                    "message": "在工作目录中未找到任何.csv提交文件",
                    "valid_submission": False,
                    "failure_type": "file_not_found",
                    "requires_code_regeneration": False
                }

            # 尝试评估每个找到的文件
            best_result = None
            all_errors = []

            for submission_file in submission_files:
                self.logger.info(f"尝试评估文件: {submission_file.name}")
                result = await self._evaluate_single_file(submission_file, dataset_name, benchmark_name)

                if result["status"] == "success":
                    # 找到第一个成功的评估就返回
                    if result.get("valid_submission", False):
                        self.logger.info(f"成功评估文件: {submission_file.name}")
                        result["submission_file"] = str(submission_file.name)
                        return result
                    elif best_result is None:
                        best_result = result
                        best_result["submission_file"] = str(submission_file.name)
                else:
                    # error状态的结果也要保存，因为包含LLM解析的重要字段
                    if best_result is None:
                        best_result = result
                        best_result["submission_file"] = str(submission_file.name)
                    all_errors.append(f"{submission_file.name}: {result.get('error', 'unknown')}")

            # 如果有任何结果，返回最好的（即使它是error状态）
            if best_result:
                self.logger.warning(f"所有文件都没有成功评估，返回最好的结果（status={best_result.get('status')}）")
                return best_result

            # 所有文件都失败且没有任何结果（理论上不应该到这里，因为上面已经保存了best_result）
            return {
                "status": "error",
                "error": "所有提交文件评估失败",
                "message": f"尝试的文件: {[str(f.name) for f in submission_files]}",
                "details": all_errors,
                "valid_submission": False,
                "failure_type": "unknown",
                "requires_code_regeneration": False
            }

        except Exception as e:
            self.logger.error(f"评估过程出错: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "message": f"评估失败: {str(e)}",
                "valid_submission": False,
                "failure_type": "unknown",
                "requires_code_regeneration": False
            }

    async def _evaluate_single_file(
        self,
        submission_file: Path,
        dataset_name: str,
        benchmark_name: str
    ) -> Dict:
        """
        评估单个提交文件

        Args:
            submission_file: 提交文件路径
            dataset_name: 数据集名称
            benchmark_name: benchmark名称

        Returns:
            评估结果
        """
        try:
            # 尝试对齐列名（如果需要）
            aligned_file = await self._align_submission_columns(submission_file, dataset_name)

            # 如果对齐成功，使用对齐后的文件；否则使用原文件
            file_to_evaluate = aligned_file if aligned_file else submission_file

            # 构建mlebench命令
            cmd = [
                self.mlebench_command,
                "grade-sample",
                str(file_to_evaluate),
                dataset_name
            ]

            self.logger.info(f"执行评估命令: {' '.join(cmd)}")

            # 执行命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )

            # 原封不动显示命令输出
            self.logger.info("=" * 80)
            self.logger.info("=== mlebench评估命令原始输出 ===")
            if result.stdout:
                self.logger.info("STDOUT:")
                self.logger.info(result.stdout)
            if result.stderr:
                self.logger.info("STDERR:")
                self.logger.info(result.stderr)
            self.logger.info(f"返回码: {result.returncode}")
            self.logger.info("=" * 80)

            if result.returncode != 0:
                return {
                    "status": "error",
                    "error": f"mlebench命令执行失败，退出码: {result.returncode}",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "valid_submission": False
                }

            # 解析输出
            return await self._parse_mlebench_output(result.stdout, result.stderr, benchmark_name)

        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "error": "评估超时",
                "message": "mlebench评估命令执行超过5分钟",
                "valid_submission": False
            }
        except Exception as e:
            self.logger.error(f"评估单个文件异常: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "status": "error",
                "error": str(e),
                "message": f"评估单个文件失败: {str(e)}",
                "valid_submission": False
            }

    async def _parse_mlebench_output(self, stdout: str, stderr: str, benchmark_name: str) -> Dict:
        """
        使用LLM解析评估命令的输出

        Args:
            stdout: 标准输出
            stderr: 标准错误
            benchmark_name: benchmark名称

        Returns:
            解析后的结果
        """
        try:
            # 构建LLM提示，让它提取所有信息
            prompt = build_mlebench_output_parsing_prompt(stdout, stderr, benchmark_name)

            # 调用LLM解析
            response = await self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=2000
            )

            self.logger.info("=" * 80)
            self.logger.info("=== LLM解析评估输出 ===")
            self.logger.info(f"LLM返回内容: {response}")
            self.logger.info("=" * 80)

            # 多种方式提取JSON
            parsed_result = self._extract_json_from_llm_response(response)

            if not parsed_result:
                self.logger.error(f"LLM返回无法解析为JSON: {response}")
                return {
                    "status": "error",
                    "error": "LLM解析输出失败",
                    "stdout": stdout,
                    "stderr": stderr,
                    "valid_submission": False,
                    "failure_type": "unknown",
                    "requires_code_regeneration": False
                }

            self.logger.info(f"解析后的结果: {json.dumps(parsed_result, ensure_ascii=False, indent=2)}")
            self.logger.info(f"requires_code_regeneration: {parsed_result.get('requires_code_regeneration')}")
            self.logger.info(f"failure_type: {parsed_result.get('failure_type')}")

            # 构建返回结果
            evaluation_success = parsed_result.get("evaluation_success", False)
            score = parsed_result.get("score")
            valid_submission = parsed_result.get("valid_submission", False)

            # 二次安全检查：确保requires_code_regeneration的正确性
            requires_regeneration = parsed_result.get("requires_code_regeneration", False)
            failure_type = parsed_result.get("failure_type", "unknown")

            # 强制规则1：如果valid_submission=false且score=null，必须重新生成代码
            if not valid_submission and score is None:
                requires_regeneration = True
                if failure_type == "unknown":
                    # 进一步检查stderr来确定失败类型
                    if "Invalid submission" in stderr or "id" in stderr.lower() or "column" in stderr.lower():
                        failure_type = "file_format_error"
                    else:
                        failure_type = "code_logic_error"
                self.logger.warning(f"安全检查：valid_submission=false且score=null，强制设置requires_code_regeneration=true")
                self.logger.warning(f"失败类型: {failure_type}")

            # 强制规则2：结果合理性检查 - 即使valid_submission=true，如果分数明显不合理也需要重新生成
            if valid_submission and score is not None:
                # 从full_report中获取阈值信息
                full_report = parsed_result.get("full_report", {})
                is_lower_better = full_report.get("is_lower_better", True)
                median_threshold = full_report.get("median_threshold")

                if median_threshold is not None:
                    # 计算分数与中位数的比例
                    if is_lower_better:
                        # 分数越低越好的情况
                        # 如果分数比中位数差10倍以上，认为不合理
                        if score > median_threshold * 10:
                            requires_regeneration = True
                            failure_type = "code_logic_error"
                            evaluation_success = False
                            self.logger.warning(f"安全检查：分数{score}远差于中位数{median_threshold}（差{score/median_threshold:.1f}倍），认为结果不合理")
                            self.logger.warning(f"强制设置requires_code_regeneration=true")
                    else:
                        # 分数越高越好的情况
                        # 如果分数比中位数差10倍以上，认为不合理
                        if score < median_threshold / 10:
                            requires_regeneration = True
                            failure_type = "code_logic_error"
                            evaluation_success = False
                            self.logger.warning(f"安全检查：分数{score}远差于中位数{median_threshold}（差{median_threshold/score:.1f}倍），认为结果不合理")
                            self.logger.warning(f"强制设置requires_code_regeneration=true")

            # 判断整体状态（注意：evaluation_success可能在合理性检查中被修改）
            if evaluation_success and score is not None and valid_submission:
                status = "success"
                gold_medal = parsed_result.get("gold_medal", False)
                silver_medal = parsed_result.get("silver_medal", False)
                bronze_medal = parsed_result.get("bronze_medal", False)
                above_median = parsed_result.get("above_median", False)

                if gold_medal:
                    message = f"恭喜！获得金牌！分数: {score}"
                elif silver_medal:
                    message = f"获得银牌！分数: {score}"
                elif bronze_medal:
                    message = f"获得铜牌！分数: {score}"
                elif above_median:
                    message = f"超过中位数！分数: {score}"
                else:
                    message = f"评估完成，分数: {score}"
            else:
                status = "error"
                # 如果是因为结果不合理导致的error，使用特殊的错误消息
                if requires_regeneration and failure_type == "code_logic_error" and score is not None:
                    message = f"代码逻辑错误：分数{score}严重偏离预期，需要重新生成代码"
                else:
                    message = parsed_result.get("error_message", "评估失败")

            return {
                "status": status,
                "score": score,
                "gold_medal": parsed_result.get("gold_medal", False),
                "silver_medal": parsed_result.get("silver_medal", False),
                "bronze_medal": parsed_result.get("bronze_medal", False),
                "any_medal": parsed_result.get("any_medal", False),
                "above_median": parsed_result.get("above_median", False),
                "valid_submission": valid_submission,
                "message": message,
                "full_report": parsed_result.get("full_report", {}),
                "raw_output": stdout,
                "raw_stderr": stderr,
                "failure_type": failure_type,
                "requires_code_regeneration": requires_regeneration,
                "error_message": parsed_result.get("error_message", "")
            }

        except Exception as e:
            self.logger.error(f"解析输出异常: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "status": "error",
                "error": f"解析输出失败: {str(e)}",
                "stdout": stdout,
                "stderr": stderr,
                "valid_submission": False,
                "failure_type": "unknown",
                "requires_code_regeneration": False
            }

    def _extract_json_from_llm_response(self, response: str) -> Optional[Dict]:
        """
        从LLM响应中提取JSON，支持多种格式

        Args:
            response: LLM响应文本

        Returns:
            解析后的字典，失败返回None
        """
        # 方法1: 直接解析
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        # 方法2: 提取```json...```标记的内容
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 方法3: 提取```...```标记的内容
        code_match = re.search(r'```\s*(\{.*?\})\s*```', response, re.DOTALL)
        if code_match:
            try:
                return json.loads(code_match.group(1))
            except json.JSONDecodeError:
                pass

        # 方法4: 查找第一个完整的JSON对象
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
                            break

        return None

    async def _align_submission_columns(
        self,
        submission_file: Path,
        dataset_name: str
    ) -> Optional[Path]:
        """
        尝试对齐submission文件的列名到标准格式

        Args:
            submission_file: 用户生成的提交文件
            dataset_name: 数据集名称

        Returns:
            对齐后的临时文件路径，如果不需要对齐或对齐失败则返回None
        """
        try:
            import pandas as pd
            import tempfile

            # 读取用户的submission文件
            user_df = pd.read_csv(submission_file)
            user_columns = set(user_df.columns)

            self.logger.info(f"用户submission列名: {list(user_df.columns)}")

            # 尝试从.cache中找到sample_submission文件
            cache_base = Path.home() / "Desktop" / "InterAgentV2" / ".cache" / "mle-bench" / "data" / dataset_name / "prepared" / "public"

            # 可能的sample submission文件名
            possible_names = [
                "sample_submission.csv",
                f"{dataset_name}_sample_submission.csv"
            ]

            # 也可能在子目录中
            sample_files = []
            if cache_base.exists():
                sample_files = list(cache_base.glob("*sample*submission*.csv"))

            if not sample_files:
                self.logger.info("未找到sample_submission文件，跳过列名对齐")
                return None

            # 读取标准格式
            sample_file = sample_files[0]
            self.logger.info(f"找到sample submission: {sample_file}")

            standard_df = pd.read_csv(sample_file, nrows=5)  # 只读取前几行了解格式
            standard_columns = list(standard_df.columns)

            self.logger.info(f"标准列名: {standard_columns}")

            # 检查是否需要对齐
            if set(user_df.columns) == set(standard_columns):
                self.logger.info("列名已匹配，无需对齐")
                return None

            # 尝试智能映射
            aligned_df = self._try_align_columns(user_df, standard_columns)

            if aligned_df is None:
                self.logger.warning("无法智能对齐列名")
                return None

            # 创建临时文件保存对齐后的数据
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.csv',
                delete=False,
                dir=submission_file.parent
            ) as tmp_file:
                aligned_df.to_csv(tmp_file.name, index=False)
                tmp_path = Path(tmp_file.name)
                self.logger.info(f"创建对齐后的临时文件: {tmp_path}")
                self.logger.info(f"对齐后列名: {list(aligned_df.columns)}")
                return tmp_path

        except Exception as e:
            self.logger.warning(f"列名对齐失败: {e}")
            import traceback
            self.logger.warning(traceback.format_exc())
            return None

    def _try_align_columns(
        self,
        user_df: 'pd.DataFrame',
        standard_columns: List[str]
    ) -> Optional['pd.DataFrame']:
        """
        尝试将用户DataFrame的列对齐到标准格式

        Args:
            user_df: 用户的DataFrame
            standard_columns: 标准列名列表

        Returns:
            对齐后的DataFrame，如果无法对齐则返回None
        """
        import pandas as pd

        user_columns = list(user_df.columns)
        aligned_df = pd.DataFrame()

        # 策略1: 直接重命名（大小写不敏感匹配）
        column_mapping = {}
        for std_col in standard_columns:
            for user_col in user_columns:
                if std_col.lower() == user_col.lower():
                    column_mapping[user_col] = std_col
                    break

        if len(column_mapping) == len(standard_columns):
            self.logger.info(f"使用策略1：直接重命名 {column_mapping}")
            return user_df.rename(columns=column_mapping)[standard_columns]

        # 策略2: 合成id列（如果标准是id列，用户有sentence_id和token_id）
        if 'id' in standard_columns and 'id' not in user_columns:
            # 检查是否有sentence_id和token_id
            has_sentence_id = any('sentence' in col.lower() for col in user_columns)
            has_token_id = any('token' in col.lower() for col in user_columns)

            if has_sentence_id and has_token_id:
                # 找到具体的列名
                sentence_col = next((col for col in user_columns if 'sentence' in col.lower()), None)
                token_col = next((col for col in user_columns if 'token' in col.lower()), None)

                if sentence_col and token_col:
                    self.logger.info(f"使用策略2：合成id列 ({sentence_col}_{token_col})")
                    # 创建id列
                    aligned_df['id'] = user_df[sentence_col].astype(str) + '_' + user_df[token_col].astype(str)

                    # 映射其他列
                    for std_col in standard_columns:
                        if std_col == 'id':
                            continue
                        # 尝试找到匹配的列
                        for user_col in user_columns:
                            if std_col.lower() == user_col.lower():
                                aligned_df[std_col] = user_df[user_col]
                                break

                    # 检查是否所有标准列都有了
                    if set(aligned_df.columns) == set(standard_columns):
                        return aligned_df[standard_columns]

        # 策略3: 拆分id列（如果用户有id，标准需要sentence_id和token_id）
        if 'id' in user_columns and 'id' not in standard_columns:
            has_sentence_in_std = any('sentence' in col.lower() for col in standard_columns)
            has_token_in_std = any('token' in col.lower() for col in standard_columns)

            if has_sentence_in_std and has_token_in_std:
                self.logger.info("使用策略3：拆分id列")
                # 尝试拆分id
                try:
                    split_ids = user_df['id'].str.split('_', n=1, expand=True)
                    sentence_col = next((col for col in standard_columns if 'sentence' in col.lower()), None)
                    token_col = next((col for col in standard_columns if 'token' in col.lower()), None)

                    if sentence_col and token_col:
                        aligned_df[sentence_col] = split_ids[0]
                        aligned_df[token_col] = split_ids[1]

                        # 映射其他列
                        for std_col in standard_columns:
                            if 'sentence' in std_col.lower() or 'token' in std_col.lower():
                                continue
                            for user_col in user_columns:
                                if std_col.lower() == user_col.lower():
                                    aligned_df[std_col] = user_df[user_col]
                                    break

                        if set(aligned_df.columns) == set(standard_columns):
                            return aligned_df[standard_columns]
                except Exception as e:
                    self.logger.warning(f"拆分id列失败: {e}")

        self.logger.warning("所有对齐策略都失败")
        return None
