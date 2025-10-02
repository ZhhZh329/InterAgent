import asyncio
import logging
from src.mle_bench.mle_file_agent import get_mle_dataset_info
from src.InterAgent.inter_agent import InterAgent
from src.utils.device_detector import detect_compute_device, get_device_recommendation_text

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def main():
    """主函数 - 测试MLE Bench数据集信息提取和InterAgent调用"""

    benchmark_name="mlebench"
    # 测试用的数据集名称
    dataset_name = "text-normalization-challenge-russian-language"
    # "jigsaw-toxic-comment-classification-challenge"
    # "denoising-dirty-documents"
    # "leaf-classification"
    # "spooky-author-identification"
    # "text-normalization-challenge-russian-language"
    # "random-acts-of-pizza"
    # "text-normalization-challenge-english-language"
    # "aerial-cactus-identification"
    # "nomad2018-predict-transparent-conductors"
    # # 可以修改为其他数据集

    try:
        # 步骤0: 检测计算设备
        logging.info("=== 检测计算设备 ===")
        device_info = detect_compute_device()
        logging.info(get_device_recommendation_text(device_info))

        logging.info(f"正在分析数据集: {dataset_name}")

        # 步骤1: 调用MLE文件代理获取数据集信息
        dataset_info = await get_mle_dataset_info(dataset_name)

        logging.info("=== 数据集信息 ===")
        logging.info(f"数据集名称: {dataset_info['dataset_name']}")
        logging.info(f"研究主题: {dataset_info['research_topic']}")
        logging.info(f"研究目标: {dataset_info['research_goal']}")
        logging.info(f"数据集路径: {dataset_info['dataset_path']}")

        # 步骤2: 调用InterAgent执行研究任务，传入设备信息
        logging.info("=== 启动InterAgent ===")
        inter_agent = InterAgent(device_info=device_info)
        
        research_result = await inter_agent.run_research_task(
            research_topic=dataset_info['research_topic'],
            research_goal=dataset_info['research_goal'],
            dataset_path=dataset_info['dataset_path'],
            benchmark_name=benchmark_name,
            dataset_name=dataset_name,
            debug=True  # 开启调试模式
        )
        
        logging.info("=== InterAgent执行结果 ===")
        logging.info(f"状态: {research_result['status']}")
        logging.info(f"消息: {research_result['message']}")
        logging.info(f"工作目录: {research_result.get('workspace_dir', 'N/A')}")
        
        # 显示文件结构分析结果
        if 'structure_info' in research_result:
            structure_info = research_result['structure_info']
            logging.info("=== 文件结构分析 ===")
            logging.info(f"分析状态: {structure_info['status']}")
            
            if 'analysis_result' in structure_info:
                analysis = structure_info['analysis_result']
                logging.info(f"训练集文件: {analysis.get('train_files', 'N/A')}")
                logging.info(f"测试集文件: {analysis.get('test_files', 'N/A')}")
                logging.info(f"数据格式: {analysis.get('data_format', 'N/A')}")
                logging.info(f"加载建议: {analysis.get('loading_suggestions', 'N/A')}")
        
        # 显示数据加载代码生成结果
        if 'data_loader_result' in research_result:
            data_result = research_result['data_loader_result']
            logging.info("=== 数据加载代码生成 ===")
            logging.info(f"生成状态: {data_result['status']}")
            if data_result['status'] == 'success':
                logging.info(f"生成文件: {data_result.get('generated_files', [])}")
        
        # 显示数据加载测试结果（无论debug模式如何都会显示）
        if 'test_result' in research_result and research_result['test_result']:
            test_result = research_result['test_result']
            logging.info("=== 数据加载测试结果 ===")
            logging.info(f"测试状态: {test_result['status']}")
            
            if test_result['status'] == 'success':
                logging.info("=== 数据结构信息 ===")
                if test_result.get('stdout'):
                    print(test_result['stdout'])
            else:
                logging.error(f"测试失败: {test_result.get('message', '')}")
                if test_result.get('stderr'):
                    logging.error(f"错误输出: {test_result['stderr']}")
        
        # 显示初步分析结果
        if 'init_analysis_result' in research_result and research_result['init_analysis_result']:
            init_analysis = research_result['init_analysis_result']
            logging.info("=== 初步分析 ===")
            logging.info(f"分析状态: {init_analysis['status']}")
            if init_analysis['status'] == 'success':
                logging.info("分析内容已在上方显示")
        
        # 显示研究代码生成结果
        if 'research_code_result' in research_result and research_result['research_code_result']:
            research_code = research_result['research_code_result']
            logging.info("=== 研究代码生成 ===")
            logging.info(f"生成状态: {research_code['status']}")
            if research_code['status'] == 'success':
                logging.info(f"生成文件: {research_code.get('generated_file', '')}")
            else:
                logging.error(f"生成失败: {research_code.get('message', '')}")

            # 显示重新生成和重试信息（如果有）
            if 'regeneration_count' in research_code and research_code['regeneration_count'] > 0:
                logging.info(f"研究代码重新生成次数: {research_code['regeneration_count']}")
            if 'total_retry_count' in research_code and research_code['total_retry_count'] > 0:
                logging.info(f"总重试次数（包括重新评估）: {research_code['total_retry_count']}")

            # 显示评估结果
            if 'full_run_result' in research_code and research_code['full_run_result']:
                full_run = research_code['full_run_result']
                if 'evaluation' in full_run and full_run['evaluation']:
                    eval_result = full_run['evaluation']
                    logging.info("=== 评估结果 ===")
                    logging.info(f"评估状态: {eval_result['status']}")
                    if eval_result['status'] == 'success':
                        logging.info(f"分数: {eval_result.get('score', 'N/A')}")
                        logging.info(f"金牌: {'是' if eval_result.get('gold_medal') else '否'}")
                        logging.info(f"银牌: {'是' if eval_result.get('silver_medal') else '否'}")
                        logging.info(f"铜牌: {'是' if eval_result.get('bronze_medal') else '否'}")
                        logging.info(f"超过中位数: {'是' if eval_result.get('above_median') else '否'}")
                        logging.info(f"提交有效: {'是' if eval_result.get('valid_submission') else '否'}")
                        logging.info(f"评估消息: {eval_result.get('message', '')}")
                    elif eval_result['status'] == 'skipped':
                        logging.info(f"评估跳过: {eval_result.get('message', '')}")
                    else:
                        logging.error(f"评估失败: {eval_result.get('error', 'unknown')}")
                        if 'stdout' in eval_result:
                            logging.info(f"评估输出: {eval_result['stdout']}")

    except Exception as e:
        logging.error(f"错误: {e}")


if __name__ == "__main__":
    asyncio.run(main())