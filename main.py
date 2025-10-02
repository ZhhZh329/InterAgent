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
        
        # 最终结果摘要（详细过程日志已在执行中实时输出）
        logging.info("\n" + "=" * 80)
        logging.info("=== 任务执行完成 ===")
        logging.info(f"最终状态: {research_result['status']}")
        logging.info(f"结果消息: {research_result['message']}")
        logging.info(f"工作目录: {research_result.get('workspace_dir', 'N/A')}")
        logging.info("=" * 80)

    except Exception as e:
        logging.error(f"错误: {e}")


if __name__ == "__main__":
    asyncio.run(main())