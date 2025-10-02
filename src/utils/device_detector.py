"""
设备检测工具 - 自动检测可用的GPU/MPS设备
支持CUDA (NVIDIA), MPS (Apple Silicon), ROCm (AMD)
"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def detect_compute_device() -> Dict[str, Any]:
    """
    检测可用的计算设备

    Returns:
        Dict包含设备信息:
        - device_type: 'cuda', 'mps', 'cpu'
        - device_name: 设备名称
        - device_count: 设备数量
        - framework_available: 可用框架列表
        - recommendation: 推荐的框架使用方式
    """
    device_info = {
        'device_type': 'cpu',
        'device_name': 'CPU',
        'device_count': 0,
        'framework_available': [],
        'recommendation': None,
        'pytorch_device': 'cpu',
        'tensorflow_device': '/CPU:0'
    }

    # 1. 检测PyTorch和CUDA
    try:
        import torch
        device_info['framework_available'].append('pytorch')

        # 检测CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            device_info['device_type'] = 'cuda'
            device_info['device_name'] = torch.cuda.get_device_name(0)
            device_info['device_count'] = torch.cuda.device_count()
            device_info['pytorch_device'] = 'cuda'
            device_info['recommendation'] = {
                'pytorch': "device = torch.device('cuda')",
                'usage': "model.to('cuda')"
            }
            logger.info(f"检测到CUDA设备: {device_info['device_name']} (共{device_info['device_count']}个)")

        # 检测MPS (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_info['device_type'] = 'mps'
            device_info['device_name'] = 'Apple Silicon GPU (MPS)'
            device_info['device_count'] = 1
            device_info['pytorch_device'] = 'mps'
            device_info['recommendation'] = {
                'pytorch': "device = torch.device('mps')",
                'usage': "model.to('mps')"
            }
            logger.info(f"检测到MPS设备: Apple Silicon GPU")
        else:
            logger.info("未检测到GPU，将使用CPU")
            device_info['recommendation'] = {
                'pytorch': "device = torch.device('cpu')",
                'usage': "model.to('cpu')"
            }
    except ImportError:
        logger.warning("PyTorch未安装，跳过PyTorch GPU检测")

    # 2. 检测TensorFlow和GPU
    try:
        import tensorflow as tf
        device_info['framework_available'].append('tensorflow')

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            if device_info['device_type'] == 'cpu':  # PyTorch未检测到GPU
                device_info['device_type'] = 'gpu'
                device_info['device_name'] = gpus[0].name
                device_info['device_count'] = len(gpus)
            device_info['tensorflow_device'] = '/GPU:0'
            if not device_info['recommendation']:
                device_info['recommendation'] = {
                    'tensorflow': "with tf.device('/GPU:0'):",
                    'usage': "自动使用GPU"
                }
            logger.info(f"TensorFlow检测到{len(gpus)}个GPU设备")
    except ImportError:
        logger.warning("TensorFlow未安装，跳过TensorFlow GPU检测")
    except Exception as e:
        logger.warning(f"TensorFlow GPU检测失败: {e}")

    return device_info


def get_device_config_string(device_info: Dict[str, Any]) -> str:
    """
    生成设备配置的代码字符串，用于在生成的代码中使用

    Args:
        device_info: detect_compute_device()返回的设备信息

    Returns:
        设备配置代码字符串
    """
    if device_info['device_type'] == 'cuda':
        return """
# GPU设备配置 (CUDA)
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
"""
    elif device_info['device_type'] == 'mps':
        return """
# GPU设备配置 (Apple Silicon MPS)
import torch
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("使用Apple Silicon GPU (MPS)")
else:
    device = torch.device('cpu')
    print("MPS不可用，使用CPU")
"""
    else:
        return """
# CPU设备配置
import torch
device = torch.device('cpu')
print("使用CPU")
"""


def get_device_recommendation_text(device_info: Dict[str, Any]) -> str:
    """
    生成人类可读的设备推荐文本

    Args:
        device_info: detect_compute_device()返回的设备信息

    Returns:
        推荐文本
    """
    if device_info['device_type'] == 'cuda':
        return f"检测到NVIDIA GPU: {device_info['device_name']}，建议使用PyTorch的CUDA加速: device = torch.device('cuda')"
    elif device_info['device_type'] == 'mps':
        return "检测到Apple Silicon GPU，建议使用PyTorch的MPS加速: device = torch.device('mps')"
    else:
        frameworks = ', '.join(device_info['framework_available']) if device_info['framework_available'] else '无'
        return f"未检测到GPU，将使用CPU运行 (可用框架: {frameworks})"


if __name__ == "__main__":
    # 测试设备检测
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("设备检测测试")
    print("=" * 60)

    device_info = detect_compute_device()

    print("\n设备信息:")
    print(f"  设备类型: {device_info['device_type']}")
    print(f"  设备名称: {device_info['device_name']}")
    print(f"  设备数量: {device_info['device_count']}")
    print(f"  可用框架: {', '.join(device_info['framework_available'])}")

    print("\n推荐配置:")
    print(get_device_recommendation_text(device_info))

    print("\n代码示例:")
    print(get_device_config_string(device_info))