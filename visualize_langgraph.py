"""
可视化LangGraph工作流
"""
from src.InterAgent.inter_agent import InterAgent


def visualize_workflow():
    """打印LangGraph工作流结构"""
    agent = InterAgent()
    workflow = agent.build_research_graph()

    print("=" * 80)
    print("LangGraph研究任务工作流可视化")
    print("=" * 80)

    print("\n节点列表 (12个):")
    print("-" * 80)
    nodes = [
        "1.  setup                    - 初始化工作目录",
        "2.  probe_structure          - 探测文件结构",
        "3.  generate_data_loader     - 生成数据加载代码",
        "4.  test_data_loader         - 测试数据加载",
        "5.  debug_data_loader        - Debug数据加载代码",
        "6.  init_analysis            - 初步分析",
        "7.  generate_research_code   - 生成研究代码",
        "8.  test_research_code       - 测试研究代码（小样本）",
        "9.  auto_fix_code            - 自动修复代码问题",
        "10. run_full_dataset         - 运行完整数据集",
        "11. optimize_full_run        - 优化并重试完整运行",
        "12. evaluate                 - 评估提交结果",
    ]

    for node in nodes:
        print(f"  {node}")

    print("\n工作流路径:")
    print("-" * 80)
    print("""
  START
    │
    ↓
  [setup] ──→ [probe_structure] ──→ [generate_data_loader] ──→ [test_data_loader]
                                                                       │
                                                           ┌───────────┴───────────┐
                                                           │                       │
                                                       [成功]                   [失败]
                                                           │                       │
                                                           │                       ↓
                                                           │              [debug_data_loader]
                                                           │                       │
                                                           └───────────┬───────────┘
                                                                       │
                                                                       ↓
                                                               [init_analysis]
                                                                       │
                                                           ┌───────────┴───────────┐
                                                           │                       │
                                                       [成功]                   [失败]
                                                           │                       │
                                                           ↓                       ↓
                                                [generate_research_code]         [END]
                                                           │
                                                           ↓
                                                  [test_research_code]
                                                           │
                                                ┌──────────┴──────────┐
                                                │                     │
                                            [成功]                 [失败]
                                                │                     │
                                                │                     ↓
                                                │             [auto_fix_code]
                                                │                     │
                                                │         ┌───────────┴──────────┐
                                                │         │                      │
                                                │     [成功]                  [失败]
                                                │         │                      │
                                                └─────────┤                      ↓
                                                          │                    [END]
                                                          ↓
                                                  [run_full_dataset]
                                                          │
                                                ┌─────────┴─────────┐
                                                │                   │
                                            [成功]               [失败]
                                                │                   │
                                                │                   ↓
                                                │         [optimize_full_run]
                                                │                   │
                                                │       ┌───────────┴──────────┐
                                                │       │                      │
                                                │   [成功]                  [失败]
                                                │       │                      │
                                                └───────┤                      ↓
                                                        │                    [END]
                                                        ↓
                                                   [evaluate]
                                                        │
                                                        ↓
                                                      [END]
    """)

    print("\n重试机制封装:")
    print("-" * 80)
    print("""
  1. auto_fix_code 节点内部:
     - 依赖安装重试: 最多3次
     - 代码Debug重试: 最多3次
     - 自动选择策略（ModuleNotFoundError → 安装，其他错误 → Debug）

  2. optimize_full_run 节点内部:
     - 完整运行重试: 最多3次
     - 根据失败类型优化（超时/OOM/其他）
     - 自动重新生成、测试、运行、评估

  注意：重试循环已封装在节点内部，图的层级保持扁平
    """)

    print("\n嵌套层级对比:")
    print("-" * 80)
    print("""
  Legacy版本:
    run_research_task()
      ├─ if test_result error (Level 1)
      │   └─ if debug success (Level 2)
      │       └─ if test success (Level 3)
      │           └─ while retry < max (Level 4)
      │               └─ if is_module_error (Level 5)
      │                   └─ if install success (Level 6)
      │
      └─ [更多嵌套逻辑...]

    最大嵌套: 6层

  LangGraph版本:
    run_research_task()
      └─ workflow.ainvoke(initial_state)  (Level 1)

    节点函数:
      └─ await self._existing_method()  (Level 1)

    路由函数:
      └─ if condition: return "next_node"  (Level 1)

    最大嵌套: 2层 (在封装的retry方法内部)
    """)

    print("\n代码行数对比:")
    print("-" * 80)
    print("""
  Legacy run_research_task:     ~450 行
  LangGraph run_research_task:   ~40 行

  减少: 91%

  但总代码量增加约750行（节点+路由+graph构建）
  用户反馈: "行数多了无所谓，langgraph增加的阅读成本也无所谓"
  关键优势: 可读性和可维护性大幅提升
    """)

    print("\n" + "=" * 80)
    print("可视化完成！")
    print("=" * 80)


if __name__ == "__main__":
    visualize_workflow()
