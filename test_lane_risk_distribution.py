"""测试车道线风险分布：单层超高斯 vs 窄核+宽肩双层曲线

该脚本用于可视化和对比两种风险分布的效果，帮助调整参数。
"""

import math
import numpy as np
import matplotlib.pyplot as plt


def one_dimensional_risk(distance: float, sigma: float) -> float:
    """标准一维高斯风险"""
    sigma = max(float(sigma), 1e-6)
    return float(math.exp(-(max(distance, 0.0) ** 2) / (2.0 * sigma ** 2)))


def super_gaussian_risk_1d(distance: float, sigma: float, beta: float) -> float:
    """一维超髙斯风险"""
    sigma = max(float(sigma), 1e-6)
    beta = max(float(beta), 1e-6)
    exponent = -((max(distance, 0.0) ** 2) / (sigma ** 2)) ** beta
    return float(math.exp(max(exponent, -80.0)))


def lane_line_risk_1d(
    distance: float,
    sigma: float,
    beta: float,
    *,
    core_sigma_scale: float = 0.45,
    shoulder_sigma_scale: float = 2.0,
    shoulder_weight: float = 0.12,
) -> float:
    """窄核+宽肩双层车道线风险。

    - core: 只在线心附近给出强惩罚
    - shoulder: 提供更宽但更弱的预警范围
    """
    sigma = max(float(sigma), 1e-6)
    beta = max(float(beta), 1e-6)
    shoulder_weight = min(max(float(shoulder_weight), 0.0), 1.0)
    core_weight = 1.0 - shoulder_weight
    core_sigma = max(sigma * float(core_sigma_scale), 1e-6)
    shoulder_sigma = max(sigma * float(shoulder_sigma_scale), 1e-6)
    core = super_gaussian_risk_1d(distance, core_sigma, beta)
    shoulder = one_dimensional_risk(distance, shoulder_sigma)
    return core_weight * core + shoulder_weight * shoulder


def plot_comparison():
    """绘制单层超高斯和双层分布的对比图"""
    # 设置距离范围（从车道中心到边界）
    distances = np.linspace(0, 2.0, 200)  # 0到2米
    
    # 配置参数
    sigma = 0.5
    beta = 2.0
    
    plt.figure(figsize=(12, 8))
    
    cases = [
        ("旧版单层超高斯", dict(kind="legacy")),
        ("新版双层曲线（默认）", dict(kind="dual", core_sigma_scale=0.45, shoulder_sigma_scale=2.0, shoulder_weight=0.12)),
        ("更宽预警肩部", dict(kind="dual", core_sigma_scale=0.45, shoulder_sigma_scale=2.6, shoulder_weight=0.12)),
        ("更轻肩部", dict(kind="dual", core_sigma_scale=0.45, shoulder_sigma_scale=2.0, shoulder_weight=0.08)),
    ]
    for label, cfg in cases:
        if cfg["kind"] == "legacy":
            risks = [super_gaussian_risk_1d(d, sigma, beta) for d in distances]
        else:
            risks = [
                lane_line_risk_1d(
                    d,
                    sigma,
                    beta,
                    core_sigma_scale=cfg["core_sigma_scale"],
                    shoulder_sigma_scale=cfg["shoulder_sigma_scale"],
                    shoulder_weight=cfg["shoulder_weight"],
                )
                for d in distances
            ]
        plt.plot(distances, risks, linewidth=2.5, label=label)
    
    # 添加参考线
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    plt.axvline(x=0.25, color='red', linestyle=':', alpha=0.5, linewidth=1.5, label='强惩罚核大致范围')
    plt.axvline(x=0.9, color='orange', linestyle=':', alpha=0.5, linewidth=1.5, label='宽肩预警仍可见范围')
    
    # 标注关键区域
    plt.fill_between([0, 0.2], 0, 1, alpha=0.10, color='red', label='贴线高惩罚区')
    plt.fill_between([0.2, 0.8], 0, 1, alpha=0.08, color='yellow', label='宽肩低强度预警区')
    plt.fill_between([0.8, 2.0], 0, 1, alpha=0.05, color='green', label='几乎不干扰换道')
    
    plt.xlabel('距车道边缘的距离 (米)', fontsize=12, fontweight='bold')
    plt.ylabel('风险值', fontsize=12, fontweight='bold')
    plt.title('车道线风险分布对比：单层超高斯 vs 双层窄核+宽肩', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 2.0)
    plt.ylim(0, 1.05)
    
    # 添加文本说明
    textstr = '调整策略:\n• core_sigma_scale 控制强惩罚核宽度\n• shoulder_sigma_scale 控制预警范围\n• shoulder_weight 控制“宽但弱”的肩部强度'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('/home/ac/@Lyz-Code/safeRL-metadrive/lane_risk_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ 对比图已保存至: lane_risk_comparison.png")
    plt.show()


def analyze_parameters():
    """分析不同参数组合的效果"""
    print("\n" + "="*70)
    print("车道线风险参数分析")
    print("="*70)
    
    test_cases = [
        {"name": "旧版单层超高斯", "kind": "legacy", "sigma": 0.5, "beta": 2.0},
        {"name": "新版默认", "kind": "dual", "sigma": 0.5, "beta": 2.0, "core_sigma_scale": 0.45, "shoulder_sigma_scale": 2.0, "shoulder_weight": 0.12},
        {"name": "更宽肩部", "kind": "dual", "sigma": 0.5, "beta": 2.0, "core_sigma_scale": 0.45, "shoulder_sigma_scale": 2.6, "shoulder_weight": 0.12},
        {"name": "更轻肩部", "kind": "dual", "sigma": 0.5, "beta": 2.0, "core_sigma_scale": 0.45, "shoulder_sigma_scale": 2.0, "shoulder_weight": 0.08},
    ]
    
    distances_of_interest = [0.0, 0.1, 0.2, 0.25, 0.5, 0.8, 1.0]
    
    print(f"\n{'配置':<25} {'距离(米)':<10} " + " ".join([f"{d:>6}" for d in distances_of_interest]))
    print("-" * 90)
    
    for case in test_cases:
        sigma = case["sigma"]
        beta = case["beta"]
        name = case["name"]
        
        risks = []
        for d in distances_of_interest:
            if case["kind"] == "legacy":
                risk = super_gaussian_risk_1d(d, sigma, beta)
            else:
                risk = lane_line_risk_1d(
                    d,
                    sigma,
                    beta,
                    core_sigma_scale=case["core_sigma_scale"],
                    shoulder_sigma_scale=case["shoulder_sigma_scale"],
                    shoulder_weight=case["shoulder_weight"],
                )
            risks.append(f"{risk:.3f}")
        
        print(f"{name:<25} {'风险值':<10} " + " ".join([f"{r:>6}" for r in risks]))
    
    print("\n" + "="*70)
    print("关键观察:")
    print("  • 新版双层曲线：0.2~0.3m 内仍保持明显惩罚，但 0.5m 后主要剩低强度肩部")
    print("  • 增大 shoulder_sigma_scale: 预警更早，但不会像直接增大单层 sigma 那样整段都偏高")
    print("  • 减小 shoulder_weight: 更不干扰正常靠边换道")
    print("  • 调 core_sigma_scale: 专门控制‘只有贴线才重罚’这件事")
    print("="*70 + "\n")


if __name__ == "__main__":
    print("开始测试车道线风险分布...")
    analyze_parameters()
    plot_comparison()
    print("\n✅ 测试完成！请查看生成的图表和数据分析。")
