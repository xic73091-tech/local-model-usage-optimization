"""
多显存优化器

将6GB显存的优化方案推广到所有显存配置(4GB/6GB/8GB/12GB/16GB/24GB)。
自动生成各显存的最优配置，并提供对比分析。

支持的显存配置：
- 4GB: 入门级显卡 (GTX 1650, RTX 3050)
- 6GB: 主流显卡 (RTX 2060, RTX 3060)
- 8GB: 中高端显卡 (RTX 3060Ti, RTX 4060)
- 12GB: 高端显卡 (RTX 3060 12G, RTX 4070)
- 16GB: 专业显卡 (RTX 4060Ti 16G, RTX 4080)
- 24GB: 旗舰显卡 (RTX 3090, RTX 4090)
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .vram_optimizer import VRAMOptimizer, OptimizationTarget, OptimizationResult

logger = logging.getLogger(__name__)


# ============================================================
# 常量定义
# ============================================================

# 标准显存配置
VRAM_CONFIGS = {
    4: {"name": "4GB", "description": "入门级显卡", "examples": "GTX 1650, RTX 3050"},
    6: {"name": "6GB", "description": "主流显卡", "examples": "RTX 2060, RTX 3060"},
    8: {"name": "8GB", "description": "中高端显卡", "examples": "RTX 3060Ti, RTX 4060"},
    12: {"name": "12GB", "description": "高端显卡", "examples": "RTX 3060 12G, RTX 4070"},
    16: {"name": "16GB", "description": "专业显卡", "examples": "RTX 4060Ti 16G, RTX 4080"},
    24: {"name": "24GB", "description": "旗舰显卡", "examples": "RTX 3090, RTX 4090"},
}

# 标准模型大小
MODEL_SIZES = [3.0, 7.0, 13.0, 30.0, 70.0]


# ============================================================
# 数据类
# ============================================================

@dataclass
class VRAMProfile:
    """显存配置画像"""
    vram_gb: int
    name: str
    description: str
    examples: str
    # 各模型的最优配置
    model_configs: Dict[str, OptimizationResult] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "vram_gb": self.vram_gb,
            "name": self.name,
            "description": self.description,
            "examples": self.examples,
            "model_configs": {
                model: result.to_dict()
                for model, result in self.model_configs.items()
            },
        }


@dataclass
class ComparisonMatrix:
    """对比矩阵"""
    vram_configs: List[int]
    model_sizes: List[float]
    # 结果矩阵 [vram_idx][model_idx]
    results: List[List[OptimizationResult]] = field(default_factory=list)

    def get_result(self, vram_gb: int, model_size_b: float) -> Optional[OptimizationResult]:
        """获取指定配置的结果"""
        try:
            vram_idx = self.vram_configs.index(vram_gb)
            model_idx = self.model_sizes.index(model_size_b)
            return self.results[vram_idx][model_idx]
        except (ValueError, IndexError):
            return None


# ============================================================
# 核心类
# ============================================================

class MultiVRAMOptimizer:
    """多显存优化器

    将优化方案推广到所有显存配置。

    用法：
        optimizer = MultiVRAMOptimizer()

        # 获取所有显存的配置
        profiles = optimizer.generate_all_profiles()

        # 获取对比矩阵
        matrix = optimizer.get_comparison_matrix()

        # 导出配置
        optimizer.export_configs("configs.json")
    """

    def __init__(self, model_sizes: Optional[List[float]] = None):
        """初始化

        Args:
            model_sizes: 要优化的模型大小列表
        """
        self.model_sizes = model_sizes or MODEL_SIZES
        self.vram_configs = sorted(VRAM_CONFIGS.keys())

        # 创建各显存的优化器
        self._optimizers: Dict[int, VRAMOptimizer] = {}
        for vram_gb in self.vram_configs:
            self._optimizers[vram_gb] = VRAMOptimizer(vram_gb=vram_gb)

        logger.info(f"MultiVRAMOptimizer 已初始化: {len(self.vram_configs)}种显存, {len(self.model_sizes)}种模型")

    # ----------------------------------------------------------
    # 核心API
    # ----------------------------------------------------------

    def generate_all_profiles(self) -> Dict[int, VRAMProfile]:
        """生成所有显存配置的画像

        Returns:
            {vram_gb: VRAMProfile}
        """
        profiles = {}

        for vram_gb in self.vram_configs:
            config_info = VRAM_CONFIGS[vram_gb]
            profile = VRAMProfile(
                vram_gb=vram_gb,
                name=config_info["name"],
                description=config_info["description"],
                examples=config_info["examples"],
            )

            # 为每个模型生成最优配置
            for model_size in self.model_sizes:
                optimizer = self._optimizers[vram_gb]
                result = optimizer.optimize(model_size, OptimizationTarget.BALANCED)
                profile.model_configs[f"{model_size}B"] = result

            profiles[vram_gb] = profile

        return profiles

    def get_comparison_matrix(self) -> ComparisonMatrix:
        """获取对比矩阵

        Returns:
            ComparisonMatrix
        """
        matrix = ComparisonMatrix(
            vram_configs=self.vram_configs,
            model_sizes=self.model_sizes,
        )

        for vram_gb in self.vram_configs:
            row = []
            optimizer = self._optimizers[vram_gb]

            for model_size in self.model_sizes:
                result = optimizer.optimize(model_size, OptimizationTarget.BALANCED)
                row.append(result)

            matrix.results.append(row)

        return matrix

    def get_optimal_config(
        self,
        vram_gb: int,
        model_size_b: float,
        target: OptimizationTarget = OptimizationTarget.BALANCED,
    ) -> OptimizationResult:
        """获取指定配置的最优配置

        Args:
            vram_gb: 显存大小 (GB)
            model_size_b: 模型参数量 (B)
            target: 优化目标

        Returns:
            OptimizationResult
        """
        # 找到最接近的显存配置
        closest_vram = min(self.vram_configs, key=lambda x: abs(x - vram_gb))

        optimizer = self._optimizers[closest_vram]
        return optimizer.optimize(model_size_b, target)

    def find_best_model_for_vram(
        self,
        vram_gb: int,
        min_speed: float = 5.0,
        min_quality: float = 0.4,
    ) -> List[dict]:
        """为指定显存找最佳模型

        Args:
            vram_gb: 显存大小 (GB)
            min_speed: 最低速度要求 (tokens/s)
            min_quality: 最低质量要求

        Returns:
            可用模型列表
        """
        closest_vram = min(self.vram_configs, key=lambda x: abs(x - vram_gb))
        optimizer = self._optimizers[closest_vram]

        results = []

        for model_size in self.model_sizes:
            result = optimizer.optimize(model_size, OptimizationTarget.BALANCED)

            if (result.estimated_speed_tps >= min_speed and
                result.quality_score >= min_quality and
                result.estimated_vram_gb <= vram_gb):

                results.append({
                    "model_size_b": model_size,
                    "quantization": result.quantization.value,
                    "gpu_layers": result.layer_allocation.gpu_layers,
                    "total_layers": result.layer_allocation.total_layers,
                    "vram_gb": round(result.estimated_vram_gb, 2),
                    "speed_tps": round(result.estimated_speed_tps, 1),
                    "quality": round(result.quality_score, 3),
                })

        # 按速度排序
        results.sort(key=lambda x: x["speed_tps"], reverse=True)

        return results

    # ----------------------------------------------------------
    # 导出功能
    # ----------------------------------------------------------

    def export_configs(self, output_path: str) -> None:
        """导出所有配置到JSON文件

        Args:
            output_path: 输出文件路径
        """
        profiles = self.generate_all_profiles()

        export_data = {
            "version": "1.0",
            "generated_at": "2026-05-29",
            "vram_configs": {},
        }

        for vram_gb, profile in profiles.items():
            export_data["vram_configs"][str(vram_gb)] = profile.to_dict()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"配置已导出到: {output_path}")

    def export_markdown_table(self, output_path: str) -> None:
        """导出Markdown对比表

        Args:
            output_path: 输出文件路径
        """
        matrix = self.get_comparison_matrix()

        lines = [
            "# 多显存优化配置对比",
            "",
            "## 概览",
            "",
            "| 显存 | 可用模型 | 推荐模型 |",
            "|------|----------|----------|",
        ]

        for vram_gb in self.vram_configs:
            best_models = self.find_best_model_for_vram(vram_gb, min_speed=5.0)
            available = len(best_models)
            recommended = best_models[0]["model_size_b"] if best_models else "无"

            lines.append(f"| {vram_gb}GB | {available}个 | {recommended}B |")

        lines.extend([
            "",
            "## 详细配置",
            "",
        ])

        # 为每个显存生成表格
        for vram_gb in self.vram_configs:
            lines.extend([
                f"### {vram_gb}GB 显存",
                "",
                "| 模型 | 量化 | GPU层 | 显存(GB) | 速度(t/s) | 质量 | 可用性 |",
                "|------|------|-------|----------|-----------|------|--------|",
            ])

            for model_size in self.model_sizes:
                result = matrix.get_result(vram_gb, model_size)
                if result:
                    fits = "YES" if result.estimated_vram_gb <= vram_gb else "NO"
                    speed = result.estimated_speed_tps
                    quality = result.quality_score

                    # 判断可用性
                    if speed >= 20 and quality >= 0.7:
                        usability = "PERFECT"
                    elif speed >= 10 and quality >= 0.5:
                        usability = "GOOD"
                    elif speed >= 5:
                        usability = "OK"
                    else:
                        usability = "SLOW"

                    lines.append(
                        f"| {model_size}B | {result.quantization.value} | "
                        f"{result.layer_allocation.gpu_layers}/{result.layer_allocation.total_layers} | "
                        f"{result.estimated_vram_gb:.2f} | {speed:.1f} | "
                        f"{quality:.3f} | {usability} |"
                    )

            lines.append("")

        # 推荐配置
        lines.extend([
            "## 推荐配置",
            "",
            "| 显存 | 推荐模型 | 量化 | 速度 | 质量 |",
            "|------|----------|------|------|------|",
        ])

        for vram_gb in self.vram_configs:
            best_models = self.find_best_model_for_vram(vram_gb, min_speed=10.0, min_quality=0.5)
            if best_models:
                best = best_models[0]
                lines.append(
                    f"| {vram_gb}GB | {best['model_size_b']}B | "
                    f"{best['quantization']} | {best['speed_tps']} t/s | "
                    f"{best['quality']:.2f} |"
                )
            else:
                lines.append(f"| {vram_gb}GB | 无满足条件的模型 | - | - | - |")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        logger.info(f"Markdown表格已导出到: {output_path}")

    # ----------------------------------------------------------
    # 对比分析
    # ----------------------------------------------------------

    def compare_vram_for_model(self, model_size_b: float) -> List[dict]:
        """对比不同显存对同一模型的配置

        Args:
            model_size_b: 模型参数量 (B)

        Returns:
            对比结果列表
        """
        results = []

        for vram_gb in self.vram_configs:
            optimizer = self._optimizers[vram_gb]
            result = optimizer.optimize(model_size_b, OptimizationTarget.BALANCED)

            results.append({
                "vram_gb": vram_gb,
                "quantization": result.quantization.value,
                "gpu_layers": result.layer_allocation.gpu_layers,
                "total_layers": result.layer_allocation.total_layers,
                "vram_used_gb": round(result.estimated_vram_gb, 2),
                "speed_tps": round(result.estimated_speed_tps, 1),
                "quality": round(result.quality_score, 3),
                "fits": result.estimated_vram_gb <= vram_gb,
            })

        return results

    def compare_models_for_vram(self, vram_gb: int) -> List[dict]:
        """对比同一显存下不同模型的配置

        Args:
            vram_gb: 显存大小 (GB)

        Returns:
            对比结果列表
        """
        results = []
        optimizer = self._optimizers.get(vram_gb) or self._optimizers[min(self._optimizers.keys())]

        for model_size in self.model_sizes:
            result = optimizer.optimize(model_size, OptimizationTarget.BALANCED)

            results.append({
                "model_size_b": model_size,
                "quantization": result.quantization.value,
                "gpu_layers": result.layer_allocation.gpu_layers,
                "total_layers": result.layer_allocation.total_layers,
                "vram_used_gb": round(result.estimated_vram_gb, 2),
                "speed_tps": round(result.estimated_speed_tps, 1),
                "quality": round(result.quality_score, 3),
                "fits": result.estimated_vram_gb <= vram_gb,
            })

        return results

    # ----------------------------------------------------------
    # 打印功能
    # ----------------------------------------------------------

    def print_comparison_table(self) -> None:
        """打印对比表"""
        matrix = self.get_comparison_matrix()

        print("=" * 100)
        print("多显存优化配置对比")
        print("=" * 100)

        # 表头
        header = f"{'模型':<10}"
        for vram_gb in self.vram_configs:
            header += f" {vram_gb}GB显存".ljust(15)
        print(header)
        print("-" * 100)

        # 数据行
        for model_size in self.model_sizes:
            row = f"{model_size}B{'':<7}"
            for vram_gb in self.vram_configs:
                result = matrix.get_result(vram_gb, model_size)
                if result:
                    fits = "YES" if result.estimated_vram_gb <= vram_gb else "NO"
                    speed = result.estimated_speed_tps
                    cell = f"{speed:.0f}t/s {fits}"
                    row += f" {cell:<14}"
                else:
                    row += f" {'N/A':<14}"
            print(row)

        print("=" * 100)

    def print_recommendations(self) -> None:
        """打印推荐配置"""
        print("\n" + "=" * 80)
        print("各显存推荐配置")
        print("=" * 80)

        for vram_gb in self.vram_configs:
            best_models = self.find_best_model_for_vram(vram_gb, min_speed=10.0, min_quality=0.5)

            print(f"\n{vram_gb}GB 显存:")
            if best_models:
                best = best_models[0]
                print(f"  推荐模型: {best['model_size_b']}B")
                print(f"  量化: {best['quantization']}")
                print(f"  GPU层数: {best['gpu_layers']}/{best['total_layers']}")
                print(f"  显存占用: {best['vram_gb']} GB")
                print(f"  速度: {best['speed_tps']} tokens/s")
                print(f"  质量: {best['quality']:.3f}")
            else:
                print("  无满足条件的模型")

        print("=" * 80)


# ============================================================
# 便捷函数
# ============================================================

def generate_all_vram_configs(model_sizes: Optional[List[float]] = None) -> Dict[int, dict]:
    """生成所有显存配置

    Args:
        model_sizes: 模型大小列表

    Returns:
        {vram_gb: config_dict}
    """
    optimizer = MultiVRAMOptimizer(model_sizes)
    profiles = optimizer.generate_all_profiles()

    return {
        vram_gb: profile.to_dict()
        for vram_gb, profile in profiles.items()
    }


def export_comparison_table(output_path: str, model_sizes: Optional[List[float]] = None) -> None:
    """导出对比表

    Args:
        output_path: 输出路径
        model_sizes: 模型大小列表
    """
    optimizer = MultiVRAMOptimizer(model_sizes)
    optimizer.export_markdown_table(output_path)


# ============================================================
# 命令行入口
# ============================================================

def main():
    """命令行演示"""
    optimizer = MultiVRAMOptimizer()

    # 打印对比表
    optimizer.print_comparison_table()

    # 打印推荐配置
    optimizer.print_recommendations()

    # 导出配置
    optimizer.export_configs("vram_configs.json")
    optimizer.export_markdown_table("VRAM_CONFIGS.md")

    print("\n配置已导出到:")
    print("  - vram_configs.json")
    print("  - VRAM_CONFIGS.md")


if __name__ == "__main__":
    main()
