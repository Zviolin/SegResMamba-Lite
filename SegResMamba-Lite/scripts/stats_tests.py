"""
统计显著性检验（实验记录总集.md Part F·统计检验 → 论文 Table 7；结果归档于总集 Part E）

读 evaluate.py 输出的逐例 CSV，配对 Wilcoxon 符号秩检验 + Holm–Bonferroni 校正。

CSV 表头（evaluate.py 生成，见 pipeline/evaluation_results/）：
  全局: CaseID,Dice_WT,Dice_TC,Dice_ET,HD95_WT,HD95_TC,HD95_ET
  病灶: CaseID,LW_Dice_WT,LW_Dice_TC,LW_Dice_ET,LW_HD95_WT,LW_HD95_TC,LW_HD95_ET

用法（在 SegResMamba-Lite 目录下执行）：
  python scripts/stats_tests.py --csv_a pipeline/evaluation_results/metrics_2.0mm.csv \
      --csv_b pipeline/evaluation_results/metrics_v7_2.0mm.csv \
      --label_a V10 --label_b V7 --out stats_vs_v7.md

检验口径：
  - 逐例配对：按 CaseID 内连接；某指标出现非有限值（如无预测时 HD95=inf）的对子
    仅在该指标下剔除，其余指标不受影响
  - scipy.stats.wilcoxon 双侧检验（全零差异对子自动按 p=1 处理）
  - Holm–Bonferroni 校正分别施加于 6 个全局指标族与 6 个病灶级指标族
  - Dice 类指标 A>B 为优，HD95 类指标 A<B 为优（表中以 Δ=mean_A-mean_B 呈现）
"""
import os
import sys
import csv
import math
import argparse

from scipy.stats import wilcoxon

# 指标族定义：全局 / 病灶级（各 6 指标，Holm 校正以族为单位）
GLOBAL_METRICS = ["Dice_WT", "Dice_TC", "Dice_ET", "HD95_WT", "HD95_TC", "HD95_ET"]
LESION_METRICS = ["LW_Dice_WT", "LW_Dice_TC", "LW_Dice_ET",
                  "LW_HD95_WT", "LW_HD95_TC", "LW_HD95_ET"]


def read_metrics_csv(path):
    """读取 evaluate.py 逐例 CSV

    Args:
        path: CSV 文件路径（表头 CaseID,Dice_WT,...）

    Returns:
        dict: {CaseID: {指标名: float}}
    """
    table = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = row["CaseID"]
            table[case_id] = {}
            for key, val in row.items():
                # 跳过 CaseID 列与多余值（旧版 CSV 表头列数少于数据列时，
                # 多余值会被 DictReader 收进 None 键的 list）
                if key is None or key == "CaseID" or not isinstance(key, str):
                    continue
                if val is None or val == "" or not isinstance(val, str):
                    continue
                try:
                    table[case_id][key] = float(val)
                except (ValueError, TypeError):
                    # 非数值列跳过
                    continue
    return table


def paired_values(table_a, table_b, metric):
    """按 CaseID 内连接取配对值，剔除非有限值对子

    Returns:
        (list_a, list_b): 等长配对列表
    """
    common = sorted(set(table_a) & set(table_b))
    a, b = [], []
    for case_id in common:
        va = table_a[case_id].get(metric)
        vb = table_b[case_id].get(metric)
        if va is None or vb is None:
            continue
        # 无预测时 HD95 可能记为 inf/nan：该指标下剔除该对子
        if not (math.isfinite(va) and math.isfinite(vb)):
            continue
        a.append(va)
        b.append(vb)
    return a, b


def holm_adjust(pvals):
    """Holm–Bonferroni 校正（族内递增累积最大值法）

    Args:
        pvals: 原始 p 值列表（与指标顺序对应）

    Returns:
        list: 校正后 p 值（顺序与输入一致）
    """
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def test_family(table_a, table_b, label_a, label_b, metrics, family_name):
    """对一个指标族执行逐指标 Wilcoxon + Holm 校正

    Returns:
        list[dict]: 每指标一行的结果
    """
    raw = []
    rows = []
    for metric in metrics:
        a, b = paired_values(table_a, table_b, metric)
        n = len(a)
        if n == 0:
            rows.append({"family": family_name, "metric": metric, "n": 0,
                         "mean_a": None, "mean_b": None, "delta": None,
                         "stat": None, "p_raw": None, "p_holm": None, "note": "无配对样本"})
            raw.append(1.0)
            continue
        mean_a = sum(a) / n
        mean_b = sum(b) / n
        diffs = [ai - bi for ai, bi in zip(a, b)]
        if all(abs(d) < 1e-12 for d in diffs):
            # 全零差异：wilcoxon 会抛异常，按无差异处理
            stat, p = None, 1.0
            note = "全部配对无差异"
        else:
            try:
                stat, p = wilcoxon(a, b, alternative="two-sided")
                note = ""
            except ValueError as exc:
                stat, p, note = None, 1.0, f"检验失败: {exc}"
        rows.append({"family": family_name, "metric": metric, "n": n,
                     "mean_a": mean_a, "mean_b": mean_b, "delta": mean_a - mean_b,
                     "stat": stat, "p_raw": p, "p_holm": None, "note": note})
        raw.append(p)
    # 族内 Holm 校正
    adjusted = holm_adjust(raw)
    for row, p_h in zip(rows, adjusted):
        row["p_holm"] = p_h
    return rows


def fmt_p(p, sig_alpha=0.05):
    """p 值格式化 + 显著性标记"""
    if p is None:
        return "-"
    star = "*" if p < sig_alpha else ""
    return f"{p:.4f}{star}"


def to_markdown(all_rows, label_a, label_b, alpha):
    """汇总为 markdown 报告"""
    lines = []
    lines.append("# 逐例配对 Wilcoxon 符号秩检验（Holm–Bonferroni 校正）")
    lines.append("")
    lines.append(f"- 模型 A: **{label_a}**；模型 B: **{label_b}**；双侧检验，α={alpha}")
    lines.append(f"- Dice 类: A-B 越大 A 越优；HD95 类: A-B 越小 A 越优")
    lines.append(f"- Holm 校正按族施加（全局 6 指标族 / 病灶级 6 指标族）")
    lines.append("")
    for family, title in [("global", "全局指标"), ("lesion", "病灶级（Lesion-wise）指标")]:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"| 指标 | n | mean({label_a}) | mean({label_b}) | Δ(A-B) | W | p (raw) | p (Holm) | 显著 |")
        lines.append("|------|--:|--------:|--------:|-------:|--:|--------:|---------:|:----:|")
        for row in all_rows:
            if row["family"] != family:
                continue
            if row["n"] == 0:
                lines.append(f"| {row['metric']} | 0 | - | - | - | - | - | - | - |")
                continue
            sig = "是" if row["p_holm"] < alpha else "否"
            stat_str = f"{row['stat']:.1f}" if row["stat"] is not None else "-"
            lines.append(
                f"| {row['metric']} | {row['n']} | {row['mean_a']:.4f} | {row['mean_b']:.4f} "
                f"| {row['delta']:+.4f} | {stat_str} "
                f"| {fmt_p(row['p_raw'], alpha)} | {fmt_p(row['p_holm'], alpha)} | {sig} |")
        lines.append("")
    lines.append("> 注: `*` 表示未校正 p<α；`显著` 列以 Holm 校正后 p<α 判定。")
    return "\n".join(lines)


def main():
    """入口：解析参数 → 读 CSV → 逐族检验 → 输出 markdown"""
    parser = argparse.ArgumentParser(description="逐例配对 Wilcoxon + Holm 校正（论文 Table 7）")
    parser.add_argument("--csv_a", required=True, help="模型 A 全局逐例 CSV（metrics_*_2.0mm.csv）")
    parser.add_argument("--csv_b", required=True, help="模型 B 全局逐例 CSV")
    parser.add_argument("--lesion_csv_a", default=None, help="模型 A 病灶级 CSV（*_lesion.csv，可选）")
    parser.add_argument("--lesion_csv_b", default=None, help="模型 B 病灶级 CSV（可选）")
    parser.add_argument("--label_a", default="A", help="模型 A 展示名")
    parser.add_argument("--label_b", default="B", help="模型 B 展示名")
    parser.add_argument("--alpha", type=float, default=0.05, help="显著性水平")
    parser.add_argument("--out", type=str, default=None, help="markdown 输出路径（缺省仅打印）")
    args = parser.parse_args()

    table_a = read_metrics_csv(args.csv_a)
    table_b = read_metrics_csv(args.csv_b)
    print(f"模型 A: {args.csv_a}（{len(table_a)} 例）")
    print(f"模型 B: {args.csv_b}（{len(table_b)} 例）")

    all_rows = []
    all_rows += test_family(table_a, table_b, args.label_a, args.label_b,
                            GLOBAL_METRICS, "global")

    # 病灶级 CSV 提供时才检验病灶族
    if args.lesion_csv_a and args.lesion_csv_b:
        lesion_a = read_metrics_csv(args.lesion_csv_a)
        lesion_b = read_metrics_csv(args.lesion_csv_b)
        all_rows += test_family(lesion_a, lesion_b, args.label_a, args.label_b,
                                LESION_METRICS, "lesion")
    else:
        print("未提供 --lesion_csv_a/--lesion_csv_b，跳过病灶级指标族")

    report = to_markdown(all_rows, args.label_a, args.label_b, args.alpha)
    print("\n" + report + "\n")

    if args.out:
        out_path = args.out if os.path.isabs(args.out) else os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.out)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
