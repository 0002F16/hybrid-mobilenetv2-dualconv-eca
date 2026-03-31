from __future__ import annotations


def caption_table_main_results() -> str:
    return (
        "Table X. Classification accuracy and efficiency metrics of MobileNetV2 architectural variants "
        "on CIFAR-10, CIFAR-100, and Tiny-ImageNet. Results reported as mean ± standard deviation over "
        "5 independent runs."
    )


def caption_table_stats() -> str:
    return (
        "Table B. Paired non-parametric significance testing of accuracy differences relative to the Baseline "
        "model. Reported statistics include the median paired difference (percentage points), a 95% bootstrap "
        "confidence interval, Wilcoxon signed-rank test (two-sided), and Holm–Bonferroni corrected p-values."
    )

