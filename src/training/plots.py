from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns


def _configure_fonts() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_training_curves_pdf(
    history: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Save train/val loss and metric curves as a multi-page PDF."""
    if not history:
        return

    _configure_fonts()
    sns.set_theme(style="whitegrid", font="Times New Roman")

    epochs = [row["epoch"] for row in history]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metric_groups = [
        (
            "Loss",
            [
                ("train_loss", "Train total"),
                ("val_loss", "Val total"),
                ("train_binary_loss", "Train binary"),
                ("val_binary_loss", "Val binary"),
                ("train_multiclass_loss", "Train multiclass"),
                ("val_multiclass_loss", "Val multiclass"),
            ],
        ),
        (
            "Binary metrics",
            [
                ("train_binary_f1", "Train F1"),
                ("val_binary_f1", "Val F1"),
                ("train_binary_precision", "Train precision"),
                ("val_binary_precision", "Val precision"),
                ("train_binary_recall", "Train recall"),
                ("val_binary_recall", "Val recall"),
            ],
        ),
        (
            "Multiclass accuracy (anomalous patches)",
            [
                ("train_multiclass_acc", "Train acc"),
                ("val_multiclass_acc", "Val acc"),
            ],
        ),
    ]

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(output_path) as pdf:
        for title, series in metric_groups:
            available = [
                (key, label)
                for key, label in series
                if any(key in row and row[key] is not None for row in history)
            ]
            if not available:
                continue

            fig, ax = plt.subplots(figsize=(8.5, 5.0))
            for key, label in available:
                ys = [row.get(key) for row in history]
                ax.plot(epochs, ys, marker="o", markersize=3, label=label)

            ax.set_xlabel("Epoch")
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend(frameon=True)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # Prefer seaborn lineplot on a tidy frame for a second polished view
            tidy_rows = []
            for row in history:
                for key, label in available:
                    if row.get(key) is not None:
                        tidy_rows.append(
                            {
                                "epoch": row["epoch"],
                                "value": row[key],
                                "metric": label,
                            }
                        )
            if not tidy_rows:
                continue
            import pandas as pd

            df = pd.DataFrame(tidy_rows)
            fig, ax = plt.subplots(figsize=(8.5, 5.0))
            sns.lineplot(
                data=df,
                x="epoch",
                y="value",
                hue="metric",
                marker="o",
                ax=ax,
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel(title)
            ax.set_title(f"{title} (seaborn)")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
