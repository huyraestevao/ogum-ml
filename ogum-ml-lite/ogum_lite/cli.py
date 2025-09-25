"""Command line interface for the Ogum Lite toolkit."""

from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, List

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

from .features import build_feature_table
from .theta_msc import OgumLite, score_activation_energies


def parse_ea_list(raw: str) -> List[float]:
    """Parse a comma separated list of activation energies."""

    try:
        return [float(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:  # pragma: no cover - defensive
        raise argparse.ArgumentTypeError(
            "Activation energies must be numeric."
        ) from exc


def cmd_features(args: argparse.Namespace) -> None:
    dataframe = pd.read_csv(args.input)
    features_df = build_feature_table(
        dataframe,
        args.ea,
        group_col=args.group_col,
        t_col=args.time_column,
        temp_col=args.temperature_column,
        y_col=args.y_column,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)

    print(f"Feature table saved to {output_path}")
    if args.print:
        print(features_df.to_string(index=False))


def cmd_theta(args: argparse.Namespace) -> None:
    ogum = OgumLite()
    ogum.load_csv(args.input)

    theta_df = ogum.compute_theta_table(
        args.ea,
        temperature_column=args.temperature_column,
        time_column=args.time_column,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    output_path = outdir / "theta_table.csv"
    theta_df.to_csv(output_path, index=False)

    print(f"θ(Ea) table saved to {output_path}")
    if args.print:
        print(theta_df.to_string(index=False))


def cmd_msc(args: argparse.Namespace) -> None:
    dataframe = pd.read_csv(args.input)
    normalize_theta = None if args.normalize_theta == "none" else args.normalize_theta

    summary, best_result, _ = score_activation_energies(
        dataframe,
        args.ea,
        metric=args.metric,
        group_col=args.group_col,
        t_col=args.time_column,
        temp_col=args.temperature_column,
        y_col=args.y_column,
        normalize_theta=normalize_theta,
    )

    print(summary.to_string(index=False))
    print(
        "best_Ea_kJ_mol={:.3f} mse_global={:.6f} mse_segmented={:.6f}".format(
            best_result.activation_energy,
            best_result.mse_global,
            best_result.mse_segmented,
        )
    )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        best_result.curve.to_csv(args.csv, index=False)
        print(f"MSC curve exported to {args.csv}")

    if args.png:
        fig, ax = plt.subplots()
        for sample_id, sample_df in best_result.per_sample.groupby(args.group_col):
            ax.plot(
                sample_df["theta"],
                sample_df["densification"],
                alpha=0.4,
                linewidth=1.0,
                label=str(sample_id),
            )
        ax.plot(
            best_result.curve["theta_mean"],
            best_result.curve["densification"],
            color="black",
            linewidth=2.0,
            label="Mean MSC",
        )
        ax.set_xlabel(r"$\Theta(E_a)$ (normalised)")
        ax.set_ylabel("Densification")
        title = (
            f"MSC collapse (Ea={best_result.activation_energy:.1f} kJ/mol) "
            f"[metric={args.metric}]"
        )
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if len(labels) > 10:
            ax.legend(handles[:1] + [handles[-1]], ["samples", "Mean MSC"], loc="best")
        else:
            ax.legend(loc="best")
        args.png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"MSC plot saved to {args.png}")


def launch_ui() -> None:
    """Launch a minimalist Gradio UI for θ(Ea) and MSC exploration."""

    def process(
        file: gr.File,
        t_min: float | None,
        t_max: float | None,
        shift: float,
        scale: float,
        ea_list: str,
        metric: str,
    ) -> tuple[pd.DataFrame, gr.File | None, gr.File | None, gr.File | None]:
        dataframe = pd.read_csv(file.name)
        if t_min is not None:
            dataframe = dataframe[dataframe["time_s"] >= t_min]
        if t_max is not None:
            dataframe = dataframe[dataframe["time_s"] <= t_max]

        dataframe = dataframe.copy()
        dataframe["rho_rel"] = (dataframe["rho_rel"] + shift) * scale

        ea_values = parse_ea_list(ea_list)
        summary, best_result, _ = score_activation_energies(
            dataframe,
            ea_values,
            metric=metric,
        )

        tmpdir = Path(tempfile.mkdtemp(prefix="ogum_ml_ui_"))

        summary_path = tmpdir / "best_ea_mse.csv"
        summary.to_csv(summary_path, index=False)

        curve_path = tmpdir / "msc_curve.csv"
        best_result.curve.to_csv(curve_path, index=False)

        zip_path = tmpdir / "theta_curves.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for sample_id, sample_df in best_result.per_sample.groupby("sample_id"):
                buffer = sample_df.to_csv(index=False).encode("utf-8")
                zf.writestr(f"theta_{sample_id}.csv", buffer)

        return (
            summary,
            gr.File.update(value=str(summary_path)),
            gr.File.update(value=str(curve_path)),
            gr.File.update(value=str(zip_path)),
        )

    with gr.Blocks() as demo:
        gr.Markdown("## Ogum ML Lite – θ(Ea) & MSC")

        file_input = gr.File(label="Ensaios CSV (long format)")
        with gr.Row():
            t_min = gr.Number(label="t_min (s)", value=None)
            t_max = gr.Number(label="t_max (s)", value=None)
            shift = gr.Number(label="shift", value=0.0)
            scale = gr.Number(label="scale", value=1.0)
        ea_box = gr.Textbox(label="Ea list (kJ/mol)", value="200,300,400")
        metric_radio = gr.Radio(
            label="Métrica de colapso",
            choices=["segmented", "global"],
            value="segmented",
        )

        process_button = gr.Button("Calcular MSC")
        summary_df = gr.Dataframe(label="Resumo θ(Ea)")
        summary_file = gr.File(label="best_ea_mse.csv")
        curve_file = gr.File(label="msc_curve.csv")
        theta_zip = gr.File(label="theta_curves.zip")

        process_button.click(
            process,
            inputs=[file_input, t_min, t_max, shift, scale, ea_box, metric_radio],
            outputs=[summary_df, summary_file, curve_file, theta_zip],
        )

    demo.launch()


def cmd_ui(_: argparse.Namespace) -> None:
    launch_ui()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ogum ML Lite CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_features = subparsers.add_parser(
        "features", help="Compute per-sample feature tables"
    )
    parser_features.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV longo com colunas sample_id,time_s,temp_C,rho_rel.",
    )
    parser_features.add_argument(
        "--output",
        type=Path,
        default=Path("features.csv"),
        help="Arquivo de saída para a tabela de features.",
    )
    parser_features.add_argument(
        "--ea",
        type=parse_ea_list,
        required=True,
        help="Lista de Ea em kJ/mol (ex.: '200,300,400').",
    )
    parser_features.add_argument(
        "--group-col",
        default="sample_id",
        help="Nome da coluna com o identificador da amostra.",
    )
    parser_features.add_argument(
        "--time-column",
        default="time_s",
        help="Nome da coluna de tempo (s).",
    )
    parser_features.add_argument(
        "--temperature-column",
        default="temp_C",
        help="Nome da coluna de temperatura (°C).",
    )
    parser_features.add_argument(
        "--y-column",
        default="rho_rel",
        help="Coluna de densificação relativa (0–1).",
    )
    parser_features.add_argument(
        "--print",
        action="store_true",
        help="Imprime a tabela resultante no stdout.",
    )
    parser_features.set_defaults(func=cmd_features)

    parser_theta = subparsers.add_parser("theta", help="Compute θ(Ea) table")
    parser_theta.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV file com ensaios",
    )
    parser_theta.add_argument(
        "--ea",
        type=parse_ea_list,
        required=True,
        help="Comma separated Ea values in kJ/mol (e.g. '200,300,400').",
    )
    parser_theta.add_argument(
        "--time-column",
        default="time_s",
        help="Nome da coluna de tempo (s).",
    )
    parser_theta.add_argument(
        "--temperature-column",
        default="temp_C",
        help="Nome da coluna de temperatura (°C).",
    )
    parser_theta.add_argument(
        "--outdir",
        type=Path,
        default=Path("exports"),
        help="Directory used to store the θ(Ea) table.",
    )
    parser_theta.add_argument(
        "--print",
        action="store_true",
        help="Print the resulting table to stdout.",
    )
    parser_theta.set_defaults(func=cmd_theta)

    parser_msc = subparsers.add_parser("msc", help="Generate Master Sintering Curve")
    parser_msc.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV file com ensaios",
    )
    parser_msc.add_argument(
        "--ea",
        type=parse_ea_list,
        required=True,
        help="Lista de Ea em kJ/mol para avaliar.",
    )
    parser_msc.add_argument(
        "--group-col",
        default="sample_id",
        help="Nome da coluna com o identificador da amostra.",
    )
    parser_msc.add_argument(
        "--temperature-column",
        default="temp_C",
        help="Nome da coluna de temperatura (°C).",
    )
    parser_msc.add_argument(
        "--time-column",
        default="time_s",
        help="Column name containing time stamps in seconds.",
    )
    parser_msc.add_argument(
        "--y-column",
        default="rho_rel",
        help="Coluna de densificação relativa (0–1).",
    )
    parser_msc.add_argument(
        "--normalize-theta",
        choices=["minmax", "none"],
        default="minmax",
        help="Normalização aplicada antes do colapso MSC.",
    )
    parser_msc.add_argument(
        "--metric",
        choices=["global", "segmented"],
        default="segmented",
        help="Métrica usada para escolher o melhor Ea.",
    )
    parser_msc.add_argument(
        "--csv",
        type=Path,
        help="Path to export the MSC table as CSV.",
    )
    parser_msc.add_argument(
        "--png",
        type=Path,
        help="Path to save the MSC figure.",
    )
    parser_msc.set_defaults(func=cmd_msc)

    parser_ui = subparsers.add_parser("ui", help="Launch Gradio interface")
    parser_ui.set_defaults(func=cmd_ui)

    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
