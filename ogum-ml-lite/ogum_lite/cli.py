"""Command line interface for the Ogum Lite toolkit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import gradio as gr
import pandas as pd

from .theta_msc import OgumLite


def parse_ea_list(raw: str) -> List[float]:
    """Parse a comma separated list of activation energies."""

    try:
        return [float(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:  # pragma: no cover - defensive
        raise argparse.ArgumentTypeError(
            "Activation energies must be numeric."
        ) from exc


def cmd_features(args: argparse.Namespace) -> None:
    """Placeholder for the upcoming feature extraction routines."""

    payload = {
        "status": "not-implemented",
        "message": "Feature extraction pipeline will be added in a future release.",
        "requested_metrics": args.metrics,
    }
    print(json.dumps(payload, indent=2))


def cmd_theta(args: argparse.Namespace) -> None:
    ogum = OgumLite()
    ogum.load_csv(args.input)

    theta_df = ogum.compute_theta_table(args.ea)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    output_path = outdir / "theta_table.csv"
    theta_df.to_csv(output_path, index=False)

    print(f"θ(Ea) table saved to {output_path}")
    if args.print:
        print(theta_df.to_string(index=False))


def cmd_msc(args: argparse.Namespace) -> None:
    ogum = OgumLite()
    ogum.load_csv(args.input)

    msc_df = ogum.build_msc(
        activation_energy=args.ea,
        densification_column=args.densification_column,
        temperature_column=args.temperature_column,
        time_column=args.time_column,
    )

    if args.csv:
        msc_df.to_csv(args.csv, index=False)
        print(f"MSC exported to {args.csv}")

    if args.png:
        ax = ogum.plot_msc(msc_df)
        ax.figure.savefig(args.png, dpi=150, bbox_inches="tight")
        print(f"MSC figure saved to {args.png}")

    if not args.csv and not args.png:
        print(msc_df.to_string(index=False))


def launch_ui() -> None:
    """Launch a minimalist Gradio UI for θ(Ea) and MSC exploration."""

    ogum = OgumLite()

    def compute_theta(file: gr.File, ea_list: str) -> pd.DataFrame:
        dataframe = pd.read_csv(file.name)
        ogum.data = dataframe
        ea_values = parse_ea_list(ea_list)
        return ogum.compute_theta_table(ea_values)

    def compute_msc(file: gr.File, ea_value: float) -> pd.DataFrame:
        dataframe = pd.read_csv(file.name)
        ogum.data = dataframe
        return ogum.build_msc(activation_energy=ea_value)

    with gr.Blocks() as demo:
        gr.Markdown("## Ogum ML Lite – θ(Ea) & MSC")

        with gr.Tab("θ(Ea)"):
            file_theta = gr.File(label="Ensaios CSV")
            ea_input = gr.Textbox(label="Ea (kJ/mol)", value="200, 300")
            theta_button = gr.Button("Calcular θ(Ea)")
            theta_output = gr.Dataframe()
            theta_button.click(
                compute_theta,
                inputs=[file_theta, ea_input],
                outputs=theta_output,
            )

        with gr.Tab("MSC"):
            file_msc = gr.File(label="Ensaios CSV")
            ea_slider = gr.Slider(
                label="Ea (kJ/mol)",
                value=200,
                minimum=50,
                maximum=600,
                step=10,
            )
            msc_button = gr.Button("Gerar MSC")
            msc_output = gr.Dataframe()
            msc_button.click(
                compute_msc,
                inputs=[file_msc, ea_slider],
                outputs=msc_output,
            )

    demo.launch()


def cmd_ui(_: argparse.Namespace) -> None:
    launch_ui()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ogum ML Lite CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_features = subparsers.add_parser(
        "features", help="Feature extraction helpers"
    )
    parser_features.add_argument(
        "--metrics",
        nargs="*",
        default=[],
        help="Specific metrics to compute (future use).",
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
        type=float,
        required=True,
        help="Activation energy (kJ/mol) for the MSC integration.",
    )
    parser_msc.add_argument(
        "--densification-column",
        default="densification",
        help="Column name containing densification values.",
    )
    parser_msc.add_argument(
        "--temperature-column",
        default="temperature_C",
        help="Column name containing temperatures in Celsius.",
    )
    parser_msc.add_argument(
        "--time-column",
        default="time_s",
        help="Column name containing time stamps in seconds.",
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
