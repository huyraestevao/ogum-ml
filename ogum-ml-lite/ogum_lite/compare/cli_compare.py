"""Command line entry-points for the comparison toolkit."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .diff_core import (
    compose_diff_summary,
    diff_mechanism,
    diff_ml,
    diff_msc,
    diff_presets,
    diff_segments,
)
from .loaders import find_run_root, load_json, load_yaml, scan_run
from .reporters import export_xlsx_compare, render_html_compare


@dataclass
class RunArtifacts:
    """Container with resolved paths for a run."""

    name: str
    manifest: dict[str, Any]
    preset: dict[str, Any] | None


def _load_run(path: Path) -> RunArtifacts:
    root = find_run_root(path)
    manifest = scan_run(root)
    preset = load_yaml(manifest.get("presets")) or load_json(manifest.get("presets"))
    return RunArtifacts(name=root.name, manifest=manifest, preset=preset)


def _serialisable(payload: dict[str, Any]) -> dict[str, Any]:
    def _convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_convert(item) for item in value]
        return value

    return _convert(payload)


def _diff_runs(run_a: RunArtifacts, run_b: RunArtifacts) -> dict[str, Any]:
    msc_diff = diff_msc(run_a.manifest.get("msc_csv"), run_b.manifest.get("msc_csv"))
    segments_diff = diff_segments(
        run_a.manifest.get("segments_table"), run_b.manifest.get("segments_table")
    )
    mechanism_diff = diff_mechanism(
        run_a.manifest.get("mechanism_csv"), run_b.manifest.get("mechanism_csv")
    )
    ml_diff = diff_ml(
        run_a.manifest.get("ml_model_card"),
        run_b.manifest.get("ml_model_card"),
        run_a.manifest.get("ml_cv_metrics"),
        run_b.manifest.get("ml_cv_metrics"),
    )
    presets_diff = diff_presets(run_a.preset, run_b.preset)
    summary = compose_diff_summary(
        presets=presets_diff,
        msc=msc_diff,
        segments=segments_diff,
        mechanism=mechanism_diff,
        ml=ml_diff,
    )
    return {
        "summary": summary,
        "presets": presets_diff,
        "msc": msc_diff,
        "segments": segments_diff,
        "mechanism": mechanism_diff,
        "ml": ml_diff,
    }


def cmd_compare_runs(args: argparse.Namespace) -> None:
    run_a = _load_run(Path(args.a))
    run_b = _load_run(Path(args.b))
    diffs = _diff_runs(run_a, run_b)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary_rows = diffs["summary"].get("kpis", [])
    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        summary_df = pd.DataFrame([{"metric": "n/a"}])

    tables: dict[str, pd.DataFrame] = {}
    if diffs["presets"].get("changed"):
        tables["Presets"] = pd.DataFrame(
            [
                {"key": key, "a": value["a"], "b": value["b"]}
                for key, value in diffs["presets"]["changed"].items()
            ]
        )
    if diffs["segments"].get("changed"):
        rows = []
        for key, value in diffs["segments"]["changed"].items():
            for metric, payload in value.items():
                rows.append(
                    {
                        "segment": "×".join(map(str, key)),
                        "metric": metric,
                        **payload,
                    }
                )
        tables["Segments"] = pd.DataFrame(rows)
    if diffs["ml"].get("metrics"):
        tables["ML"] = pd.DataFrame(diffs["ml"]["metrics"].values())

    images: dict[str, bytes] = {}
    for label in ("ml_confusion", "ml_scatter", "msc_plot"):
        path = run_a.manifest.get(label)
        if path and Path(path).exists():
            images[f"A:{label}"] = Path(path).read_bytes()
        path = run_b.manifest.get(label)
        if path and Path(path).exists():
            images[f"B:{label}"] = Path(path).read_bytes()
    if not images:
        images = None

    html_path = render_html_compare(
        outdir, {"runs": [run_a.name, run_b.name]}, {**diffs}, images=images
    )
    xlsx_path = export_xlsx_compare(
        outdir / "report.xlsx", summary=summary_df, tables=tables, images=images
    )

    payload = {
        "runs": {"a": run_a.manifest, "b": run_b.manifest},
        "diffs": diffs,
        "html": html_path,
        "xlsx": xlsx_path,
    }
    (outdir / "compare_summary.json").write_text(
        json.dumps(_serialisable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def cmd_compare_matrix(args: argparse.Namespace) -> None:
    ref_run = _load_run(Path(args.ref))
    candidates = [_load_run(Path(path)) for path in args.candidates]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        diffs = _diff_runs(ref_run, candidate)
        summary_rows = diffs["summary"].get("kpis", [])
        best_metric = None
        if summary_rows:
            best_metric = min(summary_rows, key=lambda item: item.get("delta", 0.0))
        row = {
            "candidate": candidate.name,
            "alerts": "; ".join(diffs["summary"].get("alerts", [])),
        }
        if best_metric:
            row.update(
                {
                    "metric": best_metric.get("metric"),
                    "delta": best_metric.get("delta"),
                }
            )
        rows.append(row)

        target_dir = outdir / f"{ref_run.name}_vs_{candidate.name}"
        target_dir.mkdir(parents=True, exist_ok=True)
        cmd_compare_runs(
            argparse.Namespace(
                a=args.ref,
                b=str(candidate.manifest["root"]),
                outdir=str(target_dir),
            )
        )

    ranking = pd.DataFrame(rows)
    ranking_path = outdir / "ranking.csv"
    ranking.to_csv(ranking_path, index=False)

    html = ranking.to_html(index=False)
    (outdir / "matrix.html").write_text(html, encoding="utf-8")


def build_compare_parser(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="compare_command", required=True)

    runs_parser = subparsers.add_parser("runs", help="Compare two runs")
    runs_parser.add_argument("--a", required=True, help="Run A path or zip")
    runs_parser.add_argument("--b", required=True, help="Run B path or zip")
    runs_parser.add_argument("--outdir", required=True, help="Output directory")
    runs_parser.set_defaults(func=cmd_compare_runs)

    matrix_parser = subparsers.add_parser("matrix", help="Reference vs candidates")
    matrix_parser.add_argument("--ref", required=True, help="Reference run path")
    matrix_parser.add_argument(
        "--candidates", nargs="+", required=True, help="Candidate run paths"
    )
    matrix_parser.add_argument("--outdir", required=True, help="Directory for outputs")
    matrix_parser.set_defaults(func=cmd_compare_matrix)


__all__ = ["build_compare_parser", "cmd_compare_runs", "cmd_compare_matrix"]
