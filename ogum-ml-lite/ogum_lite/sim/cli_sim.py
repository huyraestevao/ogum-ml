"""Command line helpers for simulation workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import adapters, features, linker, schema


def build_sim_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``sim`` subcommands."""

    parser_sim = subparsers.add_parser(
        "sim",
        help="Ferramentas para simulações",
    )
    sim_subparsers = parser_sim.add_subparsers(dest="sim_command", required=True)

    parser_import = sim_subparsers.add_parser(
        "import",
        help="Ingerir resultados de simulação",
    )
    parser_import.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Origem dos arquivos",
    )
    parser_import.add_argument(
        "--format",
        choices=["vtk", "xdmf", "csv"],
        required=True,
        help="Formato de entrada da simulação",
    )
    parser_import.add_argument(
        "--pattern",
        default="*.vtu",
        help="Glob para séries VTK (padrão: *.vtu)",
    )
    parser_import.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Diretório de saída para o bundle canônico",
    )
    parser_import.set_defaults(func=cmd_sim_import)

    parser_features = sim_subparsers.add_parser(
        "features",
        help="Calcular features de simulação",
    )
    parser_features.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="Diretório do bundle salvo via sim import",
    )
    parser_features.add_argument(
        "--segments",
        default=None,
        help="Intervalos de tempo no formato 't0,t1;t1,t2' (opcional)",
    )
    parser_features.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Arquivo CSV de saída para as features",
    )
    parser_features.set_defaults(func=cmd_sim_features)

    parser_link = sim_subparsers.add_parser(
        "link",
        help="Vincular features reais e simuladas",
    )
    parser_link.add_argument(
        "--exp-features",
        type=Path,
        required=True,
        help="CSV com features experimentais (requer sample_id)",
    )
    parser_link.add_argument(
        "--sim-features",
        type=Path,
        required=True,
        help="CSV com features de simulação (requer sim_id)",
    )
    parser_link.add_argument(
        "--map",
        type=Path,
        required=False,
        help="Arquivo YAML com pares sample_id: sim_id",
    )
    parser_link.add_argument(
        "--out",
        type=Path,
        required=True,
        help="CSV combinado de saída",
    )
    parser_link.set_defaults(func=cmd_sim_link)


def cmd_sim_import(args: argparse.Namespace) -> None:
    try:
        if args.format == "vtk":
            bundle = adapters.load_vtk_series(args.src, pattern=args.pattern)
        elif args.format == "xdmf":
            bundle = adapters.load_xdmf(args.src)
        else:
            bundle = adapters.load_csv_timeseries(args.src)
    except Exception as exc:  # pragma: no cover - defensive
        raise SystemExit(f"Falha ao importar simulação: {exc}") from exc

    schema.to_disk(bundle, args.outdir)
    print(f"Bundle salvo em {args.outdir}")


def _parse_segments(raw: str | None) -> list[tuple[float, float]]:
    if raw is None or not raw.strip():
        return []
    segments: list[tuple[float, float]] = []
    for chunk in raw.split(";"):
        piece = chunk.strip()
        if not piece:
            continue
        try:
            start_str, end_str = piece.split(",", 1)
            start = float(start_str)
            end = float(end_str)
        except ValueError as exc:
            raise SystemExit(
                "Segmentos devem usar formato 'inicio,fim;inicio,fim' em segundos"
            ) from exc
        segments.append((start, end))
    return segments


def cmd_sim_features(args: argparse.Namespace) -> None:
    bundle = schema.from_disk(args.bundle)
    global_df = features.sim_features_global(bundle)

    segments = _parse_segments(args.segments)
    if segments:
        seg_df = features.sim_features_segmented(bundle, segments)
        for row_idx, row in seg_df.iterrows():
            prefix = _segment_prefix(
                row_idx,
                row["segment_start_s"],
                row["segment_end_s"],
            )
            for col, value in row.items():
                if col in {"sim_id", "segment_start_s", "segment_end_s"}:
                    continue
                global_df[f"{col}_{prefix}"] = value

    args.out.parent.mkdir(parents=True, exist_ok=True)
    global_df.to_csv(args.out, index=False)
    print(f"Features salvas em {args.out}")


def _segment_prefix(idx: int, start: float, end: float) -> str:
    def fmt(value: float) -> str:
        if abs(value - round(value)) < 1e-6:
            return str(int(round(value)))
        return f"{value:.3f}".rstrip("0").rstrip(".")

    return f"segment_{idx}_{fmt(start)}_{fmt(end)}"


def cmd_sim_link(args: argparse.Namespace) -> None:
    merged = linker.link_runs(args.exp_features, args.sim_features, args.map)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"Tabela combinada salva em {args.out}")


__all__ = ["build_sim_parser", "cmd_sim_import", "cmd_sim_features", "cmd_sim_link"]
