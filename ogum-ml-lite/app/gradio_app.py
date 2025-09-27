"""Minimal Gradio interface for the Ogum ML Lite pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
import yaml
from ogum_lite.ui import orchestrator
from ogum_lite.ui.presets import load_presets, merge_presets
from ogum_lite.ui.workspace import Workspace

APP_DIR = Path(__file__).parent
DEFAULT_PRESET = APP_DIR / "presets.yaml"


def _prepare_workspace() -> Workspace:
    root = Path.cwd() / "artifacts" / "gradio-session"
    return Workspace(root)


def _serialize_log(ws: Workspace) -> str:
    log_path = ws.resolve(ws.log_name)
    if not log_path.exists():
        return ""
    tail = log_path.read_text(encoding="utf-8").strip().splitlines()[-8:]
    return "\n".join(tail)


def run_pipeline(
    file: gr.File | None, preset_yaml: str | None
) -> tuple[str, str | None]:
    if file is None:
        raise gr.Error("Envie um CSV longo para iniciar a pipeline.")

    ws = _prepare_workspace()
    preset = load_presets(DEFAULT_PRESET)
    if preset_yaml:
        try:
            override = yaml.safe_load(preset_yaml) or {}
        except yaml.YAMLError as exc:  # pragma: no cover - defensive
            raise gr.Error(f"Preset inválido: {exc}") from exc
        preset = merge_presets(preset, override)
    upload_path = ws.resolve("uploads") / Path(file.name).name
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    upload_path.write_bytes(Path(file.name).read_bytes())

    try:
        prep_csv = orchestrator.run_prep(upload_path, preset, ws)
        features_csv = orchestrator.run_features(prep_csv, preset, ws)
        theta_outputs = orchestrator.run_theta_msc(prep_csv, preset, ws)
        orchestrator.run_segmentation(prep_csv, preset, ws)
        orchestrator.run_mechanism(Path(theta_outputs["theta_table"]), preset, ws)
        orchestrator.run_ml_train_cls(features_csv, preset, ws)
    except Exception as exc:  # pragma: no cover - UI surface
        return f"Erro na pipeline: {exc}", None

    zip_path = orchestrator.build_report(ws.path, preset, ws)
    log_tail = _serialize_log(ws)
    return log_tail, str(zip_path)


def main() -> None:
    preset_text = (
        DEFAULT_PRESET.read_text(encoding="utf-8") if DEFAULT_PRESET.exists() else ""
    )

    with gr.Blocks(title="Ogum ML Lite") as demo:
        gr.Markdown("## Ogum ML Lite — Frontend Alpha (Gradio)")
        with gr.Row():
            file_input = gr.File(
                label="CSV longo", file_types=[".csv"], file_count="single"
            )
            preset_input = gr.Textbox(
                label="Preset YAML (opcional)",
                lines=20,
                value=preset_text,
            )
        run_btn = gr.Button("Executar pipeline")
        log_output = gr.Textbox(label="Log", lines=12)
        zip_output = gr.File(label="ZIP de artefatos")

        def _callback(file: Any, preset: str) -> tuple[str, str | None]:
            return run_pipeline(file, preset)

        run_btn.click(
            _callback,
            inputs=[file_input, preset_input],
            outputs=[log_output, zip_output],
        )

    if __name__ == "__main__":
        demo.launch()


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
