"""Gradio fallback interface reusing the service layer."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import yaml
from ogum_lite.ui.presets import load_presets, merge_presets
from ogum_lite.ui.workspace import Workspace

from app import gradio_jobs
from app.services import run_cli

APP_DIR = Path(__file__).parent
DEFAULT_PRESET = APP_DIR / "presets.yaml"


def _workspace() -> Workspace:
    root = Path.cwd() / "artifacts" / "gradio-session"
    return Workspace(root)


def _serialise_log(ws: Workspace) -> str:
    log_path = ws.resolve(ws.log_name)
    if not log_path.exists():
        return ""
    return "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-12:])


def run_pipeline(
    file: gr.File | None, preset_yaml: str | None
) -> tuple[str, str | None]:
    if file is None:
        raise gr.Error("Envie um CSV longo para iniciar a pipeline.")

    ws = _workspace()
    preset = load_presets(DEFAULT_PRESET)
    if preset_yaml:
        override = yaml.safe_load(preset_yaml) or {}
        preset = merge_presets(preset, override)

    upload_path = ws.resolve("uploads") / Path(file.name).name
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    upload_path.write_bytes(Path(file.name).read_bytes())

    prep_result = run_cli.run_prep(upload_path, preset, ws)
    features_result = run_cli.run_features(prep_result.outputs["prep_csv"], preset, ws)
    theta_result = run_cli.run_theta_msc(prep_result.outputs["prep_csv"], preset, ws)
    run_cli.run_segmentation(prep_result.outputs["prep_csv"], preset, ws)
    run_cli.run_mechanism(theta_result.outputs["theta_table"], preset, ws)
    run_cli.run_ml_train_cls(features_result.outputs["features_csv"], preset, ws)
    export_result = run_cli.export_report(ws.path, preset, ws)

    return _serialise_log(ws), str(export_result.outputs["zip"])


def main() -> None:
    preset_text = (
        DEFAULT_PRESET.read_text(encoding="utf-8") if DEFAULT_PRESET.exists() else ""
    )
    with gr.Blocks(title="Ogum-ML") as demo:
        with gr.Tab("Pipeline"):
            gr.Markdown("## Ogum-ML — Pipeline Lite")
            with gr.Row():
                file_input = gr.File(
                    label="CSV longo", file_types=[".csv"], file_count="single"
                )
                preset_input = gr.Textbox(
                    label="Preset YAML", lines=18, value=preset_text
                )
            run_btn = gr.Button("Executar pipeline")
            log_output = gr.Textbox(label="Log", lines=12)
            zip_output = gr.File(label="ZIP de artefatos")

            run_btn.click(
                run_pipeline,
                inputs=[file_input, preset_input],
                outputs=[log_output, zip_output],
            )

        gradio_jobs.render_jobs_tab()

    if __name__ == "__main__":
        demo.launch()


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
