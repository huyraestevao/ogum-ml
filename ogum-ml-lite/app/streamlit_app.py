"""Ogum ML Lite Streamlit dashboard."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml
from ogum_lite.ui import orchestrator
from ogum_lite.ui.presets import load_presets, merge_presets, save_presets
from ogum_lite.ui.workspace import Workspace

APP_DIR = Path(__file__).parent
DEFAULT_PRESET_PATH = APP_DIR / "presets.yaml"


def _load_default_preset() -> dict[str, Any]:
    if "preset" in st.session_state:
        return st.session_state["preset"]
    preset = load_presets(DEFAULT_PRESET_PATH)
    st.session_state["preset"] = preset
    st.session_state["preset_yaml"] = yaml.safe_dump(
        preset, sort_keys=False, allow_unicode=True
    )
    return preset


def _get_workspace() -> Workspace:
    root_text = st.session_state.get("workspace_root")
    if not root_text:
        default_root = Path.cwd() / "artifacts" / "ui-session"
        st.session_state["workspace_root"] = str(default_root)
        root_text = str(default_root)
    workspace = Workspace(Path(root_text))
    st.session_state["workspace"] = workspace
    return workspace


def _display_log(ws: Workspace) -> None:
    log_path = ws.resolve(ws.log_name)
    if not log_path.exists():
        return
    tail = log_path.read_text(encoding="utf-8").strip().splitlines()[-5:]
    st.caption("Últimos eventos")
    for line in tail:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            st.code(line)
            continue
        st.code(json.dumps(entry, indent=2, ensure_ascii=False))


def _persist_uploaded(file, ws: Workspace) -> Path:
    uploads_dir = ws.resolve("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    target = uploads_dir / file.name
    with target.open("wb") as handle:
        handle.write(file.getbuffer())
    return target


def _yaml_editor(preset: dict[str, Any], ws: Workspace) -> dict[str, Any]:
    st.subheader("Preset manager")
    uploaded = st.file_uploader(
        "Carregar preset YAML", type=["yml", "yaml"], key="preset_upload"
    )
    if uploaded is not None:
        loaded = yaml.safe_load(uploaded.read()) or {}
        merged = merge_presets(preset, loaded)
        st.session_state["preset"] = merged
        st.session_state["preset_yaml"] = yaml.safe_dump(
            merged, sort_keys=False, allow_unicode=True
        )
        st.success("Preset carregado e mesclado com sucesso.")
        preset = merged

    preset_yaml = st.text_area(
        "Preset YAML",
        value=st.session_state.get(
            "preset_yaml", yaml.safe_dump(preset, sort_keys=False, allow_unicode=True)
        ),
        height=360,
    )
    cols = st.columns(3)
    if cols[0].button("Aplicar preset"):
        try:
            loaded = yaml.safe_load(preset_yaml) or {}
        except yaml.YAMLError as exc:
            st.error(f"Erro ao interpretar YAML: {exc}")
        else:
            st.session_state["preset"] = loaded
            st.session_state["preset_yaml"] = yaml.safe_dump(
                loaded, sort_keys=False, allow_unicode=True
            )
            preset = loaded
            st.success("Preset atualizado.")
    if cols[1].button("Restaurar padrão"):
        fresh = load_presets(DEFAULT_PRESET_PATH)
        st.session_state["preset"] = fresh
        st.session_state["preset_yaml"] = yaml.safe_dump(
            fresh, sort_keys=False, allow_unicode=True
        )
        preset = fresh
        st.info("Preset padrão restaurado.")
    if cols[2].button("Salvar preset no workspace"):
        target = _timestamped_preset_path(ws)
        save_presets(preset, target)
        st.success(f"Preset salvo em {target}")
    return preset


def _timestamped_preset_path(ws: Workspace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ws.resolve(f"presets/preset-{timestamp}.yaml")


def _show_dataframe(path: Path, caption: str, rows: int = 20) -> None:
    if not path or not Path(path).exists():
        return
    df = pd.read_csv(path)
    st.caption(caption)
    st.dataframe(df.head(rows))


def main() -> None:
    st.set_page_config(page_title="Ogum ML Lite", layout="wide")
    st.title("Ogum ML Lite — Frontend Alpha")

    preset = _load_default_preset()
    workspace = _get_workspace()

    tabs = st.tabs(
        [
            "Workspace & Presets",
            "Data Prep & Validate",
            "Features",
            "θ / MSC",
            "Segmentação & Mecanismo",
            "ML",
            "Export",
        ]
    )

    with tabs[0]:
        st.subheader("Workspace")
        root_input = st.text_input(
            "Diretório base da sessão", value=st.session_state["workspace_root"]
        )
        if root_input and root_input != st.session_state["workspace_root"]:
            st.session_state["workspace_root"] = root_input
            workspace = _get_workspace()
        st.caption(f"Workspace ativo: {workspace.path}")
        preset = _yaml_editor(preset, workspace)
        _display_log(workspace)

    with tabs[1]:
        st.subheader("Data Prep & Validação")
        uploaded = st.file_uploader("Upload CSV longo", type=["csv"], key="long_upload")
        if uploaded is not None:
            input_path = _persist_uploaded(uploaded, workspace)
            st.session_state["input_csv"] = str(input_path)
            st.success(f"Arquivo salvo em {input_path}")
        input_default = st.session_state.get("input_csv", "")
        input_path_text = st.text_input(
            "ou informe caminho existente", value=input_default
        )
        if input_path_text:
            st.session_state["input_csv"] = input_path_text
        input_csv = st.session_state.get("input_csv")

        cols = st.columns(3)
        if cols[0].button("Validar long", disabled=not input_csv):
            with st.spinner("Validando dados longos..."):
                result = orchestrator.run_validation(Path(input_csv), preset)
            st.session_state["validation_long"] = result
            st.json(result)
        if cols[1].button("Pré-processar", disabled=not input_csv):
            with st.spinner("Executando preprocess derive..."):
                prep_csv = orchestrator.run_prep(Path(input_csv), preset, workspace)
            st.session_state["prep_csv"] = str(prep_csv)
            st.success(f"Derivadas salvas em {prep_csv}")
        if cols[2].button("Recarregar preset base"):
            preset = _load_default_preset()
            st.info("Preset padrão recarregado para esta sessão.")
        _display_log(workspace)

    with tabs[2]:
        st.subheader("Features")
        prep_csv = st.session_state.get("prep_csv") or st.session_state.get("input_csv")
        st.text_input(
            "Arquivo base para features", value=prep_csv or "", key="features_input"
        )
        features_input = st.session_state.get("features_input")
        cols = st.columns(2)
        if cols[0].button("Gerar features", disabled=not features_input):
            with st.spinner("Gerando tabela de features..."):
                features_csv = orchestrator.run_features(
                    Path(features_input), preset, workspace
                )
            st.session_state["features_csv"] = str(features_csv)
            st.success(f"Tabela de features salva em {features_csv}")
        if cols[1].button(
            "Validar features", disabled=not st.session_state.get("features_csv")
        ):
            with st.spinner("Validando tabela de features..."):
                report = orchestrator.run_feature_validation(
                    Path(st.session_state["features_csv"])
                )
            st.session_state["validation_features"] = report
            st.json(report)
        if st.session_state.get("features_csv"):
            _show_dataframe(
                Path(st.session_state["features_csv"]), "Prévia das features"
            )
        _display_log(workspace)

    with tabs[3]:
        st.subheader("θ(Ea) & MSC")
        base_csv = st.session_state.get("prep_csv") or st.session_state.get("input_csv")
        if st.button("Rodar θ/MSC", disabled=not base_csv):
            with st.spinner("Calculando θ(Ea) e MSC..."):
                outputs = orchestrator.run_theta_msc(Path(base_csv), preset, workspace)
            serialized: dict[str, Any] = {}
            for key, value in outputs.items():
                if isinstance(value, Path):
                    serialized[key] = str(value)
                else:
                    serialized[key] = value
            st.session_state["theta_outputs"] = serialized
            st.success(f"Melhor Ea: {outputs['best_ea']}")
        outputs = st.session_state.get("theta_outputs")
        if outputs:
            st.metric("Ea ótimo (kJ/mol)", outputs.get("best_ea"))
            theta_table = outputs.get("theta_table")
            if theta_table and Path(theta_table).exists():
                _show_dataframe(Path(theta_table), "Tabela θ(Ea)")
            msc_plot = outputs.get("msc_plot")
            if msc_plot and Path(msc_plot).exists():
                st.image(msc_plot, caption="Master Sintering Curve")
            msc_curve = outputs.get("msc_curve")
            if msc_curve and Path(msc_curve).exists():
                st.download_button(
                    "Baixar MSC CSV",
                    data=Path(msc_curve).read_bytes(),
                    file_name=Path(msc_curve).name,
                )
            theta_zip = outputs.get("theta_zip")
            if theta_zip and Path(theta_zip).exists():
                st.download_button(
                    "Baixar θ(Ea) ZIP",
                    data=Path(theta_zip).read_bytes(),
                    file_name=Path(theta_zip).name,
                )
        _display_log(workspace)

    with tabs[4]:
        st.subheader("Segmentação & Mecanismo")
        base_csv = st.session_state.get("prep_csv")
        theta_table = None
        if st.session_state.get("theta_outputs"):
            theta_table = st.session_state["theta_outputs"].get("theta_table")
        cols = st.columns(2)
        if cols[0].button("Segmentação", disabled=not base_csv):
            with st.spinner("Executando segmentação..."):
                segments = orchestrator.run_segmentation(
                    Path(base_csv), preset, workspace
                )
            st.session_state["segments_json"] = str(segments)
            st.success(f"Segmentos salvos em {segments}")
        if cols[1].button("Detectar mecanismo", disabled=not theta_table):
            with st.spinner("Detectando mudança de mecanismo..."):
                mechanism = orchestrator.run_mechanism(
                    Path(theta_table), preset, workspace
                )
            st.session_state["mechanism_csv"] = str(mechanism)
            st.success(f"Relatório salvo em {mechanism}")
        if (
            st.session_state.get("segments_json")
            and Path(st.session_state["segments_json"]).exists()
        ):
            data = json.loads(Path(st.session_state["segments_json"]).read_text())
            st.json(data)
        if (
            st.session_state.get("mechanism_csv")
            and Path(st.session_state["mechanism_csv"]).exists()
        ):
            _show_dataframe(
                Path(st.session_state["mechanism_csv"]), "Relatório de mecanismo"
            )
        _display_log(workspace)

    with tabs[5]:
        st.subheader("Treinamento ML")
        features_csv = st.session_state.get("features_csv")
        cols = st.columns(3)
        if cols[0].button("Treinar classificador", disabled=not features_csv):
            with st.spinner("Treinando classificador..."):
                result = orchestrator.run_ml_train_cls(
                    Path(features_csv), preset, workspace
                )
            st.session_state["ml_cls"] = {
                "outdir": str(result["outdir"]),
                "model_card": result["model_card"],
            }
            st.success(f"Artefatos do classificador em {result['outdir']}")
        if cols[1].button("Treinar regressor", disabled=not features_csv):
            with st.spinner("Treinando regressor..."):
                result = orchestrator.run_ml_train_reg(
                    Path(features_csv), preset, workspace
                )
            st.session_state["ml_reg"] = {
                "outdir": str(result["outdir"]),
                "model_card": result["model_card"],
            }
            st.success(f"Artefatos do regressor em {result['outdir']}")
        if cols[2].button(
            "Prever", disabled=not features_csv or not st.session_state.get("ml_cls")
        ):
            model_info = st.session_state.get("ml_cls")
            if model_info:
                model_path = Path(model_info["outdir"]) / "classifier.joblib"
                with st.spinner("Gerando predições..."):
                    output = orchestrator.run_ml_predict(
                        Path(features_csv), model_path, preset, workspace
                    )
                st.session_state["predictions_csv"] = str(output)
                st.success(f"Predições salvas em {output}")
        if st.session_state.get("ml_cls"):
            st.subheader("Métricas — Classificação")
            st.json(st.session_state["ml_cls"].get("model_card", {}))
        if st.session_state.get("ml_reg"):
            st.subheader("Métricas — Regressão")
            st.json(st.session_state["ml_reg"].get("model_card", {}))
        if st.session_state.get("predictions_csv"):
            _show_dataframe(Path(st.session_state["predictions_csv"]), "Predições")
        _display_log(workspace)

    with tabs[6]:
        st.subheader("Export & Relatórios")
        export_cfg = preset.setdefault("export", {})
        theta_outputs = st.session_state.get("theta_outputs", {})
        if theta_outputs:
            export_cfg.setdefault("msc_csv", theta_outputs.get("msc_curve"))
            export_cfg.setdefault("img_msc", theta_outputs.get("msc_plot"))
        if st.session_state.get("features_csv"):
            export_cfg.setdefault("features_csv", st.session_state.get("features_csv"))
        if st.session_state.get("ml_cls"):
            export_cfg.setdefault(
                "metrics_json",
                str(Path(st.session_state["ml_cls"]["outdir"]) / "model_card.json"),
            )
        if st.button("Gerar ZIP de artefatos"):
            with st.spinner("Gerando relatório e pacote ZIP..."):
                zip_path = orchestrator.build_report(workspace.path, preset, workspace)
            st.session_state["export_zip"] = str(zip_path)
            st.success(f"Pacote gerado em {zip_path}")
        if (
            st.session_state.get("export_zip")
            and Path(st.session_state["export_zip"]).exists()
        ):
            zip_path = Path(st.session_state["export_zip"])
            st.download_button(
                "Baixar artefatos",
                data=zip_path.read_bytes(),
                file_name=zip_path.name,
            )
        _display_log(workspace)


if __name__ == "__main__":
    main()
