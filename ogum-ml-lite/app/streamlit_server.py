"""Streamlit frontend that interacts with the collaborative backend."""

from __future__ import annotations

import json
import os
from typing import Any

import requests
import streamlit as st

API_URL = os.environ.get("OGUM_API_URL", "http://localhost:8000")


def _api_request(method: str, path: str, token: str | None = None, **kwargs) -> Any:
    url = f"{API_URL}{path}"
    headers = kwargs.pop("headers", {})
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.request(method, url, headers=headers, timeout=30, **kwargs)
    if response.status_code == 401:
        raise RuntimeError("Unauthorized. Please login again.")
    response.raise_for_status()
    if response.headers.get("content-type", "").startswith("application/json"):
        return response.json()
    return response.content


def _login(username: str, password: str) -> str:
    payload = {"username": username, "password": password}
    data = _api_request("POST", "/auth/login", json=payload)
    return data["access_token"]


def _load_runs(token: str) -> list[dict[str, Any]]:
    return _api_request("GET", "/runs/list", token=token)


def _load_jobs(token: str) -> list[dict[str, Any]]:
    return _api_request("GET", "/jobs/list", token=token)


def _submit_compare(
    token: str, base: str, candidate: str, compare_type: str
) -> dict[str, Any]:
    payload = {
        "base_run": base,
        "candidate_run": candidate,
        "compare_type": compare_type,
    }
    return _api_request("POST", "/compare/run", token=token, json=payload)


def _register_simulation(token: str, payload: dict[str, Any]) -> dict[str, Any]:
    return _api_request("POST", "/sim/import", token=token, json=payload)


def _upload_run(token: str, file, description: str | None) -> dict[str, Any]:
    files = {"file": (file.name, file.getvalue())}
    data = {}
    if description:
        data["description"] = description
    return _api_request("POST", "/runs/upload", token=token, files=files, data=data)


def _render_login() -> str | None:
    st.header("Ogum-ML Login")
    with st.form("login-form"):
        username = st.text_input("Usuário")
        password = st.text_input("Senha", type="password")
        submitted = st.form_submit_button("Entrar")
    if submitted:
        try:
            token = _login(username, password)
        except requests.HTTPError as exc:
            st.error(f"Falha no login: {exc.response.text}")
            return None
        except RuntimeError as exc:  # pragma: no cover - defensive
            st.error(str(exc))
            return None
        else:
            st.success("Login realizado com sucesso!")
            return token
    return None


def _ensure_session_state() -> None:
    if "token" not in st.session_state:
        st.session_state["token"] = None


def _render_runs_tab(token: str) -> None:
    st.subheader("Execuções disponíveis")
    runs = _load_runs(token)
    if not runs:
        st.info("Nenhuma execução disponível.")
    else:
        st.table(runs)

    st.divider()
    st.subheader("Upload de nova execução")
    uploaded = st.file_uploader(
        "Selecione um arquivo de resultados", type=["json", "csv", "zip", "xlsx"]
    )
    description = st.text_input("Descrição")
    if uploaded and st.button("Enviar execução"):
        result = _upload_run(token, uploaded, description or None)
        st.success(f"Execução registrada: {json.dumps(result, ensure_ascii=False)}")


def _render_jobs_tab(token: str) -> None:
    st.subheader("Fila de Jobs")
    jobs = _load_jobs(token)
    if not jobs:
        st.info("Nenhum job em andamento.")
    else:
        st.table(jobs)


def _render_compare_tab(token: str) -> None:
    st.subheader("Comparar execuções")
    runs = _load_runs(token)
    names = [run["name"] for run in runs]
    if len(names) < 2:
        st.info("É necessário pelo menos duas execuções para comparar.")
        return
    base = st.selectbox("Execução base", names)
    candidate = st.selectbox("Execução candidata", names, index=1)
    compare_type = st.selectbox("Tipo de comparação", ["summary", "full"])
    if st.button("Iniciar comparação"):
        response = _submit_compare(token, base, candidate, compare_type)
        st.success(f"Job criado: {response['job_id']}")


def _render_simulation_tab(token: str) -> None:
    st.subheader("Registrar bundle de simulação")
    name = st.text_input("Nome do bundle")
    description = st.text_area("Descrição")
    dataset = st.text_input("Dataset associado")
    metadata = st.text_area("Metadados (JSON opcional)")
    if st.button("Registrar simulação"):
        extra: dict[str, Any] = {}
        if metadata:
            try:
                extra = json.loads(metadata)
            except json.JSONDecodeError:
                st.error("Metadados devem ser um JSON válido.")
                return
        payload = {
            "name": name,
            "description": description or None,
            "dataset": dataset or None,
            "metadata": extra or None,
        }
        response = _register_simulation(token, payload)
        st.success(f"Simulação registrada em {response['artifact_path']}")


def main() -> None:  # pragma: no cover - entry point for streamlit
    st.set_page_config(page_title="Ogum-ML Collaborativo", layout="wide")
    _ensure_session_state()

    if st.session_state["token"] is None:
        token = _render_login()
        if token:
            st.session_state["token"] = token
        else:
            st.stop()

    token = st.session_state["token"]
    st.sidebar.title("Ogum-ML")
    if st.sidebar.button("Sair"):
        st.session_state["token"] = None
        st.experimental_rerun()

    tabs = st.tabs(
        [
            "Execuções",
            "Jobs",
            "Comparações",
            "Simulações",
        ]
    )

    with tabs[0]:
        _render_runs_tab(token)
    with tabs[1]:
        _render_jobs_tab(token)
    with tabs[2]:
        _render_compare_tab(token)
    with tabs[3]:
        _render_simulation_tab(token)


if __name__ == "__main__":  # pragma: no cover - script mode
    main()
