"""Guided exercises for the Educational Mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from . import simulators


@dataclass
class Exercise:
    """Metadata and evaluation logic for an educational exercise."""

    key: str
    statement_md: str
    inputs_spec: Mapping[str, Mapping[str, str]]
    evaluate: Callable[[Mapping[str, float]], dict]


def _generate_curves() -> list[dict]:
    time = np.linspace(0, 1800, 200)
    base_temp = 950 + 120 * np.sin(np.linspace(0, np.pi, time.size))
    offsets = (-40, 0, 45)
    curves: list[dict] = []
    for idx, offset in enumerate(offsets, start=1):
        temp = base_temp + offset
        theta = simulators.simulate_theta(temp, time, 320.0)
        theta_norm = theta / theta[-1]
        densification = 0.45 + 0.5 * np.power(theta_norm, 1.4 + 0.1 * idx)
        curves.append(
            {
                "time_s": time,
                "temp_C": temp,
                "y": densification,
                "label": f"Amostra {idx}",
            }
        )
    return curves


_CURVES = _generate_curves()
_COLLAPSE_TARGET = simulators.simulate_msc_collapse(_CURVES, 320.0)
_MEAN_THETA = _COLLAPSE_TARGET.grid_theta
_MEAN_Y = _COLLAPSE_TARGET.mean_curve
_SEGMENT_REFERENCES = {
    55: float(np.interp(0.55, _MEAN_Y, _MEAN_THETA)),
    70: float(np.interp(0.70, _MEAN_Y, _MEAN_THETA)),
    90: float(np.interp(0.90, _MEAN_Y, _MEAN_THETA)),
}
_BLAINE_REFERENCE = simulators.simulate_blaine_linearization(
    theta=_COLLAPSE_TARGET.grid_theta + 1e-6, y=_COLLAPSE_TARGET.mean_curve
)


def exercise_choose_ea() -> Exercise:
    """Exercise that asks the learner to tune the activation energy."""

    statement = (
        "**Escolha de Ea / Choosing Ea**\n\n"
        "Ajuste o valor de Ea (kJ/mol) para que as curvas colapsadas tenham o menor "
        "erro médio. Use a simulação acima como referência."
    )
    inputs = {"Ea": {"label": "Ea (kJ/mol)", "type": "number"}}

    def _evaluate(payload: Mapping[str, float]) -> dict:
        try:
            candidate = float(payload.get("Ea", 0))
        except (TypeError, ValueError):
            return {"score": 0.0, "feedback": "Informe um número válido para Ea."}
        result = simulators.simulate_msc_collapse(_CURVES, candidate)
        baseline = _COLLAPSE_TARGET.mse
        delta = abs(result.mse - baseline)
        score = max(0.0, 1.0 - delta / max(baseline, 1e-6))
        feedback = (
            f"MSE obtido: {result.mse:.4f}. O valor de referência está em torno de "
            f"{baseline:.4f}."
        )
        return {"score": round(score, 3), "feedback": feedback}

    return Exercise(
        key="choose_ea",
        statement_md=statement,
        inputs_spec=inputs,
        evaluate=_evaluate,
    )


def exercise_segments() -> Exercise:
    """Exercise about locating the 55–70–90% densification segments."""

    statement = (
        "**Segmentos 55–70–90 / 55–70–90 segments**\n\n"
        "Indique os valores de Θ normalizado quando a densificação atinge 55%, 70% e "
        "90%. Tolerância de ±0.03."
    )
    inputs = {
        "seg_55": {"label": "Θ @55%", "type": "number"},
        "seg_70": {"label": "Θ @70%", "type": "number"},
        "seg_90": {"label": "Θ @90%", "type": "number"},
    }

    def _evaluate(payload: Mapping[str, float]) -> dict:
        tolerance = 0.03
        hits = 0
        feedback_lines = []
        for level in (55, 70, 90):
            key = f"seg_{level}"
            try:
                guess = float(payload.get(key, np.nan))
            except (TypeError, ValueError):
                guess = np.nan
            reference = _SEGMENT_REFERENCES[level]
            if np.isnan(guess):
                feedback_lines.append(f"{level}%: valor inválido.")
                continue
            error = abs(guess - reference)
            if error <= tolerance:
                hits += 1
                feedback_lines.append(f"{level}%: ok (erro {error:.3f}).")
            else:
                feedback_lines.append(
                    f"{level}%: fora da tolerância (referência {reference:.3f})."
                )
        score = hits / 3
        return {"score": round(score, 3), "feedback": " ".join(feedback_lines)}

    return Exercise(
        key="segments",
        statement_md=statement,
        inputs_spec=inputs,
        evaluate=_evaluate,
    )


def exercise_blaine_n() -> Exercise:
    """Exercise about estimating ``n`` through Blaine linearisation."""

    statement = (
        "**Expoente n de Blaine / Blaine exponent n**\n\n"
        "Utilize o ajuste linear em ln(Θ) × ln(y) para estimar n. Compare com o "
        "referencial do modo educativo."
    )
    inputs = {"n": {"label": "n", "type": "number"}}

    def _evaluate(payload: Mapping[str, float]) -> dict:
        try:
            guess = float(payload.get("n", 0.0))
        except (TypeError, ValueError):
            return {"score": 0.0, "feedback": "Informe um valor numérico para n."}
        reference = _BLAINE_REFERENCE.n_est
        error = abs(guess - reference)
        tolerance = 0.1
        score = max(0.0, 1.0 - error / tolerance)
        feedback = f"n estimado: {guess:.3f}. Referência: {reference:.3f}."
        return {"score": round(score, 3), "feedback": feedback}

    return Exercise(
        key="blaine_n",
        statement_md=statement,
        inputs_spec=inputs,
        evaluate=_evaluate,
    )


EXERCISES = [exercise_choose_ea(), exercise_segments(), exercise_blaine_n()]


def get_reference_curves() -> list[dict]:
    """Return deep copies of the reference curves used in the exercises."""

    curves: list[dict] = []
    for payload in _CURVES:
        curves.append(
            {
                "time_s": np.array(payload["time_s"], copy=True),
                "temp_C": np.array(payload["temp_C"], copy=True),
                "y": np.array(payload["y"], copy=True),
                "label": payload["label"],
            }
        )
    return curves


def get_references() -> dict[str, float]:
    """Expose key reference values used across the educational mode."""

    return {
        "ea": 320.0,
        "seg_55": _SEGMENT_REFERENCES[55],
        "seg_70": _SEGMENT_REFERENCES[70],
        "seg_90": _SEGMENT_REFERENCES[90],
        "n": _BLAINE_REFERENCE.n_est,
    }
