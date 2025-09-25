# Ogum ML Lite

[![CI](https://github.com/huyraestevao/ogum-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/huyraestevao/ogum-ml/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open In Colab](https://colab.research.googleusercontent.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huyraestevao/ogum-ml/blob/main/ogum-ml-lite/notebooks/ogum_ml_demo.ipynb)

Bootstrap do toolkit **Ogum ML Lite** para cálculos de θ(Ea) e Master Sintering
Curves (MSC) compatíveis com o ecossistema Ogum 6.4. O objetivo é disponibilizar
um pacote Python leve, fácil de usar em Google Colab e pronto para integrações
com pipelines de ML.

## Visão geral

- **θ(Ea) rápido**: cálculo direto a partir de ensaios de sinterização via
  `OgumLite.compute_theta_table` ou CLI.
- **MSC robusta**: métrica segmentada (55–70–90%) para avaliar o colapso das
  curvas por amostra.
- **Compatibilidade**: nomes de colunas (`sample_id`, `time_s`, `temp_C`,
  `rho_rel`) alinhados com Ogum 6.4 e notebooks do repositório
  [ogumsoftware](https://github.com/huyraestevao/ogumsoftware).
- **Pronto para ML**: módulo `ml_hooks` reservado para integração de pipelines e
  `features` com engenharia de atributos por amostra.

## Instalação

```bash
conda env create -f environment.yml
conda activate ogum-ml-lite

pip install -e .[dev]
```

## Formato de dados (long)

Os ensaios devem estar em formato *long* com uma linha por instante de tempo:

| sample_id | time_s | temp_C | rho_rel |
|-----------|--------|--------|---------|
| S0        | 0.0    | 25.0   | 0.10    |
| S0        | 60.0   | 215.0  | 0.24    |
| S1        | 0.0    | 25.0   | 0.08    |
| S1        | 60.0   | 210.0  | 0.21    |

Outras colunas (pressão, estágio, etc.) podem ser preservadas — apenas os
nomes acima são obrigatórios para o pipeline mínimo.

## Uso rápido (CLI)

```bash
# Engenharia de atributos por amostra
python -m ogum_lite.cli features \
  --input data/ensaios_long.csv \
  --ea "200,300,400" \
  --output exports/features.csv

# Tabelas θ(Ea) por ponto (formato longo)
python -m ogum_lite.cli theta \
  --input data/ensaios_long.csv \
  --ea "200,300,400" \
  --outdir exports/

# Curva Mestra com métrica segmentada robusta
python -m ogum_lite.cli msc \
  --input data/ensaios_long.csv \
  --ea "200,300,400" \
  --metric segmented \
  --csv exports/msc_curve.csv \
  --png exports/msc.png

# UI experimental em Gradio
python -m ogum_lite.cli ui
```

O comando `msc` imprime a tabela de métricas (`mse_global`, `mse_segmented`,
`mse_0.55_0.70`, `mse_0.70_0.90`), destaca o melhor Ea e exporta a curva mestra
normalizada.

## Fluxo em Colab

1. Abra o notebook de exemplo clicando no badge **Open In Colab** acima.
2. Execute a primeira célula para gerar um CSV sintético ou carregue o seu
   arquivo (`sample_id,time_s,temp_C,rho_rel`).
3. Use as células seguintes para gerar `features.csv`, `msc_curve.csv` e o
   gráfico `msc.png` diretamente no Colab.
4. Opcional: rode `python -m ogum_lite.cli ui` em uma célula para acessar a UI
   interativa.

## Integração com Ogum 6.4

- Mesma nomenclatura de colunas e estágios.
- Resultados (θ(Ea) e MSC) podem ser exportados em CSV para ingestão no Ogum 6.4.
- Futuras versões utilizarão `ml_hooks` para acoplamento com pipelines
  existentes do Ogum e notebooks do repositório ogumsoftware.

## Desenvolvimento

Executar checagens locais antes de enviar contribuições:

```bash
ruff check .
black --check .
pytest -q
```

CI via GitHub Actions executa o mesmo pipeline de lint e testes.

## Estrutura

```
ogum_lite/
  cli.py          # CLI com subcomandos features, theta, msc, ui
  features.py     # Engenharia de atributos por amostra
  theta_msc.py    # θ(Ea), Master Sintering Curve e métricas robustas
  ml_hooks.py     # Stubs para integração futura com pipelines de ML
notebooks/
  ogum_ml_demo.ipynb  # Fluxo mínimo em Colab com CLI embutida
tests/
  test_smoke.py, test_features.py, test_msc.py
```

## Roadmap (ml_hooks)

- Registrar pipelines de classificação/regressão (`register_pipeline`).
- Carregar modelos treinados (joblib/MLflow) via `load_pipeline`.
- Padronizar contratos de entrada/saída para integração com Ogum completo.
