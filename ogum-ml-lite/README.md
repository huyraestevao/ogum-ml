# Ogum ML Lite

[![CI](https://github.com/huyraestevao/ogum-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/huyraestevao/ogum-ml/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open In Colab](https://colab.research.googleusercontent.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huyraestevao/ogum-ml/blob/main/ogum-ml-lite/notebooks/ogum_ml_demo.ipynb)

Bootstrap do toolkit **Ogum ML Lite** para cálculos de θ(Ea), derivadas
cinéticas, ajustes Arrhenius e Master Sintering Curves (MSC) compatíveis com o
ecossistema Ogum 6.4. O objetivo é disponibilizar um pacote Python leve, fácil
de usar em Google Colab e pronto para integrações com pipelines de ML.

## Visão geral

- **Pré-processamento completo**: mapeamento automático de planilhas
  (.csv/.xls/.xlsx), normalização de unidades e metadados obrigatórios
  (`composition`, `technique`).
- **Derivadas e Arrhenius**: suavização Savitzky–Golay/média móvel, cálculo de
  `dy/dt`, `dT/dt`, razões Arrhenius e ajustes globais/por estágio.
- **θ(Ea) rápido**: cálculo direto a partir de ensaios de sinterização via
  `OgumLite.compute_theta_table` ou CLI.
- **MSC robusta**: métrica segmentada (55–70–90%) para avaliar o colapso das
  curvas por amostra.
- **Compatibilidade**: nomes de colunas (`sample_id`, `time_s`, `temp_C`,
  `rho_rel`) alinhados com Ogum 6.4 e notebooks do repositório
  [ogumsoftware](https://github.com/huyraestevao/ogumsoftware).
- **Pronto para ML**: módulo `features` com engenharia global e stage-aware,
  integração com `theta_msc` e `ml_hooks` para pipelines supervisionados.

## Instalação

```bash
conda env create -f environment.yml
conda activate ogum-ml-lite

pip install -e .[dev]
```

## Formato de dados (long)

Os ensaios devem estar em formato *long* com uma linha por instante de tempo e
metadados mínimos:

| Coluna       | Descrição                                        |
|--------------|--------------------------------------------------|
| `sample_id`  | Identificador único da amostra/ensaio             |
| `time`       | Tempo em segundos ou minutos (`time_s`/`time_min`)|
| `temp`       | Temperatura em °C ou K (`temp_C`/`temp_K`)        |
| `response`   | `rho_rel`/`shrinkage_rel` (0–1)                   |
| `composition`| Legenda da liga/composição                       |
| `technique`  | Rota de sinterização (dropdown Ogum 6.4)          |

Outras colunas (pressão, estágio, etc.) podem ser preservadas — o mapeamento
automático permite alinhar planilhas heterogêneas aos nomes canônicos.

## Pipeline (Fase 3.1)

Fluxo recomendado:

1. **Importar planilha** (`io map`): inferir/ajustar o mapeamento de colunas e
   normalizar unidades.
2. **Pré-processar** (`preprocess derive`): aplicar o mapeamento, suavizar e
   gerar as derivadas (`dy/dt`, `dT/dt`, `T·dy/dt`, ...).
3. **Arrhenius** (`arrhenius fit`): calcular Ea global, por estágios padrão
   (55–70%, 70–90%) e exportar a regressão `ln(T·dy/dt)` vs `1/T`.
4. **Feature store** (`features build`): consolidar métricas globais + por
   estágio + Ea Arrhenius + θ(Ea) opcionais para ML.
5. **MSC/θ** (`msc`, `theta`): continuar com os módulos clássicos se necessário.

Outras colunas (pressão, estágio, etc.) podem ser preservadas — apenas os
nomes acima são obrigatórios para o pipeline mínimo.

## Uso rápido (CLI)

```bash
# 1) Inferir mapeamento e salvar JSON
python -m ogum_lite.cli io map \
  --input data/ensaio.xlsx \
  --out artifacts/mapping.json \
  --edit

# 2) Aplicar mapeamento + derivadas
python -m ogum_lite.cli preprocess derive \
  --input data/ensaio.xlsx \
  --map artifacts/mapping.json \
  --smooth savgol \
  --out exports/derivatives.csv

# 3) Ajustes Arrhenius (global + estágios)
python -m ogum_lite.cli arrhenius fit \
  --input exports/derivatives.csv \
  --stages "0.55-0.70,0.70-0.90" \
  --out exports/arrhenius.csv \
  --png exports/arrhenius.png

# 4) Feature store stage-aware + θ(Ea)
python -m ogum_lite.cli features build \
  --input exports/derivatives.csv \
  --stages "0.55-0.70,0.70-0.90" \
  --theta-ea "200,250,300" \
  --out exports/feature_store.csv

# Comandos legados
python -m ogum_lite.cli features --input ...
python -m ogum_lite.cli theta --input ...
python -m ogum_lite.cli msc --input ...
python -m ogum_lite.cli ui
```

O comando `msc` imprime a tabela de métricas (`mse_global`, `mse_segmented`,
`mse_0.55_0.70`, `mse_0.70_0.90`), destaca o melhor Ea e exporta a curva mestra
normalizada.

## ML (Fase 3)

Os comandos abaixo fecham o ciclo dados → features → ML com validação
estratificada por `sample_id` (GroupKFold) e artefatos persistidos em disco.

```bash
# 1) Gerar tabela de features (mesma assinatura do comando base)
python -m ogum_lite.cli ml features \
  --input data/ensaios_long.csv \
  --ea "200,300,400" \
  --output exports/features.csv

# 2) Treinar classificador (ex.: técnica de sinterização)
python -m ogum_lite.cli ml train-cls \
  --table exports/features.csv \
  --target technique \
  --group-col sample_id \
  --features heating_rate_med_C_per_s T_max_C y_final t_to_90pct_s theta_Ea_200kJ theta_Ea_300kJ \
  --outdir artifacts/cls_technique

# 3) Treinar regressor (ex.: T90_C)
python -m ogum_lite.cli ml train-reg \
  --table exports/features.csv \
  --target T90_C \
  --group-col sample_id \
  --features heating_rate_med_C_per_s T_max_C theta_Ea_300kJ theta_Ea_400kJ \
  --outdir artifacts/reg_T90

# 4) Predizer a partir de um artefato salvo
python -m ogum_lite.cli ml predict \
  --table exports/features.csv \
  --model artifacts/cls_technique/classifier.joblib \
  --out exports/preds_cls.csv

# 5) Clusterização exploratória (KMeans)
python -m ogum_lite.cli ml cluster \
  --table exports/features.csv \
  --features heating_rate_med_C_per_s T_max_C theta_Ea_300kJ \
  --k 3 \
  --out exports/clusters.csv
```

Durante o treino são impressas as métricas médias/desvio obtidas via
`GroupKFold`, garantindo que nenhuma amostra (`sample_id`) aparece em múltiplos
folds. Os artefatos exportados incluem `.joblib`, `feature_cols.json`,
`target.json` e `model_card.json` com timestamp, hiperparâmetros e métricas.

Targets canônicos compatíveis com Ogum 6.4:

| Tarefa           | Coluna alvo            |
|------------------|------------------------|
| Classificação    | `technique` (Conventional, UHS, FS, SPS, …) |
| Regressão        | `T90_C`, `rho_final`, `Ea_app_kJmol`        |

As features geradas automaticamente incluem métricas globais
(`heating_rate_med_C_per_s`, `T_max_C`, `y_final`, `t_to_90pct_s`, `dy_dt_max`,
`dT_dt_max`, `T_at_dy_dt_max_C`), colunas segmentadas (`*_s1`, `*_s2`) e Ea
Arrhenius (`Ea_arr_global_kJ`, `Ea_arr_55_70_kJ`, …), além das integrais
`theta_Ea_*` quando solicitadas.

## Tuning & Reports (Fase 4)

Novos comandos do grupo `ml` completam o ciclo **train → tune → explicar →
documentar** com artefatos prontos para revisão:

```bash
# RandomizedSearchCV para classificação (RandomForest + GroupKFold)
python -m ogum_lite.cli ml tune-cls \
  --table exports/features.csv \
  --target technique \
  --group-col sample_id \
  --features heating_rate_med_C_per_s T_max_C theta_Ea_300kJ theta_Ea_400kJ \
  --outdir artifacts/cls_technique \
  --n-iter 40 --cv 5

# RandomizedSearchCV para regressão (RandomForestRegressor)
python -m ogum_lite.cli ml tune-reg \
  --table exports/features.csv \
  --target T90_C \
  --group-col sample_id \
  --features heating_rate_med_C_per_s T_max_C theta_Ea_300kJ theta_Ea_400kJ \
  --outdir artifacts/reg_T90 \
  --n-iter 40 --cv 5

# Relatório HTML com métricas, gráficos e model card atualizado
python -m ogum_lite.cli ml report \
  --table exports/features.csv \
  --target technique \
  --group-col sample_id \
  --model artifacts/cls_technique/classifier_tuned.joblib \
  --outdir artifacts/cls_technique \
  --notes "Resultados validados em 2024-05"
```

Cada execução cria/atualiza os artefatos em `outdir`:

- `*_tuned.joblib`: pipeline completo com pré-processamento e melhor estimador;
- `param_grid.json` + `cv_results.json`: espaço de busca e resultados da busca;
- `model_card.json`: inclui histórico (`history`) com tuning, `best_params` e
  métricas médias (accuracy/F1 ou MAE/RMSE);
- `feature_importance.png`, `confusion_matrix.png`/`regression_scatter.png` e
  `report.html` com métricas, figuras embutidas (base64) e observações.

Checklist rápido de boas práticas:

1. **GroupKFold obrigatório**: sempre informar `--group-col sample_id` para
   evitar vazamento entre ensaios do mesmo corpo de prova.
2. **Features limpas**: garantir que `feature_cols.json` reflita exatamente as
   colunas usadas; o comando de tuning atualiza o arquivo automaticamente.
3. **Relatórios versionados**: utilize `--notes` para registrar contexto
   (experimento, data, analista) direto no `report.html` e no `model_card`.
4. **Reprodutibilidade**: fixe `--random-state` quando comparar execuções ou
   compartilhar notebooks/artefatos com terceiros.

## Fluxo em Colab

1. Abra o notebook de exemplo clicando no badge **Open In Colab** acima ou no
   novo `notebooks/ogum_ml_derivatives_demo.ipynb`.
2. Execute a primeira célula para mapear a planilha e gerar `derivatives.csv`.
3. Use as células seguintes para obter `arrhenius.csv`, `feature_store.csv` e a
   curva MSC.
4. Opcional: rode `python -m ogum_lite.cli ui` em uma célula para acessar a UI
   interativa com mapeamento assistido.

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
  cli.py          # CLI com subcomandos features, theta, msc, ui e ml/*
  features.py     # Engenharia de atributos por amostra
  ml_hooks.py     # Pipelines sklearn (classificação, regressão, cluster)
  theta_msc.py    # θ(Ea), Master Sintering Curve e métricas robustas
artifacts/        # Diretório sugerido para modelos e relatórios de ML
notebooks/
  ogum_ml_demo.ipynb       # Fluxo mínimo θ(Ea)/MSC em Colab
  ogum_ml_ml_demo.ipynb    # Exemplo end-to-end de ML (Fase 3)
tests/
  test_features.py, test_msc.py, test_ml_pipeline.py, test_smoke.py
```

## Roadmap (ml_hooks)

- Registrar pipelines de classificação/regressão (`register_pipeline`).
- Carregar modelos treinados (joblib/MLflow) via `load_pipeline`.
- Padronizar contratos de entrada/saída para integração com Ogum completo.
