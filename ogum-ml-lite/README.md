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
- **Segmentação automática**: CLI `segmentation` com limiares 55–70–90% ou
  modo data-driven simples.
- **Mudança de mecanismo**: ajuste piecewise linear com AIC/BIC via
  `cli mechanism`.
- **Compatibilidade**: nomes de colunas (`sample_id`, `time_s`, `temp_C`,
  `rho_rel`) alinhados com Ogum 6.4 e notebooks do repositório
  [ogumsoftware](https://github.com/huyraestevao/ogumsoftware).
- **Pronto para ML**: módulo `features` com engenharia global e stage-aware,
  integração com `theta_msc` e `ml_hooks` para pipelines supervisionados.

## Frontend Evolutivo (Fase 8)

O frontend agora usa uma arquitetura modular com design system leve, theming
dinâmico, i18n e camada de serviços compartilhada.

## Telemetria e Experimentos (Fase 10)

### Telemetria opcional

- Opt-in local pela sidebar ou via `OGUML_TELEMETRY=1`.
- Somente eventos técnicos (nome da etapa, duração, variante A/B, acertos de cache) são gravados em `workspace/telemetry.jsonl`.
- Agregue e limpe métricas com o utilitário:

```bash
python -m app.services.cli_tools telemetry aggregate --file workspace/telemetry.jsonl --out telemetry_summary.json
python -m app.services.cli_tools telemetry clean --file workspace/telemetry.jsonl
```

### Experimentos A/B de UX

- Experimentos ativos: `wizard_vs_tabs`, `msc_controls_layout`, `run_buttons_position`.
- Atribuição sticky por sessão com coleta automática nas ações do wizard.
- Exporte contagens por variante:

```bash
python -m app.services.cli_tools ab export --file workspace/telemetry.jsonl --out ab_summary.json
```

### Temas personalizados

- Temas base em `app/config/themes/` (`base.yaml`, `dark.yaml`, `custom.example.yaml`).
- Faça upload de um YAML pela sidebar ou adicione o arquivo ao diretório para disponibilizá-lo.
- A função `get_theme(dark, override)` mescla overrides com o tema base.

### Perfis de execução por técnica

- Perfis YAML em `app/config/profiles/` (`conventional`, `fs_uhs`, `sps`).
- Seleção pela sidebar aplica presets de Ea, métricas MSC, features elétricas etc.
- O diff aplicado é exibido como JSON resumido logo abaixo da seleção.

### Cache e Performance

- Cache em disco por hash de entradas idempotentes (prep, features, MSC) em `workspace/.cache/`.
- Instrumentação com `@profile_step` gera duração/memória, exibidos tanto no log quanto na telemetria.
- Inspecione ou limpe o cache via CLI:

```bash
python -m app.services.cli_tools cache stats --dir workspace/.cache
python -m app.services.cli_tools cache purge --dir workspace/.cache
```

### Streamlit (painel principal)

```bash
streamlit run app/streamlit_app.py
```

- **Layout modular**: shell comum em `app/design/layout.py` com sidebar para
  workspace/preset, header com toggle de tema e export, e páginas em
  `app/pages/*.py`.
- **Theming**: `app/design/theme.py` expõe `get_theme(dark)` para claro/escuro;
  alternância em tempo real sem recarregar o app.
- **i18n**: seletor de idioma (pt/en) alimenta `app/i18n/translate.py` com
  catálogos JSON; textos respondem imediatamente à troca de locale.
- **Services**: todas as ações orquestram a CLI via `app/services/run_cli.py`
  (tenacity + logs + telemetria) reaproveitando `ogum_lite.ui.orchestrator`.
- **UX**: toasts, validações, barras de progresso e previews foram integrados às
  páginas de Prep, Features, θ/MSC, Segmentação, Mecanismo, ML e Export.
- **Estado**: `app/services/state.py` centraliza `session_state`, workspace e
  registro de artefatos; telemetria opcional (`OGUML_TELEMETRY=0` desliga).

## Fase 17 — Modo Educacional

- **Como ativar**: escolha "Modo Educacional" no menu lateral do Streamlit. O fluxo traz três passos (Carregar Dados → MSC & n → Explorar/Exportar).
- **Conteúdo incluso**: cards conceituais bilingues (pt/en), simulações interativas de θ(Ea), colapso MSC e linearização de Blaine, além de exercícios com checagem automática.
- **Export educacional**: gere um HTML estático sempre disponível; se instalar `pip install "ogum-ml[pdf]"`, o botão de PDF usa ReportLab para montar o resumo com gráficos.
- **Limitações**: focado em treinamento. Usa dataset exemplo se nenhum CSV for enviado e não substitui as abas avançadas para análises completas.
- **Novos exercícios**: implemente funções em `app/edu/exercises.py` retornando `Exercise(key, statement_md, inputs_spec, evaluate)`. O método `evaluate` deve receber um `Mapping[str, float]` e retornar `{"score": float, "feedback": str}`.

### Modo guiado (Wizard)

- Disponível no menu lateral como primeira opção (`Modo guiado`).
- Bloqueia a navegação enquanto os artefatos obrigatórios não estiverem prontos e exibe toasts `[ok]/[warn]`.
- Cada etapa reutiliza os serviços existentes (`run_cli`) com tooltips, descrições acessíveis e persistência em `session_state`.
- Documentação completa de UX e microcópia em [`docs/DESIGN_SPEC_UX.md`](docs/DESIGN_SPEC_UX.md) e [`docs/MICROCOPY_*.yaml`](docs).

#### Como estender com nova página

1. Crie `app/pages/page_nova.py` com `render(translator: I18N) -> None`.
2. Use componentes do design system (`card`, `alert`, `toolbar`) e serviços.
3. Registre a página em `PAGES` dentro de `app/streamlit_app.py`.
4. Adicione microcópia em `docs/MICROCOPY_*.yaml` e traduções em `app/i18n/locales/*.json`.
5. Rode `python -m app.services.i18n_lint` e `pytest -q` para garantir cobertura.

#### Qualidade contínua

- `python -m app.services.i18n_lint` garante paridade de chaves pt/en.
- `python -m app.services.linkcheck` valida links internos do README e da Design Spec.
- `pytest -q` cobre o fluxo guiado, helpers de acessibilidade e integrações básicas.

### Gradio (fallback)

```bash
python app/gradio_app.py
```

Interface Blocks compacta que reutiliza `app/services/run_cli.py` para rodar o
pipeline e gerar o ZIP com os artefatos do workspace.

## Fase 11 — Experimentação Avançada

### Modelos opcionais

Os novos benchmarks suportam modelos adicionais de gradient boosting sem
depender deles por padrão. Instale apenas o que precisar:

```bash
pip install "ogum-ml[lgbm]"  # LightGBM
pip install "ogum-ml[cat]"   # CatBoost
pip install "ogum-ml[xgb]"   # XGBoost
```

Sem essas extras, os modelos continuam disponíveis via Random Forest.

### Benchmark padronizado

Use `ml bench` para executar a matriz `targets × feature_sets × modelos`, com
GroupKFold por `sample_id` e pipelines consistentes (StandardScaler +
OneHotEncoder + estimador). Um exemplo de classificação:

```bash
python -m ogum_lite.cli ml bench \
  --table features.csv \
  --task cls \
  --targets technique \
  --feature-sets basic="heating_rate_med_C_per_s,T_max_C,y_final,t_to_90pct_s" \
                  theta="theta_Ea_200kJ,theta_Ea_300kJ,theta_Ea_400kJ" \
  --models rf,lgbm,cat,xgb \
  --group-col sample_id \
  --outdir artifacts/bench_cls
```

Cada combinação gera uma pasta `<outdir>/<target>/<feature_set>/<model>/` com
`model.joblib`, `feature_cols.json`, `target.json`, `cv_metrics.json`,
`model_card.json` e `training_log.json`. As métricas consolidadas ficam em
`bench_results.csv`.

### Comparação e ranking

Depois de rodar o benchmark, gere o resumo com `ml compare`:

```bash
python -m ogum_lite.cli ml compare \
  --bench-csv artifacts/bench_cls/bench_results.csv \
  --task cls \
  --outdir artifacts/bench_cls
```

O comando cria `bench_summary.csv` (ranking por alvo/conjunto de features) e
`ranking.png` (média geral). Modelos indisponíveis são ignorados automaticamente
com aviso `[skip]`.

> 💡 Para execuções rápidas em datasets grandes, reduza `n_estimators` ajustando
> os parâmetros ao instanciar os pipelines (ex.: `--models rf` para testes
> rápidos ou adaptando o código/factory antes da rodada final).

## Fase 12 — Scheduler & Jobs

### Visão geral

- **Jobs assíncronos**: qualquer comando da CLI pode ser enviado para execução
  em segundo plano (`jobs submit --cmd "ml bench ..."`).
- **Persistência**: metadados e estados ficam em `workspace/jobs/jobs.json`, com
  diretórios individuais `workspace/jobs/<job_id>/` para logs e artefatos.
- **Agendamento**: use `--at "2025-01-10 12:00"` para agendar execuções
  futuras.

## Fase 13 — Comparador de Execuções

O módulo de comparação facilita auditorias entre execuções do Ogum-ML,
considerando presets, MSC, segmentos, mecanismo e artefatos de ML.

### CLI dedicada

Compare duas execuções (diretórios ou export ZIP):

```bash
python -m ogum_lite.cli compare runs \
  --a artifacts/bench_cls/run_A \
  --b artifacts/bench_cls/run_B \
  --outdir artifacts/compare_cls_A_vs_B
```

Para matrizes (referência × candidatos) com relatórios individuais ref vs
cada candidato:

```bash
python -m ogum_lite.cli compare matrix \
  --ref artifacts/bench_reg/run_ref \
  --candidates artifacts/bench_reg/run_1 artifacts/bench_reg/run_2 \
  --outdir artifacts/compare_reg_matrix
```

Resultados:

- `compare_summary.json` (manifest + deltas consolidados).
- `report.html` com gráficos inline (base64) e destaques principais.
- `report.xlsx` (abas Summary, Diff-Presets, Diff-ML, etc.).
- Para matriz: `ranking.csv` + `matrix.html` e subpastas com relatórios
  individuais.

> 💡 Boas práticas: mantenha `report.html`/`report.xlsx` ao lado de cada run e
> use nomes descritivos para `--outdir` (ex.: `compare_cls_2025-01_vs_02`).

### Página Streamlit “Compare Runs”

- Acesse via menu principal após abrir `streamlit run app/streamlit_app.py`.
- Selecione o modo **Runs** (A × B) ou **Matrix** (referência vs lista).
- Informe caminhos/diretórios do workspace ou faça upload prévio do ZIP.
- Clique em **Compare** / **Build matrix** para gerar os relatórios.
- Baixe os arquivos diretamente pelos botões exibidos na página ou visualize o
  HTML inline.
- **Monitoramento unificado**: Streamlit ganha a página "Jobs Monitor" e o
  Gradio recebe a aba "Jobs" para acompanhar execuções, visualizar logs e
  cancelar jobs.

### CLI

```bash
# Submeter um benchmark de ML em background
python -m ogum_lite.cli jobs submit \
  --cmd "ml bench --table features.csv --task cls --targets technique --models rf"

# Agendar para o futuro (UTC)
python -m ogum_lite.cli jobs submit \
  --cmd "ml bench --table features.csv --task cls --targets technique --models rf" \
  --at "2025-01-10 12:00"

# Listar jobs e estados atuais
python -m ogum_lite.cli jobs status

# Visualizar o tail do log
python -m ogum_lite.cli jobs logs <job_id>

# Cancelar execução em andamento
python -m ogum_lite.cli jobs cancel <job_id>
```

### Frontend

- **Streamlit**: a página "Jobs Monitor" lista todos os jobs, atualiza a cada
  poucos segundos e permite abrir o log ou cancelar execuções ativas.
- **Gradio**: a aba "Jobs" replica as ações principais (listar, ver log e
  cancelar) dentro da interface Blocks.

> ℹ️ Todos os artefatos gerados por jobs ficam em `workspace/jobs/<job_id>/`,
> mantendo o histórico completo da execução.

## Fase 14 — Ponte FEM/Simulação

Uma camada opcional conecta resultados de simulação numérica (FEM térmico,
co-sinterização, campos elétricos) ao pipeline de features do Ogum-ML.

### Formato canônico do bundle

`sim import` gera um diretório com:

- `meta.json`: solver, versão, unidades por campo (ex.: `temp_C → C`).
- `mesh.json`: contagem de nós/células e tipo de elemento (`tri`, `tet`, ...).
- `times.npy`: série temporal em segundos.
- `manifest.json`: mapeia campos → arquivos salvos.
- `node_fields/<campo>_NNN.npy` e `cell_fields/<campo>_NNN.npy`: arrays por
  passo de tempo.

Os arquivos podem ser versionados ou movidos entre workspaces sem dependências
extras.

### Ingestão

```bash
# Séries VTK (requer meshio via extra opcional)
python -m ogum_lite.cli sim import --src data/sim/vtk_series --format vtk --outdir artifacts/sim/run_vtk

# XDMF único (idem meshio)
python -m ogum_lite.cli sim import --src data/sim/results.xdmf --format xdmf --outdir artifacts/sim/run_xdmf

# CSV com séries escalares (sem dependências externas)
python -m ogum_lite.cli sim import --src data/sim/summary.csv --format csv --outdir artifacts/sim/run_csv
```

VTK/XDMF usam `meshio`; instale com `pip install "ogum-ml[sim]"`. Ausente o
pacote, as funções exibem erro orientando a instalação, enquanto CSV continua
funcionando normalmente.

### Features globais e segmentadas

```bash
python -m ogum_lite.cli sim features \
  --bundle artifacts/sim/run_csv \
  --segments "0,120;120,240" \
  --out artifacts/features_sim.csv
```

O CSV resultante contém métricas globais (`T_max_C`, `t_at_T_max_s`,
`sigma_vm_max_MPa`, integrais de tempo) e colunas extras por segmento com o
sufixo `segment_<idx>_<t0>_<t1>`.

### Vincular a runs reais

```bash
python -m ogum_lite.cli sim link \
  --exp-features artifacts/features_exp.csv \
  --sim-features artifacts/features_sim.csv \
  --map link.yaml \
  --out artifacts/features_exp_sim.csv
```

O arquivo `link.yaml` define `sample_id: sim_id`; se ausente, o comando tenta
coincidir nomes de arquivos/pastas. As colunas de simulação são sufixadas com
`_sim`, prontas para entrar em `features build` ou ML.

### Notas de unidades e export

- Temperaturas em Kelvin são convertidas automaticamente para °C (`temp_C`).
- Campos elétricos → `E_V_per_m`; tensões de von Mises → `von_mises_MPa`.
- Recomendação: exportar `temp`, `sigma_vm`, `electric_field` e escala de tempo
  em segundos a partir do solver.

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

# 5) Segmentação automática (fixed/data)
python -m ogum_lite.cli segmentation \
  --input exports/derivatives.csv \
  --mode fixed \
  --out artifacts/segments.json

# 6) Detecção de mudança de mecanismo (θ vs densificação)
python -m ogum_lite.cli mechanism \
  --theta exports/theta_Ea_300kJ.csv \
  --out artifacts/mechanism.csv

# Comandos legados
python -m ogum_lite.cli features --input ...
python -m ogum_lite.cli theta --input ...
python -m ogum_lite.cli msc --input ...
python -m ogum_lite.cli ui
```

O comando `msc` imprime a tabela de métricas (`mse_global`, `mse_segmented`,
`mse_0.55_0.70`, `mse_0.70_0.90`), destaca o melhor Ea e exporta a curva mestra
normalizada.

## Mapas de sinterização (Blaine n / MSE)

O CLI `maps` gera heatmaps Matplotlib a partir da tabela retornada por
`segment_feature_table`/feature store. Os arquivos `blaine_n_heatmap.png` e
`blaine_mse_heatmap.png` são salvos no diretório indicado (por padrão `maps/`).

```bash
python -m ogum_lite.cli maps \
  --input exports/segment_feature_table.csv \
  --outdir artifacts/maps
```

## API FastAPI + Docker

Suba a API localmente (porta 8000) com auto-reload opcional:

```bash
python -m ogum_lite.cli api --host 0.0.0.0 --port 8000 --reload
```

Os endpoints disponíveis incluem `/prep`, `/features`, `/msc`, `/segmentation`,
`/mechanism`, `/ml/train` e `/ml/predict`. A documentação interativa pode ser
acessada em `http://localhost:8000/docs`.

Para executar via Docker, construa a imagem e exponha a porta da API:

```bash
docker build -t ogum-ml-lite -f docker/Dockerfile .
docker run --rm -p 8000:8000 ogum-ml-lite
```

O container inicia a API automaticamente, permitindo testar os endpoints via
`curl` ou navegando até `/docs`.

## Validação de dados

Antes de treinar modelos ou calcular θ(Ea), valide os insumos:

```bash
# Ensaios longos (sample_id,time_s,temp_C,rho_rel/shrinkage_rel)
python -m ogum_lite.cli validate long \
  --input data/ensaios_long.csv \
  --y-col rho_rel \
  --out artifacts/validation_long.json

# Tabela de features derivadas
python -m ogum_lite.cli validate features \
  --table exports/features.csv \
  --out artifacts/validation_features.json
```

O relatório JSON lista colunas ausentes, valores fora de faixa e percentual de
dados nulos por coluna. No terminal é impresso um resumo com até 10 problemas
para inspeção rápida.

## Relatório XLSX consolidado

Monte um relatório executivo unindo métricas, curva MSC, tabela de features e
imagens (PNG) usando `export xlsx`:

```bash
python -m ogum_lite.cli export xlsx \
  --out reports/ogum_report.xlsx \
  --msc exports/msc_curve.csv \
  --features exports/features.csv \
  --metrics artifacts/cls_technique/model_card.json \
  --dataset "Campanha Ogum 2024" \
  --notes "Resultados finais após tuning" \
  --img-msc figures/msc.png \
  --img-cls figures/confusion.png \
  --img-reg figures/regression.png
```

O arquivo final contém abas **Summary**, **MSC**, **Features** e **Metrics** com
metadados (dataset, timestamps, melhores métricas) e imagens opcionais.

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

## Docker

Para empacotar a CLI/UI em um contêiner leve:

```bash
docker build -t ogum-ml:latest -f docker/Dockerfile .
docker run --rm -v "$PWD":/work ogum-ml:latest --help
```

O contêiner inicia com `python -m ogum_lite.cli` como entrypoint; basta montar o
diretório com dados e escolher o subcomando desejado (`ui`, `msc`, `ml ...`).

## Release e distribuição

O workflow `release.yml` empacota a wheel e publica ativos extras sempre que uma
tag `v*` é enviada ao repositório remoto:

```bash
git tag v0.2.0
git push origin v0.2.0
```

Além da wheel (`dist/*.whl`), arquivos em `artifacts/**` são anexados à Release.

## Export ONNX (opcional)

Para gerar modelos compatíveis com inferência embarcada, instale as dependências
opcionais e use o novo comando:

```bash
pip install skl2onnx onnxmltools

python -m ogum_lite.cli export onnx \
  --model artifacts/cls_technique/classifier_tuned.joblib \
  --features-json artifacts/cls_technique/feature_cols.json \
  --out artifacts/cls_technique/model.onnx
```

Se as dependências não estiverem instaladas ou o estimador não for um
`RandomForest*`, o comando é ignorado com um aviso (código de saída 0).

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

## Fase 15 — Deploy Colaborativo

Implemente e rode o backend FastAPI + frontend Streamlit em modo multiusuário.

### Componentes

- **API FastAPI** (`server.api_main`): autenticação JWT, runs, jobs, comparações e registro de simulações.
- **Frontend Streamlit** (`app/streamlit_server.py`): abas para login, runs, jobs, comparações e simulações.
- **Storage centralizado** (`server_storage/`): diretórios `runs/`, `jobs/`, `sims/` e `users/` compartilhados via volume.
- **Reverse proxy** (Nginx) expondo `/` para o Streamlit e `/api/` para a API.

### Executar localmente

```bash
cd docker
docker-compose up --build
```

- API: http://localhost:8000/docs
- Frontend: http://localhost:8501
- Proxy unificado: http://localhost:8080 (frontend) e http://localhost:8080/api/ (API)

### Usuários e autenticação

1. Gere `server_storage/users/users.json` com o usuário inicial:

   ```bash
   cd ..  # diretório ogum-ml-lite
   python - <<'PY'
from pathlib import Path
from server import auth

users_dir = Path("server_storage/users")
users_dir.mkdir(parents=True, exist_ok=True)
payload = '[{"username":"admin","password_hash":"' + auth.hash_password("admin123") + '","role":"admin"}]'
(users_dir / "users.json").write_text(payload)
PY
   cd docker
   ```

2. Faça login via API:

   ```bash
   curl -X POST http://localhost:8000/auth/login \
     -H "Content-Type: application/json" \
     -d '{"username":"admin","password":"admin123"}'
   ```

3. Use o token retornado nas chamadas protegidas (`Authorization: Bearer <token>`).

### Rotas principais

- `POST /auth/login` – autenticação com usuário/senha.
- `GET /runs/list` – lista execuções disponíveis.
- `POST /runs/upload` – envia novo artefato de run.
- `GET /jobs/list` – status dos jobs monitorados.
- `POST /compare/run` – agenda comparação entre runs.
- `POST /sim/import` – registra bundle de simulação.

### Smoke tests

```bash
pytest -q tests/test_auth.py tests/test_server_api.py
```

Os testes criam storage temporário, validam login, listagem de runs, upload e registro de simulação.

## Fase 16 — Publicação Científica (Zenodo/Figshare)

### Tokens e variáveis de ambiente

- Crie um token pessoal no [Zenodo](https://zenodo.org/account/settings/applications/tokens/new/)
  (ou sandbox) e outro no [Figshare](https://figshare.com/account/applications).
- Copie-os para um arquivo `.env` baseado em [`.env.example`](../.env.example) ou
  para os segredos do ambiente onde o Ogum-ML roda:

  ```bash
  export ZENODO_TOKEN=seu_token
  export FIGSHARE_TOKEN=seu_token
  # Opcional: usar sandbox
  export ZENODO_BASE=https://sandbox.zenodo.org/api
  ```

### Metadados canônicos

Crie um YAML (ex.: `docs/publish_meta.yaml`) com autores, licença e demais
campos exigidos pelos repositórios:

```yaml
title: "Ogum Lite — Execução 2025-10-01"
description: |
  Resultados completos (artefatos, relatórios e modelos) para reprodução.
version: "v0.3.0"
authors:
  - name: "Alice Doe"
    affiliation: "Ogum Research Lab"
    orcid: "0000-0002-1825-0097"
keywords: ["sintering", "ogum", "ml"]
license: "CC-BY-4.0"
funding: ["CNPq-1234"]
related_identifiers:
  - identifier: "10.5281/zenodo.1234567"
    relation: "isSupplementTo"
upload_files:
  - docs/paper_preprint.pdf
community: "open-science"
category: "dataset"
```

### Fluxo CLI

1. **Preparar bundle** (gera `bundle.zip`, checksum e `publish_manifest.json`):

   ```bash
   python -m ogum_lite.cli publish prepare \
     --run-dir artifacts/run_2025_10_01 \
     --meta docs/publish_meta.yaml \
     --outdir artifacts/publish/run_2025_10_01
   ```

2. **Publicar no Zenodo** (usa `ZENODO_TOKEN`/`ZENODO_BASE`):

   ```bash
   python -m ogum_lite.cli publish zenodo \
     --bundle artifacts/publish/run_2025_10_01
   ```

3. **Publicar no Figshare** (usa `FIGSHARE_TOKEN`/`FIGSHARE_BASE`):

   ```bash
   python -m ogum_lite.cli publish figshare \
     --bundle artifacts/publish/run_2025_10_01
   ```

4. **Consultar recibos (DOI/URL)**:

   ```bash
   python -m ogum_lite.cli publish status \
     --bundle artifacts/publish/run_2025_10_01
   ```

Os recibos (`zenodo_receipt.json`, `figshare_receipt.json`) ficam ao lado do
bundle e podem ser versionados junto com o projeto.

### Versionamento e citação

- No Zenodo, use **New version** para atualizar um DOI existente mantendo o
  `conceptdoi`; use **New deposition** para coleções independentes.
- No Figshare, crie uma nova versão diretamente a partir do artigo
  publicado para preservar o histórico.
- Inclua o DOI nas capas de relatório, dashboards e no `model_card.json`
  gerado pelo pipeline para manter o rastro de citação.
