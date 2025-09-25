# Ogum ML Lite

Bootstrap do toolkit **Ogum ML Lite** para cálculos de θ(Ea) e Master Sintering
Curves (MSC) compatíveis com o ecossistema Ogum 6.4. O objetivo é disponibilizar
um pacote Python leve, fácil de usar em Google Colab e pronto para integrações
com pipelines de ML.

## Visão geral

- **θ(Ea) rápido**: cálculo direto a partir de ensaios de sinterização via
  `OgumLite.compute_theta_table` ou CLI.
- **MSC Colab-friendly**: geração de curvas mestras sem dependências pesadas.
- **Compatibilidade**: nomes de colunas (`time_s`, `temperature_C`,
  `densification`, `datatype`) e estágios alinhados com o Ogum 6.4 e com as
  diretrizes do repositório [ogumsoftware](https://github.com/huyraestevao/ogumsoftware).
- **Pronto para ML**: módulo `ml_hooks` reservado para integração de pipelines e
  `features` com TODOs explícitos.

## Instalação

```bash
conda env create -f environment.yml
conda activate ogum-ml-lite
```

Para instalar em modo editável:

```bash
pip install -e .[dev]
```

## Uso rápido

### CLI

Calcular θ(Ea) e exportar CSV:

```bash
python -m ogum_lite.cli theta --input ensaios.csv --ea "200,300,400" --outdir exports/
```

Gerar MSC, exportando CSV e PNG:

```bash
python -m ogum_lite.cli msc --input ensaios.csv --ea 200 --png msc.png --csv msc.csv
```

Abrir interface Gradio:

```bash
python -m ogum_lite.cli ui
```

### Fluxo em Colab

1. Clone ou sincronize o repositório na sessão.
2. Instale dependências mínimas (`pip install ogum-ml-lite` via arquivo wheel ou
   modo editável).
3. Carregue datasets com as colunas padronizadas e utilize `OgumLite` para
   realizar cortes (`select_datatype`, `cut_by_time`, `baseline_shift`), calcular
   θ(Ea) e gerar MSC.
4. Utilize os notebooks em `notebooks/` como referência para manter o estilo de
   documentação e organização do Ogum 6.4.

### Integração com Ogum 6.4

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
  theta_msc.py    # Classe principal OgumLite
  features.py     # TODOs para métricas derivadas
  ml_hooks.py     # Pontos de integração com ML
notebooks/
  README.md       # Diretrizes para notebooks alinhados ao Ogum 6.4
tests/
  test_smoke.py   # Smoke test com dados sintéticos
```
