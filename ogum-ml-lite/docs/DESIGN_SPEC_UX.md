# Ogum-ML Lite — Design Spec UX (Fase 9)

## Personas e Metas

### Pesquisador de Materiais
- **Objetivo primário**: validar rapidamente curvas de sinterização, extrair θ(Ea) e gerar relatórios para discussões técnicas.
- **Necessidades**: feedback imediato de validação, presets confiáveis, possibilidade de explorar dados etapa a etapa.
- **Pontos de dor**: formatos heterogêneos de planilhas, tempo limitado para reconstruir análises.

### Engenheiro de Processos / Dados
- **Objetivo primário**: orquestrar pipelines completos (prep → ML) e padronizar artefatos para reuso.
- **Necessidades**: automação guiada, microcópia objetiva para diminuir ambiguidades, export consistente.
- **Pontos de dor**: transição entre scripts e UI, rastreabilidade dos artefatos, garantia de cobertura de traduções.

## Fluxos Principais
1. **Prep** → upload/seleção de CSV, validação e normalização.
2. **Features** → construção de `features.csv`, opcionalmente reaproveitando `prep.csv`.
3. **MSC & θ(Ea)** → escolha de amostra/métrica, execução e visualização da curva.
4. **Segmentação & Blaine** → estratégia fixa ou data-driven e geração de `segments.json`.
5. **Mecanismo** → relatório por amostra com detecção de mudanças (`τ`, R², AIC/BIC).
6. **ML** → `train-cls` ou `train-reg`, métricas de validação cruzada, artefatos de modelo e predição opcional.
7. **Export** → consolidação em `report.xlsx` e `session.zip` com rastreabilidade de presets.

## Estados de UI
- **Carregando** (`loading`): exibir *spinner* curto com mensagem técnica (ex.: “Executando MSC…”).
- **Sucesso** (`success`): toasts `[ok]` com microcópia positiva (“Dados validados com sucesso”).
- **Alerta** (`warning`): destacar ações pendentes ou artefatos ausentes (“Nenhum CSV carregado”).
- **Falha** (`error`): descrever causa humana + instrução objetiva (“Falha ao gerar features — verifique o preset”).

## Diretrizes de Acessibilidade
- **Navegação por teclado**: todos os botões essenciais possuem hints de foco e ordem previsível via `st.columns`.
- **Contraste mínimo**: textos críticos usam componentes padrão Streamlit (apoiados por o tema Ogum) evitando cores custom de baixo contraste.
- **Foco visível**: mensagens `focus_hint` explicitam atalhos (Tab / Shift+Tab / Enter) para manter contexto.
- **Descrições textuais**: gráficos Plotly recebem captions descritivas via `describe_chart` (pt/en) com resumo numérico.

## Microcópia
- **Tom**: técnico, objetivo e breve (no máximo 70 caracteres para toasts e tooltips).
- **Formato**: prefixos `[ok]`, `[warn]`, `[error]` indicam o estado diretamente.
- **Idiomas**: `pt` como padrão e `en` completo com fallback automático.
- **Localização**: todas as chaves residem em `docs/MICROCOPY_*.yaml` e `app/i18n/locales/*.json`. Toda frase usada na UI deve mapear para uma chave única.

## Estados & Artefatos do Wizard
- Persistência via `session_state` (`wizard_step`, `wizard_flags` e `wizard_context`).
- Pré-condições rígidas: só avança quando o artefato requerido está presente e registrado em `state.register_artifact`.
- Tooltips contextualizam decisões (ex.: opção de reaproveitar CSV cru em *Features*).
- Toasts reutilizam microcópia `[ok]/[warn]/[error]` alinhada ao restante do app.

## Estrutura de Extensão
- **Componentização**: novos passos devem herdar de `WizardStep` (definido em `page_wizard.py`).
- **Hooks de serviço**: nenhuma lógica de negócio duplicada; sempre orquestrar via `app/services/run_cli.py`.
- **Traduções**: toda microcópia adicionada precisa de chaves `pt/en` + atualização do lint (`python -m app.services.i18n_lint`).
- **Testes**: cenários críticos do wizard cobertos por `tests/test_wizard_flow.py` com mocks simples dos serviços.

## Vocabulário Padronizado
- **“Validação”**: refere-se à checagem estrutural do CSV.
- **“Preparação”**: execução do CLI `prep` com geração de `prep.csv`.
- **“Artefato”**: qualquer arquivo registrado em `state.list_artifacts()`.
- **“Sessão”**: contexto corrente do workspace (persistido em `artifacts/ui-session`).

## Glossário de Microcópia
Chaves detalhadas em `docs/MICROCOPY_pt.yaml` / `docs/MICROCOPY_en.yaml` cobrem:
- Upload, seleção e validação de dados.
- Execução de features, MSC, segmentação, mecanismo, ML e exportação.
- Tooltips de ajuda por passo.
- Mensagens de foco e acessibilidade.
