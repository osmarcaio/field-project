# TTI-HydroMet — previsão de nível da estação 413

Projeto de previsão hidrológica de curto prazo na bacia do rio Tamanduateí usando o dataset **TTI-HydroMet / iFAST**. O alvo atual é exclusivamente o **nível (`FLUm`) da estação 413**.

## Objetivo atual

A cada 10 minutos, estimar a variação do nível nas próximas 2 h:

\[
\Delta H_{2h}(t)=H(t+2h)-H(t)
\]

Além da previsão em metros, o projeto tenta reconhecer episódios de subida forte (`>=1 m`, `>=2 m`, `>=3 m`). Esses valores são **faixas de subida para pesquisa** e **não são cotas oficiais de atenção/alerta/emergência**.

## Dados usados

- nível atual e histórico da estação **413**;
- chuva de **23 pluviômetros** distribuídos na bacia;
- níveis a montante das estações **1000490, 143 e 283**;
- radar meteorológico com **434 células de aproximadamente 1 km**, atualmente resumidas por média, máximo, percentis, fração molhada e janelas temporais;
- resolução temporal principal: **10 min**.

A base bruta não deve ser versionada no GitHub. Baixe o TTI-HydroMet e coloque-o em `data/raw/`.

## Arquitetura candidata atual

A previsão de magnitude é **híbrida**:

1. **Regime normal:** `XGBoost + radar`.
2. A TCN calcula `P(DeltaH >= 1 m)`.
3. Se esse score ultrapassa **0,8061**, o sistema entra internamente em regime de subida forte.
4. Nesse regime, a magnitude passa a ser a média das previsões **TCN + GRU**.

Em 2024, o roteamento neural ocorreu em apenas **~2,46%** dos timestamps.

| Modelo de magnitude | MAE 2024 | RMSE 2024 |
|---|---:|---:|
| XGBoost + radar | 6,09 cm | 20,30 cm |
| TCN | 7,39 cm | 20,96 cm |
| GRU | 7,48 cm | 20,90 cm |
| **Híbrido por regime** | **6,07 cm** | **19,70 cm** |

O XGBoost continua sendo melhor para o comportamento cotidiano; as redes são usadas principalmente porque ajudam nos extremos.

## Sinais de subida forte

A TCN possui três heads de classificação. Os thresholds probabilísticos foram definidos em **2023**, sem usar 2024 para calibrá-los.

| Faixa real em 2 h | Threshold TCN | Recall por timestamp (2024) | Precision por timestamp (2024) | Recall por episódio* |
|---|---:|---:|---:|---:|
| >= 1 m | 0,8061 | 77,6% | 43,8% | 96,2% |
| >= 2 m | 0,8211 | 68,1% | 44,7% | 91,4% |
| >= 3 m | 0,8787 | 52,2% | 28,6% | 83,3% |

\* Nova avaliação v7: episódios são separados quando há mais de 60 min sem uma janela real `DeltaH >= 1 m`. Em 2024 isso produz 53 episódios `>=1 m`, 35 `>=2 m` e 18 `>=3 m`.

A precision baixa nos casos severos é uma limitação real: o sistema ainda produz muitos falsos sinais. Por isso o projeto não deve ser descrito como um sistema autônomo de alerta público.

### Persistência do sinal

A v7 também testa exigir 1, 2 ou 3 ativações consecutivas. A configuração candidata exige **2 timestamps consecutivos (20 min)** para reduzir sinais isolados. Essa escolha foi feita em 2024, que é o conjunto de desenvolvimento final; 2025 permanece reservado para teste cego.

## Avaliação por episódio

Prever de 10 em 10 minutos continua correto operacionalmente. O que mudou é a avaliação: uma tempestade não deve valer como dezenas de enchentes independentes.

O notebook `07c` mede, além das métricas por timestamp:

- recall e precision por episódio;
- primeiro sinal correto do episódio;
- atraso desde o início retrospectivo da faixa severa;
- antecedência até o **pico de referência**;
- recall com pelo menos 30, 60, 90 e 120 min de antecedência;
- falsos blocos de sinal.

Sem cotas oficiais da 413, o “pico de referência” é o fim da janela de 2 h que contém a maior subida do episódio. Ele é uma proxy de oportunidade operacional, não uma cota crítica.

## Notebooks

Ordem principal:

```text
00  setup e inventário
01  geografia e sensores
02  auditoria temporal e base mestra
03  EDA, features e splits
04  baselines e XGBoost
05  extremos e ablation
06  radar agregado
06b radar espacial com as 434 células via PCA       [opcional]
07  modelos avançados e incerteza
07b TCN e GRU temporais
07c avaliação operacional por episódio              [v7]
07d relatórios integrados                            [v7]
08  arquitetura híbrida + teste cego de 2025        [NÃO rodar antes de congelar]
09  previsão operacional / hindcast da 413
```

## Radar espacial

A arquitetura principal ainda reduz as 434 células do radar a estatísticas agregadas. O `06b_radar_espacial_pca_434.ipynb` é um experimento opcional que usa todas as células e aprende componentes espaciais com `IncrementalPCA`, ajustado somente no treino. Ele **não altera automaticamente** a arquitetura final: só deve ser incorporado se melhorar 2024, especialmente os extremos.

## Divisão temporal

- treino: 2015–2022;
- validação: 2023;
- desenvolvimento final: 2024;
- **teste cego:** 2025-01-01 a 2025-03-29.

Não usar 2025 para escolher modelo, threshold ou hiperparâmetro antes de executar o notebook 08.

## Estrutura GitHub

Versionar principalmente `notebooks/`, `utils/`, `README.md`, requirements e resultados finais pequenos. Não versionar `.venv/`, base bruta, parquets processados, checkpoints grandes ou `outputs/` completos.

## Limitações atuais

- somente a estação 413 é alvo;
- os limiares 1/2/3 m não representam risco oficial;
- magnitude das maiores subidas ainda é sistematicamente subestimada;
- precision dos sinais `>=3 m` é baixa;
- radar espacial completo ainda é experimental;
- intervalos de incerteza globais não são confiáveis nos extremos.

## Próximo marco

Revisar o `07c`, congelar `models/architecture_candidate_v7.json` e só então executar o `08` uma única vez para avaliar 2025.
