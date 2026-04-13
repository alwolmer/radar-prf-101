# Fontes de Dados

Este documento lista as fontes externas usadas no projeto, o papel de cada uma no pipeline da BR-101 e como elas entram nas camadas bronze e silver atualmente implementadas.

## Visão geral

| Fonte | Papel no projeto | Bronze atual | Silver atual |
| --- | --- | --- | --- |
| PRF | fato principal de acidentes | `data/bronze/prf_accidents` | `data/silver/prf_accidents_standardized` |
| DNIT | geometria da BR-101 | `data/bronze/dnit_road_network` | `data/silver/dnit_br101_corridor` |
| IBGE municípios | base territorial municipal | `data/bronze/ibge/municipios` | `data/silver/ibge_territorial_preprocessed/municipalities_preprocessed` |
| IBGE RGI | base territorial regional | `data/bronze/ibge/rgi` | `data/silver/ibge_territorial_preprocessed/rgis_preprocessed` |

## 1. PRF

### Fonte oficial

- Portal de dados abertos da PRF:
  `https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf`

### O que fornece

- tabelas anuais de acidentes por ocorrência
- colunas operacionais e descritivas usadas para construir o histórico de acidentes da BR-101

### Uso no projeto

A PRF é a fonte principal de fatos. Hoje ela é usada para:

- baixar os arquivos anuais de ocorrências
- consolidar as ocorrências em uma bronze particionada

### Jobs implementados

- Bronze: [src/etl/bronze/prf_source2bronze.py](../src/etl/bronze/prf_source2bronze.py:1)
- Silver: [src/etl/silver/prf_bronze2silver.py](../src/etl/silver/prf_bronze2silver.py:1)

### Saídas materializadas

- Bronze: `data/bronze/prf_accidents`
- Silver: `data/silver/prf_accidents_standardized`

## 2. DNIT

### Fonte oficial

- Repositório DNIT das bases geométricas do SNV:
  `https://servicos.dnit.gov.br/dnitcloud/index.php/s/oTpPRmYs5AAdiNr`

### O que fornece

- geometrias lineares da malha rodoviária federal
- snapshots históricos usados para derivar o traçado da BR-101

### Snapshots usados hoje

- `201703A.zip`
- `202107A.zip`
- `202601A.zip`

### Uso no projeto

O DNIT é a fonte geométrica da rodovia. Hoje ele é usado para:

- isolar os segmentos da BR-101 na bronze
- dissolver as geometrias por snapshot
- gerar centerlines e buffers por snapshot
- construir a união das centerlines e a união do corredor da BR-101

### Jobs implementados

- Bronze: [src/etl/bronze/dnit_source2bronze.py](../src/etl/bronze/dnit_source2bronze.py:1)
- Silver: [src/etl/silver/dnit_bronze2silver.py](../src/etl/silver/dnit_bronze2silver.py:1)

### Saídas materializadas

- Bronze: `data/bronze/dnit_road_network`
- Silver: `data/silver/dnit_br101_corridor`

## 3. IBGE Municípios

### Fonte oficial

- Malha municipal 2024:
  `https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2024/Brasil/BR_Municipios_2024.zip`

### O que fornece

- polígonos oficiais de municípios
- hierarquia municipal para RGI
- atributos administrativos e regionais usados em enriquecimento futuro

### Uso no projeto

Hoje os municípios do IBGE são usados para:

- materializar a malha territorial oficial na bronze
- padronizar códigos, nomes, CRS, tipo geométrico e área poligonal na silver

Observação importante:

- a silver atual do IBGE para antes de qualquer interseção com DNIT ou PRF
- portanto, ainda não há recorte oficial “municípios em escopo da BR-101” persistido como parte do pipeline implementado

### Jobs implementados

- Bronze: [src/etl/bronze/ibge_municipios_src2bronze.py](../src/etl/bronze/ibge_municipios_src2bronze.py:1)
- Silver: [src/etl/silver/ibge_bronze2silver.py](../src/etl/silver/ibge_bronze2silver.py:1)

### Saídas materializadas

- Bronze: `data/bronze/ibge/municipios`
- Silver: `data/silver/ibge_territorial_preprocessed/municipalities_preprocessed`

## 4. IBGE RGI

### Fonte oficial

- Regiões Geográficas Imediatas 2024:
  `https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2024/Brasil/BR_RG_Imediatas_2024.zip`

### O que fornece

- polígonos oficiais de RGI
- atributos de hierarquia regional do IBGE

### Uso no projeto

Hoje as RGIs do IBGE são usadas para:

- materializar a malha regional oficial na bronze
- padronizar códigos, nomes, CRS, tipo geométrico e área poligonal na silver

Assim como no caso dos municípios, a interseção com o corredor DNIT ainda não faz parte da silver implementada.

### Jobs implementados

- Bronze: [src/etl/bronze/ibge_rgi_src2bronze.py](../src/etl/bronze/ibge_rgi_src2bronze.py:1)
- Silver: [src/etl/silver/ibge_bronze2silver.py](../src/etl/silver/ibge_bronze2silver.py:1)

### Saídas materializadas

- Bronze: `data/bronze/ibge/rgi`
- Silver: `data/silver/ibge_territorial_preprocessed/rgis_preprocessed`

## 5. Fronteira da camada silver atual

As transformações atualmente implementadas na silver ainda são monofonte:

- PRF transforma apenas dados da PRF
- DNIT transforma apenas dados do DNIT
- IBGE transforma apenas dados do IBGE

Ficam para a camada gold:

- classificação PRF x corredor DNIT
- recorte territorial IBGE x geometria DNIT
- atribuição de acidentes PRF a município e RGI

## 6. Operação local

As fontes são processadas pelos jobs versionados em [dvc.yaml](../dvc.yaml:1) e executados via [Makefile](../Makefile:1).

Comandos principais:

```bash
make bronze
make silver
make dvc-repro-bronze
make dvc-repro-silver
```

## 7. Observações

- Os datasets materializados são versionados com DVC.
- Os notebooks, especialmente `notebooks/eda-p1.ipynb`, continuam sendo a referência exploratória para as próximas etapas de integração.
- Fontes adicionais como clima, calendário e feriados ainda não fazem parte do pipeline implementado neste repositório.
