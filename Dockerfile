FROM python:3.11-slim

# 1. Instala Java (essencial para Spark) e Curl
RUN apt-get update && apt-get install -y \
    default-jdk \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Instala o UV para gerenciar o ambiente interno
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 3. Cache de dependências (copia apenas o que define os pacotes)
COPY pyproject.toml uv.lock ./

# Instala tudo no ambiente do container
RUN uv sync --frozen

# 4. CONFIGURAÇÃO DO SEDONA (Onde o erro morre)
# Essas variáveis garantem que qualquer script rodado aqui dentro use Scala 2.12
ENV SEDONA_VERSION=1.5.1
ENV SPARK_VERSION=3.5
ENV SCALA_VERSION=2.12

# O segredo: injetar os JARs corretos em qualquer execução do PySpark
ENV PYSPARK_SUBMIT_ARGS="--repositories https://artifacts.unidata.ucar.edu/repository/unidata-all/ --packages org.apache.sedona:sedona-spark-shaded-${SPARK_VERSION}_${SCALA_VERSION}:${SEDONA_VERSION},org.datasyslab:geotools-wrapper:1.5.1-28.2 --conf spark.sql.extensions=org.apache.sedona.spark.SedonaExtensions pyspark-shell"

# Adiciona o diretório src ao path para seus scripts se encontrarem
ENV PYTHONPATH="/app:/app/src"

# Define o PATH para usar o venv do uv automaticamente
ENV PATH="/app/.venv/bin:$PATH"
