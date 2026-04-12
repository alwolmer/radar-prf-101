#!/bin/bash
set -eu

ENV_FILE="${1:-.env.docker}"
TARGET_DIR="${2:-/opt/spark-jars}"
WORK_DIR="$(mktemp -d)"
POM_FILE="$WORK_DIR/pom.xml"

case "$ENV_FILE" in
    */*) ;;
    *) ENV_FILE="./$ENV_FILE" ;;
esac

cleanup() {
    rm -rf "$WORK_DIR"
}

trap cleanup EXIT

set -a
. "$ENV_FILE"
set +a

: "${SPARK_VERSION:=3.5}"
: "${SCALA_VERSION:=2.12}"
: "${SEDONA_VERSION:=1.5.1}"
: "${GEOTOOLS_WRAPPER_VERSION:=1.5.1-28.2}"
: "${HADOOP_AWS_VERSION:=3.4.2}"
: "${AWS_JAVA_SDK_BUNDLE_VERSION:=1.12.367}"
: "${SPARK_MAVEN_REPOSITORIES:=https://artifacts.unidata.ucar.edu/repository/unidata-all/}"

REPOSITORIES_XML=""
INDEX=1
OLD_IFS="${IFS}"
IFS=','
for REPO in ${SPARK_MAVEN_REPOSITORIES}; do
    TRIMMED_REPO="$(printf '%s' "$REPO" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    if [ -z "$TRIMMED_REPO" ]; then
        continue
    fi
    REPOSITORIES_XML="${REPOSITORIES_XML}
    <repository>
      <id>repo-${INDEX}</id>
      <url>${TRIMMED_REPO}</url>
    </repository>"
    INDEX=$((INDEX + 1))
done
IFS="${OLD_IFS}"

cat > "$POM_FILE" <<EOF
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>
  <groupId>local.radar_prf_101</groupId>
  <artifactId>spark-jars</artifactId>
  <version>1.0.0</version>
  <repositories>${REPOSITORIES_XML}
  </repositories>
  <dependencies>
    <dependency>
      <groupId>org.apache.hadoop</groupId>
      <artifactId>hadoop-aws</artifactId>
      <version>${HADOOP_AWS_VERSION}</version>
    </dependency>
    <dependency>
      <groupId>com.amazonaws</groupId>
      <artifactId>aws-java-sdk-bundle</artifactId>
      <version>${AWS_JAVA_SDK_BUNDLE_VERSION}</version>
    </dependency>
    <dependency>
      <groupId>org.apache.sedona</groupId>
      <artifactId>sedona-spark-shaded-${SPARK_VERSION}_${SCALA_VERSION}</artifactId>
      <version>${SEDONA_VERSION}</version>
    </dependency>
    <dependency>
      <groupId>org.datasyslab</groupId>
      <artifactId>geotools-wrapper</artifactId>
      <version>${GEOTOOLS_WRAPPER_VERSION}</version>
    </dependency>
  </dependencies>
</project>
EOF

mkdir -p "$TARGET_DIR"
mvn -f "$POM_FILE" -B -q dependency:copy-dependencies \
    -DincludeScope=runtime \
    -DoutputDirectory="$TARGET_DIR"
