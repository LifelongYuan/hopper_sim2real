#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
JAVA_DIR="${DIR}/../lcm_types/java"

if [ -d "${JAVA_DIR}" ] && [ -f "${JAVA_DIR}/my_types.jar" ] && [ -f "${JAVA_DIR}/lcm.jar" ]; then
    export CLASSPATH="${JAVA_DIR}/my_types.jar:${JAVA_DIR}/lcm.jar"
    cd "${JAVA_DIR}"
fi

exec lcm-spy
