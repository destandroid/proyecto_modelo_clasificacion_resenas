#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

LOGFILE="execution.log"
NOTEBOOK="scrapp.ipynb"
OUTPUT="output.ipynb"
APP_PATH="datalake/3_CONSUMPTION_ZONE/app.py"

start_time=$(date '+%d/%m/%Y %H:%M:%S')
echo "🔁 Inicio de ejecución: $start_time" | tee "$LOGFILE"
echo "📘 Ejecutando cuaderno: $NOTEBOOK" | tee -a "$LOGFILE"

jupyter nbconvert --to notebook --execute "$NOTEBOOK" --output "$OUTPUT" >> "$LOGFILE" 2>&1
status=$?

end_time=$(date '+%d/%m/%Y %H:%M:%S')
echo "🕓 Fin de ejecución del notebook: $end_time" | tee -a "$LOGFILE"

if [ $status -eq 0 ]; then
    echo "✅ Cuaderno ejecutado sin errores." | tee -a "$LOGFILE"
    echo "🚀 Iniciando dashboard Streamlit..." | tee -a "$LOGFILE"

    # Ejecutar Streamlit en segundo plano
    streamlit run "$APP_PATH" >> "$LOGFILE" 2>&1 &
    STREAMLIT_PID=$!

    # Obtener IP local automáticamente
    LOCAL_IP=$(hostname -I | awk '{print $1}')

    echo ""
    echo "🌐 Puedes ver el dashboard en tu navegador en:"
    echo "  👉 Local URL: http://localhost:8501"
    echo "  👉 Network URL: http://$LOCAL_IP:8501"
    echo ""
    echo "🖥️ El dashboard está en ejecución. Escribe 'salir' para detenerlo."

    while true; do
        read -p ">> " input
        if [ "$input" == "salir" ]; then
            echo "⏹️ Cerrando Streamlit (PID $STREAMLIT_PID)..." | tee -a "$LOGFILE"
            kill $STREAMLIT_PID
            break
        else
            echo "❓ Escribe 'salir' para detener el dashboard."
        fi
    done
else
    echo "❌ Error durante la ejecución del cuaderno. El dashboard no será iniciado." | tee -a "$LOGFILE"
fi

echo "📄 Log completo en: $LOGFILE"
