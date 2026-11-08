# Machinelearning_iudigitall
Repo aprendizaje machine learning

python -m pip install --upgrade huggingface_hub
# S30 - EA3 Unidad 2: Aprendizaje Supervisado · Predicción de Deserción (Churn)

## Integrantes
- (Escribe aquí tu nombre y el de tus compañeros de grupo)

---

## 0. Pregunta de Negocio y Dataset de Hugging Face
- *Pregunta de negocio:*  
  ¿Qué clientes tienen mayor probabilidad de abandonar el servicio de telecomunicaciones (Churn), para que el equipo de fidelización y retención pueda priorizar intervenciones comerciales preventivas antes de que cancelen su suscripción?
- *¿Quién se beneficia?:* El equipo de Customer Success, Fidelización y Operaciones Comerciales.
- *¿Qué decisión apoya?:* Permite focalizar descuentos, planes de lealtad y soporte técnico prioritario únicamente en los clientes con alto riesgo de fuga, optimizando el presupuesto de retención.
- *Dataset de Hugging Face:* [scikit-learn/churn-prediction](https://huggingface.co/datasets/scikit-learn/churn-prediction)
- *Tamaño:* 7.043 filas y 21 columnas.
- *Variable objetivo ($y$):* Churn (1 = abandona el servicio, 0 = se mantiene activo).

---

## 1. Preprocesamiento de Datos
- *Calidad e Imputación:* La columna TotalCharges presentaba registros en blanco correspondientes a clientes recién ingresados (tenure = 0), los cuales fueron imputados con 0.0.
- *Limpieza:* Se excluyó el identificador único customerID para evitar sobreajuste y fuga de datos.
- *Codificación:* Se aplicó One-Hot Encoding a las variables categóricas (Contract, InternetService, PaymentMethod, etc.) con representación numérica (dtype=float).
- *División Estratificada:* Partición 80% entrenamiento (5.634 clientes) y 20% prueba (1.409 clientes), preservando la tasa base de deserción del 26.5% en ambos conjuntos.

---

## 2. Modelo Elegido y Funcionamiento
- *Algoritmo:* *Árbol de Decisión (DecisionTreeClassifier)* (Aprendizaje Supervisado).
- *Justificación:* Los árboles de decisión permiten una interpretabilidad total mediante reglas de negocio claras ("IF-THEN"), facilitando que los directores comerciales comprendan exactamente qué combinación de servicios y contratos detona el abandono.
- *Cómo funciona paso a paso:*
  1. *Raíz y divisiones:* El algoritmo evalúa cada variable candidata buscando el umbral de corte que maximiza la reducción de impureza de Gini ($G = 1 - \sum p_k^2$).
  2. *Partición recursiva:* Separa el conjunto de clientes en nodos hijos cada vez más homogéneos.
  3. *Control de profundidad:* Con validación cruzada estratificada (5 pliegues) se identificó que una profundidad máxima de max_depth = 3 previene el sobreajuste y mantiene una alta generalización.

---

## 3. Métricas y Evaluación del Modelo
- *Exactitud (Accuracy):* ~80% (superior al 73.5% del modelo perezoso de responder siempre "se queda").
- *ROC-AUC:* ~0.82 (muy superior al azar de 0.50).
- *PR-AUC:* ~0.60 frente a una tasa base de deserción del 26.5%.
- *Matriz de Confusión para el Negocio:*
  - *Falsos Negativos (Costo Crítico):* Fugas no detectadas que implican pérdida directa de ingresos recurrentes.
  - *Falsos Positivos (Costo Menor):* Clientes leales a los que se les ofrece una promoción preventiva por precaución.

---

## 4. Documento de Hallazgos y Respuesta al Negocio
1. *Regla Principal de Abandono:* El tipo de contrato (Contract_Month-to-month) junto con la antigüedad (tenure ≤ 12 meses) son los dos factores más determinantes. Los clientes nuevos con contrato mensual presentan una tasa de abandono superior al 45%.
2. *Sensibilidad en Fibra Óptica:* Clientes con servicio de Internet por fibra óptica que no cuentan con soporte técnico o seguridad en línea tienen mayor propensión a la fuga debido a cobros mensuales altos sin valor agregado percibido.
3. *Recomendaciones Accionables:*
   - Incentivar la migración de contratos mensuales a contratos anuales mediante un descuento en los primeros 3 meses.
   - Ofrecer paquetes de soporte técnico bonificado a clientes con fibra óptica durante su primer año.
4. *Equidad:* La evaluación de tasas entre géneros (Male vs. Female) demostró paridad en el recall (~0.52 en ambos), garantizando que las decisiones comerciales no presentan sesgo demográfico.

---

## 5. Instrucciones de Ejecución
bash
# 1. Clonar el repositorio
git clone <URL_DE_TU_REPOSITORIO>
cd <NOMBRE_DEL_REPO>

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar el script o notebook
python script.py