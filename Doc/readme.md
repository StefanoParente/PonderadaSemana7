## 1) Carregamento e Junção dos Dados

O notebook começa carregando os arquivos fornecidos pelo IEEE-CIS e faz a junção deles usando o campo `TransactionID`. Os arquivos incluem:

- `train_transaction.csv`: contém informações sobre as transações.
- `train_identity.csv`: traz dados sobre identidade e dispositivos.
- `sample_submission.csv`: serve como guia para submissão no Kaggle.

Após o merge, o DataFrame final tem 590.540 linhas e 434 colunas, com uma mistura de tipos de dados: 399 colunas numéricas (float64), 31 categóricas (object) e 4 inteiros (int64).

Quando olhamos para a variável `isFraud`, que indica se a transação é fraudulenta, percebemos um grande desbalanceamento:
- 96,5% das transações não são fraudes (0).
- Apenas 3,5% são fraudes (1), o que torna a classe positiva bem rara.

Por causa disso, usar apenas a acurácia como métrica não faz sentido. O notebook prioriza métricas como ROC-AUC, PR-AUC, Recall e F1-score, além de aplicar técnicas para lidar com esse desbalanceamento.

## 2) Análise Exploratória (EDA)

### 2.1 Estatísticas e Distribuições
A análise exploratória revela padrões interessantes nos dados:
- A coluna `TransactionAmt` (valor da transação) segue uma distribuição parecida com log-normal, com a maioria dos valores entre 10 e 500. Fraudes aparecem em várias faixas, mas com leve concentração em valores baixos e médios.
- Na coluna `ProductCD`, o produto "W" é o mais comum, mas "C" e "H" têm uma proporção maior de fraudes, mesmo sendo menos frequentes.
- Em `card4`, cartões Visa e Mastercard dominam, com Visa apresentando uma taxa de fraude relativamente maior.
- Ao analisar o campo temporal `TransactionDT` (convertido para dias em `TransactionDay`), vemos picos de fraude em certas janelas de tempo, sugerindo sazonalidade. Isso indica que criar variáveis temporais, como contagens móveis por cartão ou dispositivo, pode ser útil.
- Sobre dispositivos (`Device`), as taxas de fraude são parecidas entre desktop e mobile, mas o impacto aparece mais em combinações, como "dispositivo raro com valor alto".

### 2.2 Dados Ausentes
Muitas colunas, especialmente as `id_*` e `D*`, têm muitos valores ausentes (algumas com mais de 85% de dados faltando). As piores são:
- `id_24` (99,2%), `id_07` (99,1%), `id_26` (99,1%), `id_21` (99,1%), `id_25` (99,1%).
- `dist2` (93,6%), `D7` (93,4%), `D13` (89,5%), `D14` (89,5%), `D12` (89,0%).

Para lidar com isso, o notebook adota algumas estratégias:
- Remove colunas com valores ausentes extremos (acima de 95%).
- Cria flags binárias ("is_null") para indicar valores ausentes em grupos de variáveis como `id_*` e `D*`.
- Faz imputação: usa a mediana para colunas numéricas e o rótulo "Unknown" para categóricas.

### 2.3 Correlações e Precauções
As correlações lineares entre variáveis são geralmente fracas, devido à heterogeneidade das colunas `V*` e `D*`. Para evitar vazamento de dados (data leakage), o notebook toma cuidado ao calcular variáveis temporais apenas com base em informações anteriores e separa corretamente os conjuntos de treino, validação e teste.

**Principais descobertas da EDA:**
1. O desbalanceamento é um grande desafio, então o foco deve ser maximizar o recall da classe positiva (fraudes) sem aumentar muito os falsos positivos.
2. Variáveis temporais, produtos e cartões ganham mais relevância quando combinadas com informações como valor, IP, e-mail ou dispositivo.
3. Colunas `id_*` são esparsas, mas podem ser úteis se transformadas em flags ou normalizadas.
4. O notebook usa gráficos como histogramas logarítmicos de `TransactionAmt`, contagens por `ProductCD` e `card4`, e curvas de taxa de fraude por dia para visualizar esses padrões.

## 3) Preparação dos Dados

### 3.1 Limpeza e Seleção de Variáveis
O notebook remove colunas com muitos valores ausentes ou pouca variabilidade. Variáveis contínuas, como `TransactionAmt`, `V*` e `D*`, são normalizadas (usando Min-Max ou padronização). Para variáveis categóricas, é feita uma codificação especial (detalhada na seção 4).

Após essas transformações, a matriz de atributos fica com cerca de 428 colunas (contra 434 iniciais), mantendo a capacidade preditiva e reduzindo o ruído.

### 3.2 Organização Temporal
Os dados são ordenados pelo campo `TransactionDT`. O notebook cria sequências temporais baseadas em chaves como cartão ou dispositivo, com padding para lidar com comprimentos diferentes, preparando os dados para o modelo LSTM. Além disso, são criadas variáveis temporais, como dia, hora, lags e contagens móveis.

### 3.3 Divisão dos Dados
Os dados são divididos em três partes, respeitando a ordem temporal e o balanceamento:
- Treino: 70% (~414.542 linhas, 428 colunas).
- Validação: 15% (~90.568 linhas, 428 colunas).
- Teste: 15% (~85.430 linhas, 428 colunas).

### 3.4 Lidando com o Desbalanceamento
Para enfrentar o desbalanceamento, o notebook considera pesos maiores para a classe positiva na função de perda e analisa diferentes limiares de decisão com base na curva Precision-Recall. Técnicas como SMOTE (oversampling sintético) são mencionadas como possibilidade futura, mas evitadas por enquanto para não distorcer a ordem temporal.

## 4) Modelo LSTM (Tabular + Embeddings)

### 4.1 Entradas
O modelo usa:
- Aproximadamente 213 variáveis numéricas (após limpeza e normalização).
- Cinco variáveis categóricas codificadas como embeddings:
  - `ProductCD` (6 categorias).
  - `card4` (6 categorias).
  - `DeviceType` (4 categorias).
  - `P_emaildomain` (61 categorias, com normalização de domínios).
  - `R_emaildomain` (62 categorias, com normalização semelhante).

### 4.2 Estrutura do Modelo
O modelo combina:
- Embeddings para as variáveis categóricas, com tamanhos proporcionais à raiz quadrada da cardinalidade.
- Uma ou duas camadas LSTM para capturar padrões temporais, com dropout para evitar overfitting.
- Uma camada densa final com ativação sigmóide para prever a probabilidade de fraude.
- A função de perda é a Binary Cross-Entropy, ajustada com pesos para a classe positiva.
- O otimizador usado é o Adam, com Early Stopping monitorando o ROC-AUC de validação.

O notebook mostra um resumo do modelo (`LSTMTabular`), detalhando as camadas de embeddings e lineares.

## 5) Treinamento e Resultados

### 5.1 Processo de Treinamento
O treinamento usa mini-batches com padding de sequências no DataLoader. O Early Stopping é ativado para evitar overfitting, e os modelos são salvos na pasta `ieee_lstm_model/`. Durante o treinamento, são registrados a perda, o ROC-AUC e o PR-AUC (AP) na validação.

### 5.2 Resultados na Validação
As métricas obtidas foram:
- ROC-AUC: ~0,818.
- PR-AUC (AP): ~0,292.
- Acurácia: ~0,564 (menos relevante por causa do desbalanceamento).
- Precisão: ~0,066.
- Recall: ~0,703.
- F1-score: ~0,122.

O recall alto com precisão baixa reflete a priorização de detectar fraudes, mesmo que isso gere mais falsos positivos. Ajustar o limiar de decisão pode equilibrar precisão e recall, dependendo das prioridades do negócio.

### 5.3 Curvas de Aprendizado
A perda no treino diminui consistentemente, e o ROC-AUC na validação estabiliza em ~0,82, sem sinais claros de overfitting. O notebook sugere verificar os gráficos gerados para confirmar a ausência de gaps significativos entre treino e validação.

## 6) Discussão e Possíveis Melhorias

**O que deu certo:**
- A combinação de variáveis temporais e categóricas (com embeddings) levou a um ROC-AUC sólido (~0,82).
- O uso de pesos na função de perda e a avaliação via PR-AUC ajudaram a ajustar o modelo para o desbalanceamento.

**Limitações:**
- A precisão (~6,6%) ainda é baixa no limiar padrão, o que é comum em problemas de fraude.
- Colunas como `id_*` e `D*` com muitos valores ausentes exigem mais trabalho para extrair informações úteis.

**Sugestões para o futuro:**
1. Criar variáveis temporais mais sofisticadas por entidade (como cartão, e-mail ou IP), incluindo médias móveis, contagens ou entropia por janela.
2. Ajustar o limiar de decisão com base no custo de falsos negativos versus falsos positivos.
3. Aplicar técnicas de calibração (como Platt ou Isotônica) para melhorar as probabilidades previstas.
4. Experimentar ensembles, combinando o LSTM com modelos como XGBoost ou LightGBM.
5. Testar funções de perda como Focal Loss, mais adequadas para desbalanceamento extremo.
6. Melhorar a qualidade dos dados, com padronização de e-mails, parsing robusto de dispositivos ou uso de informações geográficas de IPs (se permitido).

## 7) Como Reproduzir o Experimento

### 7.1 Ambiente
O notebook foi desenvolvido em Python 3.11 ou superior, usando bibliotecas como `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `torch`, `torchvision` e `torchaudio`. Ele inclui instruções para instalar o PyTorch (versão CPU) se necessário.
