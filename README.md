# Aplicação de Técnicas Avançadas de Machine Learning para Detecção de Anomalias em Operações de Compras em uma Empresa de Gases Industriais

#### Aluno: [Lucas Valiate](https://github.com/lucasvaliate)
#### Aluno: [Paula Felippe](https://github.com/paulacfelippe)
#### Aluno: [Thiago Lima](https://github.com/thgblima)
#### Orientadora: [Manoela Kohler](https://github.com/manoelakohler).


---

Trabalho apresentado ao curso [BI MASTER](https://ica.puc-rio.ai/bi-master) como pré-requisito para conclusão de curso e obtenção de crédito na disciplina "Projetos de Sistemas Inteligentes de Apoio à Decisão".

<!-- para os links a seguir, caso os arquivos estejam no mesmo repositório que este README, não há necessidade de incluir o link completo: basta incluir o nome do arquivo, com extensão, que o GitHub completa o link corretamente -->
- [Link para o código](https://github.com/lucasvaliate/trabalhopuc/blob/main/Modelagem_de_dados_com_RFECV_SVM_RF.ipynb). <!-- caso não aplicável, remover esta linha -->


### Resumo

Através da aplicação de modelos de machine learning de aprendizagem supervisionada, este trabalho analisou uma base de compras de uma empresa multinacional buscando anomalias nas transações realizadas, a partir de transações previamente categorizadas pela Companhia.

O principal objetivo era identificar a estratégia de modelagem que proporcionasse o melhor resultado com a máxima performance possível. Para isto, realizaram-se testes comparativos de metodologias de balanceamento e redução de dimensionalidade, bem como de modelagem.


### 1. Introdução
Este trabalho tem como foco a aplicação prática de técnicas avançadas de Machine Learning para identificar anomalias em compras de materiais em uma empresa do segmento de gases industriais. A empresa possui um extenso conjunto de dados que documenta suas operações de compra. Entretanto, este conjunto de dados, rico e complexo, pode ser difícil de interpretar e analisar manualmente.

Para facilitar a análise, utilizamos um script em Python que, por meio de uma série de lógicas, pré-processou os dados e identificou cinco possíveis categorias de anomalias. Estas foram codificadas como labels, sendo:

Anomalia 0: Sem anomalia
Anomalia 1: Itens que possuem o mesmo item, filial, mês de compra, prazo de pagamento e fornecedores diferentes com preço maior que o mínimo;
Anomalia 2: Itens que possuem o mesmo item, fornecedor, filial, mês de compra, prazo de pagamento e preços diferentes;
Anomalia 3: Itens que possuem o mesmo item, fornecedor, filial, mês de compra, prazo de pagamento e preços muito diferentes - Preço considerado outlier estatístico – maior do que média + 2 desvio padrão;
Anomalia 4: Bad Data - Itens que possuem preços muito diferentes do normal no Brasil. Preço considerado outlier estatístico – maior do que média + 3 desvio padrão a nível Brasil.

A presença desses labels oferece uma oportunidade única de aplicar técnicas de aprendizagem supervisionada. Essa abordagem de Machine Learning permite que um modelo seja treinado para aprender os padrões existentes nos dados e, assim, prever as labels de anomalias com base em novas entradas de dados.

Neste estudo, aplicamos o conhecimento adquirido no curso de MBA para desenvolver um modelo de aprendizagem supervisionada robusto que possa ser treinado e testado com este conjunto de dados. O objetivo final é criar uma ferramenta que possa ser usada para identificar de forma eficiente e precisa possíveis anomalias em futuras operações de compra, contribuindo para uma maior eficácia operacional e controle de custos na empresa.


### 2. Modelagem

Na realização de nosso projeto de conclusão de MBA em Machine Learning, empregamos uma série de técnicas e procedimentos para otimizar a análise dos nossos dados. A seguir, resumimos as principais etapas do processo:

Análise Exploratória dos Dados: Inicialmente, conduzimos uma análise exploratória detalhada para avaliar a qualidade do nosso conjunto de dados. Ao examinar a quantidade de dados, o tipo de dados imputados e a variabilidade desses dados, conseguimos identificar e eliminar colunas que eram menos relevantes ou que apresentavam pouca variação.

Ajuste dos Tipos de Dados: Identificamos a presença de várias variáveis categóricas que precisavam ser convertidas em formato numérico para análise subsequente. Devido ao tamanho substancial do nosso conjunto de dados, optamos pelo método LabelEncoder em vez do método One-Hot Encoding, uma vez que este último poderia ter aumentado excessivamente a dimensionalidade dos dados. O LabelEncoder converte cada categoria em um número inteiro, permitindo uma representação numérica eficiente das variáveis categóricas.

Tratamento de Valores Ausentes: Utilizando a matriz MSNO, conseguimos identificar e visualizar a presença de valores ausentes em nosso conjunto de dados. Notamos que a coluna 'custo medio brasil' possuía uma quantidade significativa de valores faltantes. Considerando a importância dessa coluna para nossa análise, optamos por imputar os valores ausentes com o valor mais frequente da coluna, também conhecido como moda.

Modelagem: Durante o nosso estudo de pós-graduação, nos deparamos com dois desafios comuns em machine learning, a alta dimensionalidade da nossa base de dados e o desbalanceamento de classes. Para abordar estes problemas, empregamos várias estratégias e testamos diferentes combinações de técnicas de balanceamento de classes e redução de dimensionalidade em dois algoritmos de aprendizado de máquina, Máquina de Vetores de Suporte (SVM) e Random Forest.

Em um primeiro passo, estabelecemos uma linha de base, treinando ambos os modelos, SVM e Random Forest, sem qualquer ajuste para o desbalanceamento de classes.

Posteriormente, utilizamos as seguintes técnicas de balanceamento: 

i. Balanceamento com Oversampling: Diante do desbalanceamento de classes evidente em nosso conjunto de dados, introduzimos a técnica de oversampling. Esta técnica foi escolhida para aumentar a representação das classes minoritárias, criando cópias sintéticas dessas observações. 

ii. Balanceamento com Classweight: Diferente do oversampling, o classweight atribui pesos às classes, de acordo com sua frequência no conjunto de treino. Assim, classes minoritárias recebem maior peso, o que permite que o modelo as considere mais relevantes durante o treinamento. 

No que concerne às técnicas de redimensionamento, foram aplicados o PCA (Principal Component Analysis) e a RFE (eliminação recursiva de recursos), que são duas técnicas diferentes de redução de dimensionalidade que podem ser usadas para lidar com conjuntos de dados grandes e complexos, como a base de dados do problema em questão:
	i. PCA: Realiza a decomposição de matriz, buscando projetar os dados em um espaço de menor dimensão, mantendo o máximo de variação possível, encontrando os componentes principais do conjunto de dados que sejam mais significativos e capturem a maior parte da variação total dos dados. O resultado final do PCA é uma projeção dos dados em um novo espaço de menor dimensão.
	ii. RFECV: método iterativo de seleção de recursos, buscando a cada iteração encontrar um subconjunto de recursos que melhoram a precisão do modelo. Neste sentido,começa-se com um conjunto completo de recursos e a cada nova rodada remove os recursos menos importantes, treinando o modelo novamente a cada iteração até que um número desejado de recursos seja selecionado.


Finalmente, os algoritmos de aprendizado de máquina utilizados foram o Support Vector Machine (SVM) e o Random Forest (RF):

i) Support Vector Machine (SVM): algoritmo de aprendizado de máquina amplamente utilizado para problemas de classificação, onde o objetivo é atribuir rótulos a instâncias de dados com base em suas características.
Para o nosso problema, a capacidade de lidar com conjuntos de dados de alta dimensionalidade era um fator importante. E esse algoritmo, mesmo em espaços de atributos com muitas variáveis, poderia encontrar um hiperplano de separação eficaz. 
Além dessa necessidade, um outro ponto importante para o nosso problema é a resistência ao overfitting, uma vez que ele busca encontrar um hiperplano de separação que minimize o erro de classificação e maximize a margem, resultando em um modelo mais generalizável. 
No entanto, é importante ressaltar que o desempenho do SVM pode ser afetado quando há desbalanceamento de classes nos dados. Problema que existia no nosso contexto, mas que havia sido tratado com as técnicas mencionadas acima.


ii) Random Forest (RF): conglomerado de árvores de decisão individuais, onde cada árvore é treinada com uma amostra aleatória dos dados de treinamento e a previsão final é determinada pela combinação das previsões de todas as árvores.
	Um benefício que nos levou a utilização desse algoritmo é sua capacidade de fornecer estimativas de importância das características. Isso permite identificar quais características têm maior influência na classificação, fornecendo insights valiosos para entender os aspectos mais relevantes do problema em estudo. Além disso, o algoritmo lida bem com problemas de desequilíbrio de classes, atribuindo peso e evitando vieses na classificação.
	Porém, o custo computacional era intenso, especialmente em nosso caso que tínhamos um conjunto de dados grande. E, diferente de uma árvore de decisão única, em que seria possível seguir o fluxo de decisão e entender  como cada característica contribui para a classificação. A interpretação do RF se torna mais complexa.



### 3. Discussão de Resultados


![image](https://github.com/lucasvaliate/trabalhopuc/assets/132826869/083a6a40-f995-44a4-a9d2-344f38e4b7a8)





RandomForest versus SVM: Em geral, o RandomForestClassifier superou o SVM em todas as métricas. Para destacar alguns números, o RandomForestClassifier com RFECV e Oversampling teve um recall de 83,66%, acurácia de 97,06% e F1 de 97,06%, enquanto o melhor SVM (RFECV com Classweight) teve um recall de 43,14%, acurácia de 87,32% e F1 de 43,77%. Isso representa uma melhoria de mais de 40 pontos percentuais em todas as métricas quando comparamos o melhor RandomForestClassifier com o melhor SVM.

PCA versus RFECV: O RFECV continuou a mostrar um desempenho superior em comparação ao PCA para redução de dimensionalidade. O RandomForestClassifier com RFECV e Classweight obteve um recall de 81,79%, acurácia de 97,11% e F1 de 97,11%, enquanto o melhor desempenho usando PCA (RandomForestClassifier com Classweight) alcançou um recall de 47,04%, acurácia de 90,74% e F1 de 53,91%. Isso sugere uma melhoria de mais de 30 pontos percentuais na maioria das métricas quando usamos RFECV em vez de PCA.

Oversampling versus ClassWeight: O RandomForestClassifier com RFECV e Oversampling mostrou um desempenho ligeiramente melhor do que o RandomForestClassifier com RFECV e ClassWeight, principalmente em termos de Recall (83,66% vs 81,79%) e F1 (97,06% vs 97,11%). A diferença na acurácia foi quase imperceptível (97,06% vs 97,11%). Isso sugere que o Oversampling pode ter ajudado a melhorar o desempenho do modelo para a classe minoritária. No entanto, o uso de ClassWeight também resultou em um bom desempenho e pode ser preferível em situações onde o Oversampling pode levar a overfitting.

Destacamos também o caso do RandomForestClassifier com PCA e Oversampling, que apresentou uma acurácia e F1 de 99,28%. No entanto, após análises adicionais, identificamos que este caso estava com overfitting, o que significa que embora tivesse um desempenho excepcional nos dados de treinamento, não generalizou bem para dados novos ou de teste.

### 4. Conclusões

Conforme abordado pela discussão de resultados, o modelo que obteve a melhor performance foi modelo que utilizou o modelo de machine learning Random Forest, associado às técnicas de RFECV para redução de dimensionalidade e Classweight para balanceamento do conjunto de dados. 

Por fim, como sugestão para estudos futuros, sugerimos uma busca de hiperparâmetros para otimizar o desempenho do modelo. Técnicas como GridSearchCV ou RandomizedSearchCV da biblioteca scikit-learn permitiriam. Além disso, há a possibilidade de utilização de modelos não supervisionados como autoencoders e oneclass que mitigariam os problemas relacionados ao balanceamento da base. 

---

Matrícula: 202.190.033 / 202.190.032 / 202.100.142

Pontifícia Universidade Católica do Rio de Janeiro

Curso de Pós Graduação *Business Intelligence Master*


