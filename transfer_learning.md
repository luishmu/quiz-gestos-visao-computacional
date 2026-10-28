### Transfer learning na arquitetura proposta

A solução adota transfer learning na modalidade de **extração de características** (*feature extraction*). Nessa abordagem, um modelo pré-treinado numa tarefa ampla e com muitos dados é reaproveitado, sem alteração dos seus pesos, como extrator de representações. Sobre essas representações treina-se apenas um classificador leve, específico para a tarefa de interesse.

O modelo base é o **MediaPipe Hand Landmarker**, uma rede neural treinada pelo Google num grande volume de imagens de mãos, com variação de pessoas, tons de pele, iluminação e fundos. A partir de cada quadro da webcam, ele estima a posição tridimensional de 21 pontos anatômicos da mão (pulso, articulações e pontas dos dedos). No projeto, esse modelo fica **congelado**: seus pesos não são ajustados, e ele funciona como a primeira etapa do pipeline.

As 63 coordenadas resultantes passam por duas transformações:

1. **Normalização geométrica:** translação pelo pulso, espelhamento da mão esquerda para parecer direita e escala pela maior distância ao pulso.
2. **Extração de features invariantes:** ângulos de dobra dos dedos, distâncias relativas à palma e direção da palma e do polegar.

Só então um **perceptron multicamadas (MLP)** é treinado para distinguir os gestos do quiz. Na terminologia de transfer learning, o Hand Landmarker é o *backbone* pré-treinado e o MLP é a *cabeça* treinada para a nova tarefa.

Essa decisão tem consequências práticas importantes:

- **Eficiência de dados:** o conhecimento mais difícil de obter, que é localizar a mão e suas articulações em imagens reais, já vem do modelo pré-treinado. Por isso, alguns milhares de quadros coletados por poucas pessoas bastam para treinar o classificador. Treinar uma rede do zero sobre os pixels exigiria um volume de dados muito maior.
- **Robustez a fatores irrelevantes:** como o classificador recebe só a geometria da mão, fundo, iluminação, vestuário e tom de pele não chegam até ele. Isso reduz o risco de o modelo aprender atalhos espúrios, como associar um gesto ao cenário em que foi gravado.
- **Generalização entre pessoas:** a avaliação usa validação cruzada deixando uma pessoa de fora (*leave-one-group-out*). Com 8.228 quadros de 4 participantes, cada pessoa serve uma vez como teste para modelos treinados com as outras três. Nesse protocolo, o baseline por regras atingiu 87,9% ± 4,7% de acurácia média, o MLP sobre coordenadas normalizadas 91,2% ± 2,0% e o MLP sobre features invariantes 92,8% ± 2,2% (F1 macro médio de 0,882, 0,913 e 0,929). Mesmo sem features elaboradas, um classificador simples sobre os landmarks já generaliza bem para pessoas que não participaram do treino, o que evidencia o valor da representação vinda do modelo pré-treinado. As features invariantes trazem um ganho adicional modesto: superam as coordenadas em 3 dos 4 participantes, mas a diferença média (~1,6 ponto percentual) fica dentro da variação entre pessoas. As regras são as mais sensíveis a quem faz o gesto, variando de 81% a 94% conforme o participante. Num split com dois participantes no teste, os erros restantes do MLP invariante se concentram em `quatro` (precisão de 0,75), `palma` (revocação de 0,78) e `tres` (revocação de 0,81), enquanto `joinha` foi classificado sem erros.
- **Custo computacional:** o classificador é pequeno e roda em tempo real na CPU, o que permite a interação fluida exigida pelo quiz.

Uma alternativa seria o **ajuste fino** (*fine-tuning*) de uma rede convolucional pré-treinada, como MobileNet ou ResNet, diretamente sobre as imagens. Ela foi considerada menos adequada a este cenário. Com poucos participantes, uma rede que opera sobre pixels tende a se ajustar a características do ambiente de coleta. Além disso, alguns pares de gestos diferem por detalhes geométricos finos, como `joinha` e `deslike`, que se distinguem apenas pela direção do polegar. A representação por landmarks expõe esse tipo de diferença de forma explícita.

Como trabalho futuro, a mesma lógica de transfer learning pode ser estendida à **adaptação ao usuário**. O MLP treinado com todos os participantes serviria de ponto de partida e seria ajustado com poucos segundos de amostras de uma pessoa nova. Isso tende a reduzir os erros de quem não participou da coleta, a um custo de coleta mínimo.
