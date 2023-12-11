# Lunar Lander Project

## Lunar Lander Environment
Estamos usando o ambiente ‘Lunar Lander’ da biblioteca gymnsasium da OpenAI. Este ambiente lida com o problema de pousar um módulo de pouso em uma plataforma. As etapas para configurar esse ambiente são mencionadas na página GitHub da documentação biblioteca gymnasium  OpenAI e na documentação da OpenAI. A seguir estão as variáveis env em resumo para entender o ambiente em que estamos trabalhando.

- Estado: O estado/observação é apenas o estado atual do ambiente. Existe um espaço de estado contínuo de 8 dimensões e um espaço de ação discreto.
- Ação: Para cada estado do ambiente, o agente executa uma ação com base em seu estado atual. O agente pode escolher entre quatro ações distintas: do_nothing, fire_left_engine, fire_right_engine e fire_main_engine.
- Recompensa: O agente recebe uma pequena recompensa negativa toda vez que age. Isso é feito na tentativa de ensinar o agente a pousar o foguete da maneira mais rápida e eficiente possível. Se o módulo cair ou parar, o episódio é considerado completo e receberá -100 ou +100 pontos adicionais, dependendo do resultado.

## Análise da implementação do algorimo deep q-learning
O modelo de final tem os seguintes hiperparâmetros:

taxa de aprendizado = 0,001
gama = 0,99
epsilon replay_memory_buffer_size = 500000
epsilon_decay = 0,995

Após 600 o agente está totalmente treinado. Ele aprende a manusear o foguete perfeitamente e aterrissa o foguete perfeitamente a cada vez.

![alt text](https://github.com/insper-classroom/lunarlander-gu/blob/main/images/Modelo_Treinado.gif)

Análise de resultados
Figura 1. A recompensa para cada episódio de treinamento

![alt text](https://github.com/insper-classroom/lunarlander-gu/blob/main/images/Figure_1_Reward%20for%20each%20training%20episode.png)

A Figura 1 mostra os valores de recompensa por experiência no momento do treinamento. As linhas azuis indicam a recompensa para cada episódio de treinamento e a linha laranja mostra a média móvel dos últimos 100 episódios. O agente continua aprendendo com o tempo e o valor da média móvel aumenta com os episódios de treinamento.

A recompensa média nos episódios anteriores é principalmente negativa porque o agente acabou de começar a aprender. Eventualmente, o agente começa a ter um desempenho relativamente melhor e a recompensa média começa a subir e se tornar positiva após 300 episódios. Após 514 episódios, a média móvel cruza 200 e o treinamento é concluído. Há alguns episódios em que o agente recebeu prêmios negativos neste momento, mas acredito que se o agente puder continuar treinando, essas instâncias serão reduzidas.

Figura 2. A recompensa para cada episódio de teste

![alt text](https://github.com/insper-classroom/lunarlander-gu/blob/main/images/Figure_2_Reward%20for%20each%20testing%20episode.png)

A Figura 2 mostra o desempenho do modelo treinado para 100 episódios no ambiente Lunar Lander. O modelo treinado está tendo um bom desempenho no ambiente com todas as recompensas sendo positivas. A recompensa média por 100 episódios de teste é 205.

## Utilizando a implementação do DQL da Bibblioteca stable_baselines3

Foi utilizado o jupternotebook para usar essa biblioteca e todas as suas dependencias serão instaladas ao rodar a primeira celula 

Um problema que eu tive na implementação foi que não consegui mudar o epsilon por isso utilizei mais episódios para compensar.

os resultados a seguir demostram que foi um sucesso más a implementação poderia ser melhor se eu tivese conseguido alterar todos os hiperparametros.

![alt text](https://github.com/insper-classroom/lunarlander-gu/blob/main/images/LunarLander_StatsBaselineDQL.gif)

Figura 3. A recompensa para cada episódio de treinamento

![alt text](https://github.com/insper-classroom/lunarlander-gu/blob/main/images/LunarLander_StatsBaselineDQL.png)



