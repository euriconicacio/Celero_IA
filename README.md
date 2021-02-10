# Celero_IA

### Desafio Celero - IA com Python

O arquivo com o código Python para análise de sentimento é o **celero.py**. Ele possui limite para importação de arquivos para treinamento e teste - estes limites podem ser impostos adicionando os parâmetros "*limit=xxx*" nas chamadas das funções *load_training_data* e *load_test_data* (cf. linhas 174 e 175), bem como eliminados ao suprimir tal parâmetro. De maneira similar, foram estabelecidas 20 iterações para aperfeiçoamento do modelo (cf. linha 99).

Exemplo de execução:

![exec](https://github.com/euriconicacio/Celero_IA/blob/main/exec.png)

Obs.: não foi realizado o upload dos arquivos de treinamento e teste para os reviews positivos e negativos por motivos de celeridade e volume de dados. Todavia, todos seguem a estrutura original do arquivo compactado baixado em http://ai.stanford.edu/~amaas/data/sentiment/.
