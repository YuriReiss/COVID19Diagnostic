# Diagnóstico de COVID-19 Utilizando CNN 

Universidade Federal de Goiás - UFG

Alunos: 
Milton Alexandre Souza Demarchi e Yuri dos Reis de Oliveira.

Trabalho realizado através do treinamento de um modelo de Rede Neural Convolucional utilizando do dataset COVID-19 Radiography Database disponibilizado pela plataforma Kaggle.

Para gerar treinar os 3 modelos que foram implementados, basta utilizar dos seguintes comandos:
"python modeloOriginalMinimamenteModificado.py" para gerar o modelo Minimamente Modificado,
"python modeloPorExploratorio.py" para gerar o Modelo por Exploratório,
"python modeloUsandoArquiteturaDensenet.py" para gerar o modelo utilizando arquitetura Densenet.

Após isso, os modelos serão salvos na pasta models. Utilize o arquivo load.py para testar os modelos gerados, surgirá um menu interativo.

Atenção: Os modelos que já estão commitados aqui no repositório possuem assinatura da máquina gerada. Então, caso vá utilizar um dos modelos salvos na pasta "models", poderá ocorrer algum erro. Sugerimos que execute o treinamento em sua máquina e gere os modelos novamente, assim não haverá problema.

Estamos utilizando python 3.7.4, pois o tensorflow aceita apenas entre 3.6 e 3.9 (https://www.tensorflow.org/install?hl=pt-br). A versão do tensorflow utilizada foi a 2.11.0. 
Uma sequência de comando que fizemos foi:
pip install scikit-learn
pip install opencv-python matplotlib
pip install tensorflow
pip install opencv-python
pip install scikit-learn
pip install matplotlib

Isso após instalar a versão 3.7.4 do python. O SO Windows 11 foi utilizado.
Mas fique a vontade pra instalar as bibliotecas da forma que for mais fácil.
