# Classificação de Imagens usando PyTorch

Este projeto em Python utiliza a biblioteca PyTorch para treinar um modelo de classificação de imagens usando a arquitetura de uma rede neural simples. O objetivo é classificar dígitos escritos à mão da base de dados MNIST.

## Funcionamento

- Carregamento dos Dados: Utiliza o dataset MNIST para treinamento do modelo.
- Definição do Modelo: Define uma arquitetura de rede neural simples para classificação de imagens.
- Treinamento: Utiliza o otimizador Adam e a função de perda CrossEntropy para treinar o modelo.
- Salvando/Carregando o Modelo: Salva o estado treinado do modelo em um arquivo e carrega-o novamente.
- Inferência em uma Nova Imagem: Carrega uma nova imagem, a converte em tensor e realiza inferência com o modelo treinado.

Esse projeto é um exemplo básico de classificação de imagens usando PyTorch e pode ser expandido para problemas mais complexos.
