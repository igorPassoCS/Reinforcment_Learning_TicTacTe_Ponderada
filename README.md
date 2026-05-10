# Tic Tac Toe com RL - Igor P. Sguissardi

Projeto de Reinforcement Learning para jogo da velha, com:
- ambiente customizado (`TicTacToeEnv`) modelado como MDP;
- agente `QLearningAgent` tabular;
- treinamento com Equacao de Bellman;
- graficos de desempenho e convergencia.

Abra o arquivo TicTacToe_RL.ipynb, e o desenvolvimento todo está lá.

## Vídeo explicativo
[Assista aqui](https://youtu.be/4-O6yoGhJwY?si=lVb3Ypj8l3MG99DV)

## Setup rapido
Caso você queira rodar você mesmo

### Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux / macOS
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Executar

Com o ambiente ativo:
```bash
jupyter notebook
```

Abra `TicTacToe_RL.ipynb` e execute as celulas em ordem.
