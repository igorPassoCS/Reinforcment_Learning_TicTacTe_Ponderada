# Project Handoff: RL Tic-Tac-Toe

This repository contains a Jupyter Notebook implementation of Tic-Tac-Toe as a custom Reinforcement Learning environment and a from-scratch Tabular Q-Learning agent.

## Main File

- `TicTacToe_RL.ipynb`: primary notebook containing the environment, sanity check, Q-learning agent, training loop, metrics, and plots.
- `README.md`: currently empty.

## Current Implementation Status

### Step 1: Simulation Environment Development

Implemented in `TicTacToe_RL.ipynb`.

Main class:

```python
class TicTacToeEnv:
```

Important components:

- `self.board = np.zeros(9, dtype=int)`: board represented as a 1D array of 9 positions.
- `AGENT = 1`, `OPPONENT = -1`, `EMPTY = 0`.
- `WINNING_LINES`: all 8 Tic-Tac-Toe win conditions.
- `reset()`: resets the environment and returns the initial board copy.
- `get_state_hash()`: returns the current board as an immutable tuple for tabular RL.
- `get_available_actions()`: returns valid moves, i.e. empty board indices.
- `check_winner()`: checks whether the agent or opponent has won.
- `_is_draw()`: checks terminal draw condition.
- `step(action)`: applies the agent move, checks terminal conditions, then applies a random opponent move if needed.
- `render()`: prints the board to standard output.

Rewards:

- Agent win: `+1`
- Opponent win: `-1`
- Draw: `0`
- Non-terminal step: `0`

The opponent is already built into `env.step(action)` and chooses randomly from valid actions.

### Step 2: Agent Training With Bellman Equations

Implemented later in `TicTacToe_RL.ipynb`.

Main class:

```python
class QLearningAgent:
```

Important components:

- `self.q_table = {}`: empty dictionary for tabular action-value storage.
- `_state_key(state)`: converts NumPy board states into tuple keys.
- `_ensure_state_exists(state)`: initializes unseen states with `np.zeros(9, dtype=float)`.
- `choose_action(state, available_actions)`: epsilon-greedy action selection.
- `update(state, action, reward, next_state, next_available_actions)`: Bellman update.
- `decay_epsilon()`: multiplicative epsilon decay with lower bound.

The Bellman update implemented is:

```python
bellman_target = reward + self.gamma * max_next_q
new_q = current_q + self.alpha * (bellman_target - current_q)
```

This corresponds to:

```text
Q(s,a) = Q(s,a) + alpha * [R + gamma * max(Q(s',a')) - Q(s,a)]
```

For terminal states, `next_available_actions` is passed as an empty list, making `max_next_q = 0.0`.

### Training Loop

Main function:

```python
def train(env, agent, episodes=10000, batch_size=100):
```

The loop runs:

1. `env.reset()`
2. `agent.choose_action(state, available_actions)`
3. `env.step(action)`
4. `agent.update(...)`
5. `agent.decay_epsilon()`

The default training run is:

```python
metrics = train(env, agent, episodes=10000, batch_size=100)
```

Metrics tracked:

- `rewards_per_episode`
- `max_delta_per_episode`
- `batch_numbers`
- `agent_wins`
- `opponent_wins`
- `draws`

Plots generated:

- `Match Outcomes over Time`
- `Policy Convergence (Max Delta Q)`

## Dependencies

The notebook uses:

- Python 3.12
- `numpy`
- `matplotlib`
- Jupyter/IPython kernel support

Recommended minimal `requirements.txt` content:

```txt
numpy==2.4.4
matplotlib==3.10.9
ipykernel==7.2.0
```

Install in a fresh environment with:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If no `requirements.txt` exists yet, create one before continuing development.

## How To Validate

Run the notebook cells from top to bottom.

Expected behavior:

- The sanity check should play one full random episode and print a final board.
- The Q-learning cell should train for 10,000 episodes.
- Final epsilon should decay to approximately `0.05`.
- The Q-table should contain learned states.
- The first plot should show match outcomes per batch.
- The second plot should show maximum Q-value change per episode.

For terminal validation outside Jupyter, use the project virtual environment and a non-interactive matplotlib backend:

```bash
MPLBACKEND=Agg venv/bin/python3
```

Then execute notebook code manually or through a notebook runner if one is installed.

## Important Design Notes

- The environment is not OpenAI Gym-based. It is intentionally implemented from scratch for the assignment rubric.
- No external RL libraries should be added.
- The opponent is random and is part of the environment transition dynamics.
- The agent learns only from the final environment reward returned by `step()`.
- Invalid actions are avoided by passing only `env.get_available_actions()` into `choose_action()`.
- The Q-table stores 9 Q-values for every state, but action selection and max-next-value calculations only consider valid actions.

## Suggested Next Steps

1. Create a clean `requirements.txt` with the minimal dependencies.
2. Add a small evaluation function that runs the trained agent with `epsilon = 0.0` or a greedy policy.
3. Add a chart comparing performance before and after training.
4. Consider saving/loading the learned Q-table with `pickle` or JSON-compatible serialization.
5. Improve the README with setup instructions, notebook execution instructions, and assignment rubric mapping.
6. Optionally refactor the notebook code into `.py` modules after the assignment notebook is complete.

## Files A Future Agent Should Avoid Changing Casually

- Do not change the reward values unless the rubric changes.
- Do not replace the from-scratch Q-learning implementation with an RL library.
- Do not remove the random opponent from `env.step()` unless adding a separate configurable opponent mode.
- Do not change board encoding (`1`, `-1`, `0`) without updating every method that depends on it.
