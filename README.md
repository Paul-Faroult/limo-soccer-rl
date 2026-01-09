\# Limo Soccer RL



\## Autonomous Robot Soccer using Reinforcement Learning



This project explores the training of an autonomous wheeled robot to play a simplified

soccer game using reinforcement learning techniques (PPO).



The work follows a progressive experimental pipeline, starting from a simple goal-scoring

task and culminating in a competitive multi-agent duel.



---



\## Objectives



\- Learn robust navigation and ball control

\- Avoid overfitting to fixed trajectories

\- Handle adversarial behaviors

\- Compare classical and hierarchical RL approaches



---



\## Experimental Pipeline



The project is fully versioned using Git branches:



\### Main stages (validated)



\- \*\*Stage 1 — But cage (Lidar-based)\*\*

&nbsp; - Fixed robot start

&nbsp; - Ball at center

&nbsp; - Lidar-only observations



\- \*\*Stage 2 — Classic RL\*\*

&nbsp; - Randomized robot and ball positions

&nbsp; - PPO with γ = 0.99



\- \*\*Stage 3 — Static opponent\*\*

&nbsp; - Frozen adversary

&nbsp; - Explicit collision-avoidance reward



\- \*\*Stage 4 — Duel\*\*

&nbsp; - Two agents, independent goals

&nbsp; - Competitive reinforcement learning



\### Experiments (abandoned or exploratory)



\- Hierarchical reinforcement learning

\- PPO with γ = 0.995

\- Static opponent without collision penalty



---



\## Repository Structure



\- `main` → Final duel implementation

\- `dev` → Integration branch

\- `stage/\*` → Validated development milestones

\- `experiment/\*` → Exploratory or discarded approaches



---



\## Tools \& Libraries



\- Python

\- Stable-Baselines3 (PPO)

\- Gymnasium

\- NumPy

\- PyGame (rendering)

\- TensorBoard (training analysis)



---



\## Results



The final agent demonstrates:

\- Robust ball interception

\- Adaptive opponent avoidance

\- Competitive goal scoring behavior



---



\## Author

Faroult Paul

Nguyen Ba Thong



Academic project developed for reinforcement learning research and experimentation.



