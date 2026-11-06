How the agent trains

Training is coordinated by run_deep_q_learning(), a module-level function defined in DeepQLearning.py. It creates the environments and agent, runs the training episodes, and periodically evaluates and saves the controller.

The environment is implemented separately in QLearning.py, imported as qlearning. Calls such as qlearning.potential_shaped_reward() therefore refer to functions defined in that file.

1. Configure the experiment

DeepExperimentDefaults is a frozen dataclass defined in DeepQLearning.py. It holds the training settings, including episode counts, network architecture, learning rate, replay capacity, exploration schedule, and evaluation interval.

Its environment_settings() method converts the shared settings into an instance of qlearning.ExperimentDefaults, the configuration class defined in QLearning.py.

At the beginning of run_deep_q_learning():

seed_everything() seeds the random generators and requests deterministic PyTorch operations.

select_device() chooses the configured CPU, CUDA, or MPS device.

Two qlearning.LorenzEnvEuler instances are created: one for training and one for evaluation.

A DoubleDQNAgent instance is created to manage action selection and learning.

Training and evaluation each default to 50 Lyapunov times. The environment uses lyapunov_times_to_steps(), defined in QLearning.py, to calculate the horizon:

H = round(lyapunov_times / (0.9056 × 0.01))

This gives a default horizon of 5,521 steps.

2. Create the networks and replay memory

DoubleDQNAgent.__init__() constructs the components that persist throughout training:

online_network: the network whose weights are updated through learning.

target_network: a separate network used to calculate learning targets.

normalizer: a StateNormalizer instance.

replay: a ReplayBuffer instance.

optimizer: a PyTorch Adam optimizer attached to the online network’s parameters.

QNetwork, StateNormalizer, and ReplayBuffer are classes defined in DeepQLearning.py.

QNetwork inherits from PyTorch’s nn.Module. Its __init__() method defines the layers using nn.Sequential, and its forward() method passes input states through those layers.

The default architecture is:

3 state inputs → Linear(3, 2) → Tanh
               → Linear(2, 2) → Tanh
               → Linear(2, 9) → Q-value for each action

This contains two hidden layers and three linear layers in total. The output layer has no activation.

Each output is a Q-value: an estimate of the discounted future reward associated with choosing that action and following the learned policy afterward. The controller prefers higher Q-values.

The hidden activation can be configured as tanh or relu. This setting is separate from the tanh expression in the reward’s state-cost function.

Initially, the target network receives a copy of the online network’s weights.

3. Reset the environment

At the beginning of each episode, run_deep_q_learning() calls:

state = training_env.reset()

reset() is a method of LorenzEnvEuler, defined in QLearning.py. It selects an initial state, resets the environment’s step counter and termination flag, and returns the starting coordinates.

By default, initial states come from a balanced attractor ensemble containing 20 states from each lobe. ExperimentDefaults.make_training_initial_state_sampler() builds the sampler using make_attractor_initial_states() and make_ensemble_initial_state_sampler().

Training cycles through shuffled ensemble members. Evaluation uses a fixed array produced by ExperimentDefaults.make_evaluation_initial_states(). With the default settings, evaluation uses the same ensemble states as training.

Resetting the environment does not reset the agent’s networks, optimizer, replay memory, or exploration schedule.

4. Select a forcing action

For each environment step, the training loop calls:

action = agent.select_action(state, training=True)

select_action() is a method of DoubleDQNAgent. It first calls StateNormalizer.normalize() to clip and rescale the coordinates into [-1, 1].

During training, it uses epsilon-greedy selection:

With probability epsilon, choose a uniformly random action index.

Otherwise, pass the normalized state through the online network and choose the highest-Q action.

DoubleDQNAgent.epsilon is defined as a property. It calculates exploration from the current global_step:

ε = ε_start
    + min(1, global_step/epsilon_decay_steps)
      × (ε_end − ε_start)

By default, epsilon decreases from 1.0 to 0.02 over 500,000 observed training transitions.

The agent’s _preferred_argmax() method resolves equal Q-values by preferring the smallest absolute forcing, then the lower action index.

5. Advance the Lorenz system and calculate reward

The training loop passes the action index to the environment:

next_state, environment_reward, done, info = training_env.step(action)

LorenzEnvEuler.step(), defined in QLearning.py, converts the index into a forcing value and calls the class’s static method _euler_step() to advance the Lorenz equations.

It then calculates the environment reward using the state cost and control penalty. The shared lobe_state_cost() function handles fixed-point proximity and the smooth penalty based on the x coordinate.

Back in run_deep_q_learning(), the environment reward is passed through the module-level function qlearning.potential_shaped_reward():

r = r_env + w[γΦ(s_next) − Φ(s)]

That function uses negative_target_potential(), also defined in QLearning.py, to calculate the potential from distance to the negative fixed point.

The resulting shaped reward is used for learning and accumulated into the episode return.

6. Store the transition

The training loop calls:

agent.observe(state, action, reward, next_state, done)

DoubleDQNAgent.observe() normalizes the current and next states, stores the transition through ReplayBuffer.add(), and increments global_step.

Each replay entry contains:

(normalized_state, action_index, reward, normalized_next_state, done)

ReplayBuffer uses preallocated NumPy arrays as circular storage. Once full, new transitions overwrite the oldest entries.

Storing experience allows the agent to learn from earlier interactions rather than only the latest transition.

7. Sample experience and update the online network

After storing the transition, the loop calls:

training_stats = agent.train_step()

train_step() is a method of DoubleDQNAgent. It returns without updating until replay contains at least:

max(minimum_replay_size, batch_size)

The defaults are 10,000 transitions for warm-up and 128 transitions per batch.

Once enough experience exists, ReplayBuffer.sample() selects a random batch without replacement. This reduces the dependence between consecutive samples from the Lorenz trajectory.

For each sampled transition, the update is:

a_next = preferred_argmax Q_online(s_next, a)

target = r + γ(1 − done)Q_target(s_next, a_next)

prediction = Q_online(s, action)

loss = smooth_l1_loss(prediction, target)

The online network selects the next action, and the target network evaluates it. This separation defines the Double DQN update and helps reduce Q-value overestimation.

Targets are calculated inside torch.no_grad(). Terminal transitions use only the observed reward because the future-value term is multiplied by zero.

train_step() then:

Clears previous gradients using optimizer.zero_grad().

Calls loss.backward() to calculate gradients.

Clips their norm using nn.utils.clip_grad_norm_().

Calls optimizer.step() to update the online network’s weights.

These updates move the predicted Q-values toward the calculated targets. The returned gradient_norm records the norm before clipping.

8. Update the target network

Inside DoubleDQNAgent.train_step(), the online weights are copied into the target network when:

global_step % settings.target_update_interval == 0

The default interval is 1,000 observed transitions. The target network remains unchanged between these copies, helping stabilize the learning targets.

An episode-based target update schedule is a possible future experiment.

9. Repeat and evaluate

The interaction and learning cycle repeats until the environment terminates or the configured training step cap is reached:

Select action
    → Advance environment
    → Calculate shaped reward
    → Store transition
    → Sample replay and learn
    → Continue from next state

After the episode, run_deep_q_learning() records rewards, episode length, exploration level, learning diagnostics, and selected trajectories.

At the configured interval, it calls evaluate_deep_q_learning(), a module-level function in DeepQLearning.py.

Evaluation calls agent.select_action(state, training=False), so it always acts greedily. It does not call observe() or train_step() and therefore does not add replay entries or update network weights.

DoubleDQNAgent.checkpoint() captures the network, optimizer, counters, and random-generator states. The functions checkpoint_is_eligible() and checkpoint_rank() determine which evaluated checkpoint qualifies as the best controller.

After all training episodes, run_deep_q_learning() returns a dictionary containing the training history, evaluations, configuration, and checkpoints. The command-line main() function then saves the run archive and generates plots.DeepQLearning.py 
This file implements a neural network for Q learning in order to create a controller for the negative lobe of the lorenz system the majority of the time. 
The controller observes three continuous coordinates chooses a discrete forcing value, and learns to favor the negative lobe using minimal forcing 
Note the file also includes plotting code to be moved eventually 

Lorenz 
The file currently still uses euler integration with time step 0.01
To be adjusted to rk4 in a future edit
The system is defined using:
```text
dx/dt = 10(y − x) + u
dy/dt = x(28 − z) − y
dz/dt = xy − (8/3)z


s_next = s + 0.01 × (dx/dt, dy/dt, dz/dt)
```
Forcing `u` enters the x derivative to be propagated to the other variables
The def action array is defined as 
`np.linspace(-2.5, 2.5, 9)`:


```text
Action index:  0      1       2       3      4     5      6      7      8
Forcing u:   -2.5  -1.875  -1.25  -0.625   0   0.625   1.25   1.875   2.5
```
The agent then returns an action index for the environment to convert that index into a forcing(u) value 

Reward function: 

The reward function is based off the lobe occupancy, it uses the non trivial fixed point formulas:  C+=(-1),(-1),-1, C-=-(-1),-(-1),-1) to calculate the reward based on the distance to the actual fixed point. To do this it uses the euclidean distance formula for the negative lobe. 
Note there is no reward if a certain distance to the positive lobe 
 `0` within Euclidean distance `2` of the negative fixed point.
 `1` within distance `2` of the positive fixed point.
 Otherwise, `0.5 × (1 + tanh(x / 2))`.
Note: Negative-lobe occupancy is the percentage of finite post-step x coordinates below zero. The initial observation is excluded, and `x=0` belongs to the positive side.
Env reward is def as: 
r_env = −[state_cost(s_next)/(H + 1) + α(u/u_ref)²/H]
Potential shaping is implemented,
r = r_env + w[γΦ(s_next) − Φ(s)]
Φ(s) = −d/(1 + d)
d = ||s − negative_fixed_point|| / ||negative_fixed_point||
`evaluate_deep_q_learning()` runs the greedy policy over an explicit finite `(n, 3)` array of initial states
It records action controls, trajectories, lengths, divergence flags, and target proximity statistics without adding replay entries or updating the network 

Normalization 
The network uses continuous cords without bins
StateNormalizer clips each cord to its bounds and rescales into `[-1, 1]`:
```text
normalized = (clip(state, lower, upper) − center) / half_width
center = (lower + upper) / 2
half_width = (upper − lower) / 2
```
Default bounds are `x, y ∈ [-30, 30]` and `z ∈ [0, 60]
Normalized arrays use float32 

Training 
At 
Training and evaluation are each default 50 lyapunov time 


```text
H = round(lyapunov_times / (0.9056 × 0.01))
```
Default horizon is 5,521 steps 
Our reward function for training and eval is tanh, with the ability to reimplement relu
Initial states
 Initial states are random between training episodes, while for eval they are always the same 
Training uses epsilon greedy selection, where it takes a random action to help inform the target algorithm  
Evaluation always chooses greedily.
Never uses epsilon action
Note: for target def 
Epsilon decreases linearly with number of observed training transitions: 
```text
ε = ε_start + min(1, global_step/decay_steps) × (ε_end − ε_start)
```
Defaults from 1.0 to 0.02 under current l-time given a run 
Can and will change under future runs 
Training also falls back to reference-state sampling when `training_ic_perturbation` is zero.


Learning update
`train_step()` waits until replay contains at least `max(minimum_replay_size, batch_size)` entries.
For each sampled transition, the update is:
a_next = preferred_argmax Q_online(s_next, a)
target = r + γ(1 − done)Q_target(s_next, a_next)
prediction = Q_online(s, action)
loss = smooth_l1_loss(prediction, target)

The online network selects the next action and the target network evaluates it 
Targets are computed without gradients
 torch.no_grad()
Terminal transitions use only the observed reward because the future value term is *0
Optimization uses Adam, Huber loss (`smooth_l1_loss`), and gradient norm clipping.
For training: 
train_step() then:


Clears previous gradients using optimizer.zero_grad().


Calls loss.backward() to calculate gradients.


Clips their norm using nn.utils.clip_grad_norm_().


Calls optimizer.step() to update the online network’s weights.
 
During active learning online weights are copied to the target network when `global_step % target_update_interval == 0`
We use observed transitions rather than episode count
Maybe we shouldn’t, maybe we'll try the other option 
Gradient_norm records the norm reported by the clipping op before clipping 

Double DQN algorithm 
The default QNetwork is: 
```text
3 state inputs → Linear(3, 2) → Tanh
               → Linear(2, 2) → Tanh
               → Linear(2, 9) → Q-value for each action
```
Where the final layer has no activation
The default uses two hidden layers with 2 neurons each to keep computation minimal.

Commands
Full experiment
```sh
python Q-Learning/DeepQLearning.py --device auto
```
Choose an archive root
 ```sh
python Q-Learning/DeepQLearning.py --smoke --device cpu --output-directory ./dqn_results
```
Regenerate plots without training
```sh
python Q-Learning/DeepQLearning.py --plot-run /path/to/deep_q_learning_run.pkl.gz
```
Compare with a tabular checkpoint
```sh
python Q-Learning/DeepQLearning.py --device cpu --tabular-run /path/to/q_learning_run.pkl.gz
```
Run focused tests
```sh
python -m unittest discover -s Q-Learning -p 'TestDeepQLearning.py'
```

