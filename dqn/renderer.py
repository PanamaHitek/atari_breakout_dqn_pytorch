import os
import torch
import dqn as dqn
from atari_wrappers import wrap_deepmind, make_atari
from collections import deque

# path to the pre-trained policy network
pretrained_policy_path = "trained_policies/trained_policy_network.pt"
# maximum number of steps allowed per episode
max_episode_steps=400000
# number of evaluation episodes to run
num_eval_episode = 10

# function to load the pre-trained model from file
def load_pretrained_model(model, path):
    if os.path.isfile(path):
        # if the file exists, load the pre-trained model
        print("Loading pre-trained model from", path)
        model.load_state_dict(torch.load(path))
    else:
        # if the file does not exist, raise an error
        raise FileNotFoundError("Pre-trained model not found.")

# function to evaluate the policy network's performance over a specified number of episodes
def evaluate(policy_net, device, environment, n_actions, num_episode):
    # wrap the environment with DeepMind wrappers to preprocess the inputs
    environment = wrap_deepmind(environment)
    # create an action selector object to choose actions according to the policy network's output
    action_selector = dqn.ActionSelector(0.05, policy_net, n_actions, device)
    # initialize an empty list to store the rewards obtained in each episode
    episodes_rewards = []
    # create a deque object to store the last 5 frames of the game as input to the policy network
    frame_stack = deque(maxlen=5)
    # run the specified number of episodes
    for i in range(num_episode):
        # reset the environment at the beginning of each episode
        environment.reset()
        # initialize the episode reward to zero
        current_episode_reward = 0
        # fill the frame stack with the first 5 frames of the game
        for _ in range(10):
            pixels, _, done, _ = environment.step(0)
            pixels = dqn.get_frame_tensor(pixels)
            frame_stack.append(pixels)
        # play the game until the episode ends
        while not done:
            # render the game screen
            environment.render()
            # concatenate the last 4 frames of the frame stack into a single state tensor
            state = torch.cat(list(frame_stack))[1:].unsqueeze(0)
            # select an action to take using the policy network and the action selector object
            action, _ = action_selector.select_action(state)
            # take the chosen action and observe the next state and reward
            pixels, reward, done, info = environment.step(action)
            pixels = dqn.get_frame_tensor(pixels)
            # add the new frame to the frame stack
            frame_stack.append(pixels)
            # update the episode reward
            current_episode_reward += reward
        # print the reward obtained in this episode
        print("Reward: ",current_episode_reward)
        # add the episode reward to the list of rewards obtained in all episodes
        episodes_rewards.append(current_episode_reward)

# main function to set up the game environment, load the pre-trained model, and run evaluation episodes
def main():
    # set the device to use for running the model (CPU or GPU)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # name of the Atari game
    env_name = 'Breakout'
    # create the Atari game environment with DeepMind wrappers
    env_raw = make_atari('{}NoFrameskip-v4'.format(env_name),max_episode_steps, True)
    # wrap the environment with DeepMind wrappers to preprocess the inputs
    env = wrap_deepmind(env_raw, frame_stack=False, episode_life=True, clip_rewards=True)
    # get the number of possible actions in the environment
    possible_actions = env.action_space.n
    # create a DQN object to store the policy network and move it to the selected device (CPU or GPU)
    policy_net = dqn.DQN(possible_actions, device).to(device)
    # load the pre-trained model from file into the policy network
    load_pretrained_model(policy_net, pretrained_policy_path)
    # evaluate the performance of the policy network over a specified number of episodes
    evaluate(policy_net, device, env_raw, possible_actions, num_episode=num_eval_episode)

if __name__ == '__main__':
    main()
