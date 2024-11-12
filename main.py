
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
from RL_agent_model_free.model_free_RL_train import GymAgent
from models.LSTM_model import LSTMModel, estimate_loss, get_action_batch
import gym


################----MODEL-FREE RL AGENT----################
#Agent hyperparameters
max_runs_reward = 400 #maximize the number of steps during training simulation (equivalent to the maximum reward because you can get at most 1 for reward per time step)
image_size_rows = 50 #max_runs_reward
num_images = 1000
num_random_images = 200
environment = 'CartPole-v1'
file_path_RL = 'saved_models/cartpole_policy_net.pth'
file_path_save = 'data/cartpole_data.npy'
train_model_free_agent = False
collect_data = False


#initialize agent class
agent = GymAgent(environment)
if train_model_free_agent:
    print("Training model free agent...")
    agent.train(max_runs_reward,file_path_RL)


if collect_data:    
    print("Collecting data...")
    agent.collect_data(file_path_RL,num_images,image_size_rows,file_path_save)
    #agent.collect_random_data(num_random_images,image_size_rows)

#load data
print("loading data...")
data = np.load(file_path_save)



####################----MODEL-BASED LSTM to learn the state-action-rewards----################
# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 40 # what is the maximum context length for predictions?
max_iters = 1000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 10
n_embd = 6
hidden_size = 100
num_layers = 1
# ------------

torch.manual_seed(1337)

perform_initial_training = True #set to False if you want to load a pre-trained model
#-------------------initial training-------------------#

model = LSTMModel(n_embd, hidden_size, num_layers=num_layers)

# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

if perform_initial_training:
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # Ensure model is in training mode
    model.train()

    for iter in range(max_iters):

        
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, data, eval_iters, block_size, batch_size, num_images)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")
        
        # sample a batch of             data
        batch, target_ts_batch = get_action_batch(data, batch_size, block_size, num_images, train_test='train')
    
        # evaluate the loss and backpropagate
        outputs, loss = model(batch, target_ts_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()


        optimizer.step()
    #save the model
    torch.save(model.state_dict(), 'saved_models/cartpole_lstm_model.pth')
else:
    model.load_state_dict(torch.load('saved_models/cartpole_lstm_model.pth'))
    model.eval()

# generate from the model
model.eval()
context = batch[0,0:block_size,:].squeeze(0)
predicted_actions = model.generate(context, max_new_tokens=10)


plt.figure()
plt.imshow(predicted_actions.detach().numpy())
plt.show()

def visualize(env_name, model):
    env = gym.make(env_name, render_mode='human')

    

    state,_ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        env.render()
        state_action_reward = model.generate(context, max_new_tokens=1)
        #print("state action reward shape: ",state_action_reward.shape)
        action = state_action_reward[-1,-2]
        if action > 0.5:
            action = 1
        else:
            action = 0
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
    

    env.close()
    print(f"Total Reward: {total_reward}")


visualize(environment, model)


#-------------------fine-tuning-------------------#