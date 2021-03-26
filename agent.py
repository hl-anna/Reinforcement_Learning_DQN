import numpy as np
import torch
from collections import deque


class Agent:

    # Function to initialise the agent
    def __init__(self):
        
        # Set the episode length
        self.episode_length = 500
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        self.total_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        self.continuous_action = None
        
        
        self.epsilon = None
        self.network = DQN()
        self.target_net = DQN()
        self.replay_buffer = ReplayBuffer()
        self.minibatch_size = 100
        self.episode_count = 0
        self.update_frequency = 300
        self.training_bool = True
        self.greedy_test_bool = False
        self.greedy_end_distance = np.sqrt(2) #setting max distance as default

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        
        if self.num_steps_taken % self.episode_length == 0:
            
            if self.greedy_test_bool and self.greedy_end_distance <= 0.03:
                self.greedy_test_bool = False

            elif self.episode_count > 10:
                print('greedy episode result', self.greedy_end_distance)
            
            if self.episode_count >= 10 and not self.greedy_test_bool: #if 
                self.epsilon = 0
                self.episode_length = 100
                self.training_bool = False
                self.greedy_test_bool = True

            else:     
                self.epsilon = min(0.9, (1/(self.episode_count-3) if self.episode_count > 3 else 1))
                self.episode_length = max(500-self.episode_count*20, 100)
                self.episode_count += 1
                self.training_bool = True
                self.greedy_test_bool = False
                print('starting training episode ', self.episode_count,' of length ', self.episode_length, ' with epsilon ', self.epsilon )
                
            self.num_steps_taken = 0

            return True
        
        else:
            return False
            
    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        
        if np.random.uniform() < max(0,self.epsilon):
            action = action = np.random.choice([0,1,2,3],1,p=[41/100, 28/100, 3/100, 28/100])[0]
        else:
            self.get_greedy_action(state)
            action = self.action
        #self.epsilon = self.epsilon - self.eps_decay
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        self.total_steps_taken += 1
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = action
        self.continuous_action = self._discrete_action_to_continuous()
        
        return self.continuous_action
    
    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self):
        step_size = 0.02
        if self.action == 1: #up / north
            continuous_action = np.array([0, step_size], dtype=np.float32)
        
        elif self.action == 0: #right / east
            continuous_action = np.array([step_size, 0], dtype=np.float32)
            
        elif self.action == 3: # down / south
            continuous_action = np.array([0, -step_size], dtype=np.float32)
            
        elif self.action == 2: #left / west
            continuous_action = np.array([-step_size, 0], dtype=np.float32)
        
        self.continuous_action = continuous_action
        
        return continuous_action


    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        if (self.state[0] == next_state[0]) and (self.state[1] == next_state[1]):
            reward = 0.03 - distance_to_goal - 0.3
        else:
            reward = 0.03 - distance_to_goal

        # Create a transition
        transition = (self.state, self.action, reward, next_state)
        
        if self.training_bool:
            self.train_network(transition)
            
        if self.greedy_test_bool and self.num_steps_taken == self.episode_length:
            self.greedy_end_distance = distance_to_goal #this is only used to decide whether to stop training early
            
        return transition

    def train_network(self, transition):
        ###TRAIN HERE###
        
        #replay buffer stuff
        self.replay_buffer.add_transition(transition)
        self.replay_buffer.weights.append(max(self.replay_buffer.weights) if len(self.replay_buffer.weights)>0 else 1)

        #sample minibatch
        if len(self.replay_buffer.buffer) > self.minibatch_size:
                minibatch_tuple = self.replay_buffer.sample_minibatch(self.minibatch_size)
                self.network.train_q_network(minibatch_tuple, self.target_net)
                
                #update weights for the sampled transitions
                for k in self.replay_buffer.indices:
                    self.replay_buffer.weights[k] = abs(self.network.losses[list(self.replay_buffer.indices).index(k)].item())+self.replay_buffer.eps_const
                    #note: probabilities are updated in the sample_minibatch() function
                
        #target network update
        if ((self.total_steps_taken+1) % self.update_frequency) ==0:
            self.network.update_target_nn(self.target_net)
    
    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        # Here, the greedy action is fixed, but you should change it so that it returns the action with the highest Q-value
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        predictions = self.network.q_network.forward(state_tensor).data.numpy()[0]
        action = np.argmax(predictions)
        self.action = action
        continuous_action = self._discrete_action_to_continuous()
        return continuous_action


# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's 
    # input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=50)
        self.layer_2 = torch.nn.Linear(in_features=50, out_features=50)
        self.layer_3 = torch.nn.Linear(in_features=50, out_features=50)
        self.layer_4 = torch.nn.Linear(in_features=50, out_features=50)
        self.output_layer = torch.nn.Linear(in_features=50, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. 
    # In this example, a ReLU activation function is used for both hidden layers, but the output 
    # layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = torch.nn.functional.relu(self.layer_3(layer_2_output))
        layer_4_output = torch.nn.functional.relu(self.layer_4(layer_3_output))
        output = self.output_layer(layer_4_output)
        return output


# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        
        # Define the optimiser which is used when updating the Q-network. The learning rate 
        # determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.01)
        self.losses = None

    # Function that is called whenever we want to train the Q-network. Each call to this 
    # function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, minibatch_tuple, target_nn):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(minibatch_tuple, target_nn)
        # Compute the gradients based on this loss, i.e. the gradients of the loss 
        # with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        
        return loss.item()

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, minibatch_tuple, target_nn):
        
        #calculate target Q' values for each successor state in the minibatch 
        state_q_target_values = target_nn.q_network.forward(minibatch_tuple[3]) # minibatch size x 4
        
        #create list of best actions for each S'
        action_list = []
        for i in range(len(state_q_target_values)):
            action_list.append(np.argmax(state_q_target_values.detach().numpy()[i]))
        
        #transform into a tensor
        action_tensor = torch.tensor(action_list)
        action_tensor = action_tensor.type(torch.LongTensor)
        
        #get network values
        state_q_network_values = self.q_network.forward(minibatch_tuple[3]) # minibatch size x 4
        
        #get q values for main network based on best action of target network
        state_action_q_values = state_q_network_values.gather(dim=1, index=action_tensor.unsqueeze(-1)).squeeze(-1)
        action_tensor1 = minibatch_tuple[1].type(torch.LongTensor)
       
        network_prediction = self.q_network.forward(minibatch_tuple[0]).gather(dim=1, index = action_tensor1.unsqueeze(-1)).squeeze(-1)
        
        self.losses = [(r+(0.9*val) - pred) for pred,r,val in zip(network_prediction,minibatch_tuple[2],state_action_q_values)]
        loss = torch.nn.MSELoss()(network_prediction, minibatch_tuple[2]+ 0.91*state_action_q_values)
        return loss
    
    def update_target_nn(self, target_nn):
        target_nn.q_network.load_state_dict(self.q_network.state_dict())
        
    
class ReplayBuffer(object):
    
    def __init__(self):
        max_len = 10000 #all 3 deques need to have the same capacity
        self.buffer = deque(maxlen=max_len) #harry said 100k might be better but can i even make 100k transitions in 10mins
        self.weights = deque(maxlen=max_len)
        #self.probabilities = deque(maxlen=max_len)
        self.eps_const = 0.01 #this should be linked to max delta apparently
        self.alpha_const = 0.8
        self.indices = None
        
    def add_transition(self, transition):
        self.buffer.append(transition)
    
    def sample_minibatch(self, minibatch_size):
        
        #updating probabilities for the full replay buffer before sampling
        sum_weigths = sum([weight**self.alpha_const for weight in self.weights])
        probabilities = [(weight**self.alpha_const)/sum_weigths for weight in self.weights]
        
        #the actual sampling
        minibatch_indices = np.random.choice(range(len(self.buffer)), minibatch_size, replace = False, p = probabilities)
        self.indices = minibatch_indices #save as we need these in the Agent class
        
        #creating the output
        minibatch_states = [transition[0] for i, transition in enumerate(self.buffer) if i in minibatch_indices]
        minibatch_actions = [transition[1] for i, transition in enumerate(self.buffer) if i in minibatch_indices]
        minibatch_rewards = [transition[2] for i, transition in enumerate(self.buffer) if i in minibatch_indices]
        minibatch_next_st = [transition[3] for i, transition in enumerate(self.buffer) if i in minibatch_indices]

        #everything needs to be a tensor
        minibatch_tensor_s = torch.tensor(minibatch_states).float()
        minibatch_tensor_a = torch.tensor(minibatch_actions).float()
        minibatch_tensor_r = torch.tensor(minibatch_rewards).float()
        minibatch_tensor_ns = torch.tensor(minibatch_next_st).float()
        
        return (minibatch_tensor_s, minibatch_tensor_a, minibatch_tensor_r, minibatch_tensor_ns)





















