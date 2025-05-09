import glob
import os
import pickle
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from backend.model.Possesion import Possession
from backend.model.utils import split_list_by_v


class PlayDataset(Dataset):
    def __init__(self, root_dir):
        self.filepath_list = glob.glob(os.path.join(root_dir, "**/*.pkl"), recursive=True)

    def __len__(self):
        return len(self.filepath_list)

    def __getitem__(self, idx):
        filepath = self.filepath_list[idx]
        play = None
        if os.path.getsize(filepath) > 0:
            with open(filepath, "rb") as f:
                play = pickle.load(f)
        return play


def collate_batch(batch: list[Possession]):
    # tensor sequence length 11*121*3 #complement T dimensions to 121
    sample_freq = 5  # downsample
    sequence_length = int(24 * 25 / sample_freq + 1)  # 24s sequence length
    list_24s = [
        i * (1 / sample_freq) for i in range(0, sequence_length)
    ]  # sequence_length coressponding to game clock
    list_24s.reverse()  # Reverse, change from 24s to 0s

    # Set the time step to the length of the sequence and initialize an empty NumPy array states_batch to store the state information, which will have the shape [batch_size, time_steps, 4]
    time_steps = sequence_length  # 121
    states_batch = np.array([]).reshape(-1, time_steps, 3)

    # Initialize two empty NumPy arrays, states_padding_batch and states_hidden_BP_batch, to store padding states and hidden states
    states_padding_batch = np.array([]).reshape(-1, time_steps)
    states_hidden_BP_batch = np.array([]).reshape(-1, time_steps)

    num_agents = np.array([])  # Number of agents per poss

    # player_newids = pd.read_csv('players.csv')

    for play in batch:
        play_tensor = None
        # agent_ids = np.array([])
        for agent in play.agents:  # total of 11 agents
            agent_team = agent.teamid

            if agent_team != play.def_teamid:
                single_agent_array = np.array(
                    [[x, y, v] for x, y, v in zip(agent.x, agent.y, agent.v, strict=False)][
                        ::sample_freq
                    ]
                )  # Sampled at sample_freq, i.e. 5 frames/s
                single_agent_tensor = torch.Tensor(
                    torch.tensor(np.array([single_agent_array.transpose()]))
                )
                single_agent_tensor = single_agent_tensor.to(torch.float)

                if play_tensor == None:
                    play_tensor = single_agent_tensor
                else:
                    play_tensor = torch.cat(
                        [play_tensor, single_agent_tensor], dim=0
                    )  # 获得6个agent的tensor shape=[6,3,time_steps]

        play_tensor = play_tensor
        # The dimension of tensor A is 6
        A, D, T = play_tensor.size()
        # If there are less than 11 agents: complement scence_tensor with -1, and agent_ids with -1.
        if A < 6:
            new_dim = torch.full((6 - A, D, T), -1)
            play_tensor = torch.cat([new_dim, play_tensor], dim=0)
            # agent_ids = np.pad(agent_ids, (11 - A, 0), 'constant', constant_values=(-1,))

        # agent_ids_batch = np.append(agent_ids_batch,agent_ids)
        play_tensor = torch.transpose(play_tensor, dim0=1, dim1=2)  # shape[A,T,D]
        time_24s = play.shot_clock_list[::sample_freq]  # Interval between sample_freq values

        # Calculate padding size for header and tail padding
        head_padding_size = 0
        end_padding_size = 121 - (head_padding_size + len(time_24s) + 1)  # endtime～0

        # [11,121,4] padding the temporal T dimension of scence_tensor with zeros
        states_feat = torch.nn.functional.pad(
            play_tensor,
            (0, 0, head_padding_size + 1, end_padding_size, 0, 0),
            mode="constant",
            value=0,
        )
        # states_padding: padding of the position of the tensor
        # bool (11,121) padding
        states_padding = states_feat[:, :, 0]
        states_padding = states_padding == 0

        # mask process #according Velocity masking
        states_hidden_BP = np.ones((len(states_feat), time_steps)).astype(np.bool_)  # True为hidden
        current_step = len(time_24s)  #
        # states_hidden_BP[:,:current_step] = False #Actual trajectory is False

        flag = random.choice(["A", "B"])

        if flag == "A":
            split_time = len(time_24s) * 2 // 3
            states_hidden_BP[:, :split_time] = False  # Actual trajectory is False
        else:
            states_hidden_BP[:, : current_step + 1] = False
            v_tensor = play_tensor[:, :, 2] >= 4  # Obtaining the velocity tensor [A,T,1]
            index_v = split_list_by_v(v_tensor)
            for index, lst in enumerate(index_v):
                if len(lst) >= 2:
                    random_elements = random.sample(lst, 2)
                    for element in random_elements:
                        s, e = element
                        states_hidden_BP[index, s:e] = True
                else:
                    try:
                        s, e = random.choice(lst)
                    except:
                        s, e = 0, 0
                        states_hidden_BP[index, s:e] = True
            states_hidden_BP[:, 1:4] = False  # Prevents the entire paragraph from being hidden.

        # print(states_hidden_BP.shape)

        num_agents = np.append(
            num_agents, len(states_feat)
        )  # numpy array(batch_size,) [6，6，6，6，6，6]

        states_batch = np.concatenate((states_batch, states_feat), axis=0)
        states_padding_batch = np.concatenate((states_padding_batch, states_padding), axis=0)
        states_hidden_BP_batch = np.concatenate((states_hidden_BP_batch, states_hidden_BP), axis=0)

    num_agents_accum = np.cumsum(np.insert(num_agents, 0, 0)).astype(
        np.int64
    )  # Now insert 0 at index=0 and sum cumulatively [0, 11, 22, ....]
    agents_batch_mask = np.ones(
        (num_agents_accum[-1], num_agents_accum[-1])
    )  # Create an all-1 [A*batch,A*batch] matrix

    for i in range(len(num_agents)):
        # Construct (11*batch_size, 11*batch_size) matrix for batch_size chunks all 1 matrix
        agents_batch_mask[
            num_agents_accum[i] : num_agents_accum[i + 1],
            num_agents_accum[i] : num_agents_accum[i + 1],
        ] = 0

    states_batch = torch.FloatTensor(states_batch)
    agents_batch_mask = torch.BoolTensor(agents_batch_mask)
    states_padding_batch = torch.BoolTensor(states_padding_batch)
    states_hidden_BP_batch = torch.BoolTensor(states_hidden_BP_batch)
    agent_ids_batch = torch.empty(1, 1)

    return (
        states_batch,
        agents_batch_mask,
        states_padding_batch,
        states_hidden_BP_batch,
        num_agents_accum,
        agent_ids_batch,
    )
