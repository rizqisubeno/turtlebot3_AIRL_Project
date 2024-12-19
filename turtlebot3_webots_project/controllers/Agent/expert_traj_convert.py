import os
import numpy as np
import torch as th

import matplotlib.pyplot as plt
from .library.normalize import NormalizeObservation

# (NEW) try to add ranking

# Specify the directory containing the expert_traj folder
folder_path = "./expert_traj"

#specify the weight of exponential ranking
weight = 0.1

# Iterate through all files in the folder
for file_name in os.listdir(folder_path):
    # Check if the file is a .npy file and ends with '_0.npy' or '_1.npy'
    if file_name.endswith(".npy"):
        file_path = os.path.join(folder_path, file_name)
        
        # Extract the suffix (e.g., '0' or '1') from the file name
        suffix = file_name.split('_')[-1].split('.')[0]
        
        # Load the .npy file
        print(f"{file_path=}")
        data = np.load(file_path, allow_pickle=True)
        
        # Dynamically create a variable named traj_<suffix> and assign the data to it
        # print(data)
        globals()[f"traj_{suffix}"] = data
        num_ep = len(data)
        num_steps_on_ep = data[0]['acts'].shape
        print(f"Loaded {file_name} into traj_{suffix} with size {num_ep} episode and max {num_steps_on_ep} steps")

        ## Patch for error on the code
        for i in range(len(data)):
            if num_steps_on_ep[0] + 1 < data[i]['obs'].shape[0]:
                print(f"fix the error on idx:{i}")
                data[i]['obs'] = data[i]['obs'][1:]

        for i in range(len(data)):
            if data[i]['obs'].shape[0] != num_steps_on_ep[0]+1:
                raise IndexError(f"stop on i:{i} : {data[i]['obs'].shape[0]} != {num_steps_on_ep[0]+1}")

        # convert to torch array
        traj_list = []
        #####################################################################################################
        #adding ranking search based low distance travelled and time completion to goal
        temp_rank = []
        # distance travelled
        
        for i_traj in range(len(data)):
            # searching over the first step loop
            temp_dist = 0.000
            for i in range(len(data[i_traj]['obs'])-1):
                temp_dist += data[i_traj]['obs'][i][-2] - \
                             data[i_traj]['obs'][i+1][-2]

            temp_rank.append([np.round(temp_dist,3), i_traj])

        #time completion to goal
        for i_traj in range(len(data)):
            idx = 0
            # searching over the first step loop
            for target in data[i_traj]['infos']:
                if target['target']==True:
                    break
                else:
                    idx+=1
            # print(f"idx: {idx} of {len(data[i_traj]['infos'])}")
            temp_rank[i_traj] = [temp_rank[i_traj][0]] + [idx] + [temp_rank[i_traj][1]]

        temp_rank = np.asarray(temp_rank)
        # print(f"{temp_rank=}")
        #sort first column and then second column
        ind = np.lexsort((temp_rank[:,1],temp_rank[:,0]))
        temp_rank = temp_rank[ind]
        # print(f"{temp_rank=}")

        # create i_traj based probability rank (exponential)
        max_rank = np.max(temp_rank[:,2])
        min_rank = np.min(temp_rank[:,2])
        # print(f"{max_rank=}")
        # print(f"{min_rank=}")
        # rank = np.arange(max_rank+1 if min_rank==0 else max_rank, min_rank if min_rank==0 else min_rank-1,-1)
        rank = np.arange(min_rank+1 if min_rank==0 else min_rank, max_rank+2 if min_rank==0 else max_rank)
        # rank = rank/np.sum(rank)
        rank = np.exp(-rank*weight)/np.sum(np.exp(-rank*weight))
        # plt.plot(rank)
        # plt.show()
        # print(rank.shape)

        temp_rank = temp_rank.tolist()
        for i_traj in range(len(temp_rank)):
            temp_rank[i_traj] = temp_rank[i_traj][:3] + [rank[i_traj]]
        temp_rank = np.asarray(temp_rank)
        # print(temp_rank)
        # raise ValueError("stop here")
                
        #delete column 1 and column2 
        temp_rank = temp_rank[:,2:]
        # print(f"{temp_rank=}")
        #####################################################################################################

        # for i_traj in range(len(data)):
        #     for i_obs in range(len(data[i_traj]['obs'])-1):
        #         traj_dict['state'].append(data[i_traj]['obs'][i_obs])
        #         traj_dict['action'].append(data[i_traj]['acts'][i_obs])
        #         traj_dict['reward'].append(0)
        #         traj_dict['dones'].append(data[i_traj]['infos'][i_obs]['target'])
        #         traj_dict['next_state'].append(data[i_traj]['obs'][i_obs+1])
        for idx in range(len(temp_rank)):
            rank_idx = temp_rank[idx][1]
            i_traj = int(temp_rank[idx][0])
            traj = {'state':[], 'action':[], 'reward':[], 'dones':[], 'next_state':[]}
            for i_obs in range(len(data[i_traj]['obs'])-1):  
                # print(i_obs) 
                traj['state'].append(data[i_traj]['obs'][i_obs])
                traj['action'].append(data[i_traj]['acts'][i_obs])
                traj['reward'].append([0])
                traj['dones'].append([data[i_traj]['infos'][i_obs]['target']])
                traj['next_state'].append(data[i_traj]['obs'][i_obs+1])

            traj['state'] = th.as_tensor(data=np.array(traj['state']), dtype=th.float32)
            traj['action'] = th.as_tensor(data=np.array(traj['action']), dtype=th.float32)
            # to reduce memory cause reward not recording
            traj['reward'] = th.as_tensor(data=np.array(traj['reward']), dtype=th.uint8) 
            traj['dones'] = th.as_tensor(data=np.array(traj['dones']), dtype=th.bool)
            traj['next_state'] = th.as_tensor(data=np.array(traj['next_state']), dtype=th.float32)

            traj_list.append([traj, {'rank':rank_idx}])
        # traj_list = th.as_tensor(data=np.array(traj_list), dtype=th.float32)

        # print(f"{traj_list[0][0]['action']=}")
        # print(f"{traj_list[0][1]['rank']=}")
        # raise ValueError("stop here")

        # traj_dict['state'] = th.as_tensor(data=np.array(traj_dict['state']), dtype=th.float32)
        # traj_dict['action'] = th.as_tensor(data=np.array(traj_dict['action']), dtype=th.float32)
        # # to reduce memory cause reward not recording
        # traj_dict['reward'] = th.as_tensor(data=np.array(traj_dict['reward']), dtype=th.uint8) 
        # traj_dict['dones'] = th.as_tensor(data=np.array(traj_dict['dones']), dtype=th.bool)
        # traj_dict['next_state'] = th.as_tensor(data=np.array(traj_dict['next_state']), dtype=th.float32)

        # globals()[f"traj_dict_{suffix}"] = traj_dict
        # th.save(traj_dict,
        #         file_path[:-3]+"pth")
        th.save(traj_list,
                file_path[:-3]+"pth")

