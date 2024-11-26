import os
import numpy as np
import torch as th

# Specify the directory containing the expert_traj folder
folder_path = "./expert_traj"
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
        print(f"Loaded {file_name} into traj_{suffix}")

        # convert to torch array
        traj_dict = {}
        traj_dict['state'] = []
        traj_dict['action'] = []
        traj_dict['reward'] = []
        traj_dict['dones'] = []
        traj_dict['next_state'] = []
        for i_traj in range(len(data)):
            for i_obs in range(len(data[i_traj]['obs'])-1):
                traj_dict['state'].append(data[i_traj]['obs'][i_obs])
                traj_dict['action'].append(data[i_traj]['acts'][i_obs])
                traj_dict['reward'].append(0)
                traj_dict['dones'].append(data[i_traj]['infos'][i_obs]['target'])
                traj_dict['next_state'].append(data[i_traj]['obs'][i_obs+1])

        traj_dict['state'] = th.as_tensor(data=np.array(traj_dict['state']), dtype=th.float32)
        traj_dict['action'] = th.as_tensor(data=np.array(traj_dict['action']), dtype=th.float32)
        # to reduce memory cause reward not recording
        traj_dict['reward'] = th.as_tensor(data=np.array(traj_dict['reward']), dtype=th.uint8) 
        traj_dict['dones'] = th.as_tensor(data=np.array(traj_dict['dones']), dtype=th.bool)
        traj_dict['next_state'] = th.as_tensor(data=np.array(traj_dict['next_state']), dtype=th.float32)

        globals()[f"traj_dict_{suffix}"] = traj_dict
        th.save(traj_dict,
                file_path[:-3]+"pth")

