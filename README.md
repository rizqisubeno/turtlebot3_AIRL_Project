# Turtlebot3 AIRL Project
Simulation TB3 Using Adversarial Inverse Reinforcement Learning on Webots Environment.

Works on the Following Ubuntu Version :
- [x] Ubuntu 20.04
- [x] Ubuntu 22.04
- [x] Ubuntu 24.04


Project Progress : 
- [x] Reinforcement Learning based on Modified [Clean's RL Repo (PPO)](https://github.com/Josh00-Lu/cleanrl/blob/sync/cleanrl/ppo_continuous_action_truncted.py)
- [ ] Reinforcement Learning based on Clean's RL Repo (SAC)
- [ ] Adversarial Inverse Reinforcement Learning (AIRL) based on [Toshikawa Repo](https://github.com/toshikwa/gail-airl-ppo.pytorch)
- [ ] Try With Modified AIRL (Confidence Aware IL) based on [Stanford-ILIAD](https://github.com/Stanford-ILIAD/Confidence-Aware-Imitation-Learning)

## Guide:
## 1. <Strong>Install Webots R2023b Simulation</strong>
- See the releases package from Webots Github, https://github.com/cyberbotics/webots/releases
## 2. <strong>Install Required Package (ROS2, Protobuf, ZMQ, and pip dependencies)</strong>

- First, Installing ROS2 Package. Actually, ROS2 package is optional (because the ros2 package is only used for visualization of robot movement to rviz). The code works in ROS2 Humble and Jazzy. To install ROS2 Humble, see https://docs.ros.org/en/humble/Installation.html. Otherwise, to install ROS2 Jazzy, see https://docs.ros.org/en/jazzy/Installation.html.
  
- Second, Installing Protobuf the base message communication between RL Agent and Turtlebot3 Robot.

  ```bash
  # Protobuf version 3.21.12
  git clone github.com/protocolbuffers/protobuf.git --branch v3.21.12

  # entering git directory
  cd protobuf

  #update dependencies for package build
  git submodule update --init --recursive

  # build process using cmake & make
  mkdir build
  cd build
  cmake .. -DCMAKE_CXX_STANDARD=14
  make -j4
  sudo make install
  ```

- Third, Installing ZMQ library for c++. The communication protocol to communicate RL Agent (Python) to Turtlebot3 Robot (C++).
  ```bash
  #(for Ubuntu)
  sudo apt install libzmq3-dev
  ```

- Finally, Installing python dependencies. The code works in Python 3.10 and above. Please make sure you use virtualenv so that there is no conflict with other packages.
  ```bash
  pip install -r requirements.txt

  #You can add on ~/.bashrc if you bash; if you zsh, you can add on ~/.zshrc
  export WEBOTS_HOME="/usr/local/webots"
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$WEBOTS_HOME/lib/controller

  #Optional
  export ROS_DISTRO={your_ros2_distro_name}
  export ROS_PYTHON_VERSION={you_ros2_python_version}
  ```
  
