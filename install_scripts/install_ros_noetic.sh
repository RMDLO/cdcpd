#!/usr/bin/env bash

sudo -u root sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo -u root apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo -u root apt update
sudo -u root apt install -y ros-noetic-desktop-full
echo 'source /opt/ros/noetic/setup.bash' >> ~/.bashrc
source ~/.bashrc
sudo -u root apt-get install -y python3-catkin-tools librospack-dev libclass-loader-dev libroslz4-dev libactionlib-dev libxmlrpcpp-dev librosconsole-dev
sudo -u root apt install -y python3-pip
pip3 install --user git+https://github.com/catkin/catkin_tools.git

