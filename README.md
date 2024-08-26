Handling the Cannikin Law: Accelerating Federated Learning by Coordinating Pacesetters and Stragglers
=
This repository contains the implementation of Join Hands federated learning and the code for reproducing the experiments in the paper. "Handling the Cannikin Law: Accelerating Federated Learning by Coordinating Pacesetters and Stragglers"

Test-bed
---
ITX−3588J (OS: 64 − bit Ubuntu 23.10, CPU: 8 core 64Bit, NPU) as the server, 3 Raspberry Pi5 (OS: 64 − bit Ubuntu 23.10, CPU: 4 Cortex-A76, GPU: 800 MHz VideoCore 7), 3 Raspberry Pi4 (OS: 64 − bit Ubuntu 23.10, CPU: 4 Cortex-A72, GPU: 600 MHz VideoCore 6), 1 NVIDIA Jetson Nano (OS: 64 − bit Ubuntu 23.10, CPU: 4 Cortex-A57, GPU: 128 core NVIDIA Maxwell) and 1 NVIDIA Jetson TX−2 (OS:64 − bit Ubuntu 23.10, CPU: 2 Cortex-A57 and 2 Denver2 64bit CPU, GPU: 256 NVIDIA CUDA core) as clients. The above devices construct a LAN through WIFI hotspots. 


Dependencies
------
**Standalone Cluster:** Apache Spark and Hadoop

**python**: hdfs,pytorch,sorket,numpy,pyspark

Usage
----
**Server:** python -master.py

**Client:** python -slave.py


