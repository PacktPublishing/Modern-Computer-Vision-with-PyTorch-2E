# Install CUDA drivers
Following instructions assume you have a CUDA compatible GPU with at least 8GB VRAM (GTX1070 or better) as part of hardware. 

## 1. Ubuntu 22.04
The installation of PyTorch GPU in Ubuntu 22.04 can be summarized in the following points,
•   Install CUDA by installing nvidia-cuda-toolkit.  
•   Install the cuDNN version compatible with CUDA.  
•   Export CUDA environment variables.  

### 1.1 Installing CUDA
First open a terminal and run 
```bash
$ sudo apt install nvidia-cuda-toolkit
```

which directly installs the latest version of CUDA in Ubuntu. After installing CUDA, run 
```bash
$ nvcc -V
```
You will get an output similar to the following to verify if you had a successful installation,
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Thu_Nov_18_09:45:30_PST_2021
Cuda compilation tools, release 11.5, V11.5.119
Build cuda_11.5.r11.5/compiler.30672275_0
```
Note the CUDA version above `release 11.5`

### 1.2 Installing CUDNN
After above step, visit - https://developer.nvidia.com/rdp/cudnn-download - and download the CUDNN package that matches your CUDA version which is highlighted above. Once downloaded run

```
wget https://developer.download.nvidia.com/compute/cudnn/9.0.0/local_installers/cudnn-local-repo-ubuntu2204-9.0.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.0.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.0.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn
```

### 1.3 Reboot your system
```
sudo reboot now
```

### 1.4 Export CUDA environment variables

The CUDA environment variables are needed by PyTorch for GPU support. To set them, we need to append them to `~/.bashrc` file by modifying the file's last two lines as follows,
```bash
export PATH=/usr/local/cuda/bin{PATH:+:{PATH}}  
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Load the exported environment variables by running,
```bash
$ source ~/.bashrc
```

Finally we can check if everything went fine by running 
```bash
$ nvidia-smi
```
<img src='https://i.imgur.com/KKsdGf2.png' alt='image' style='max-height: 400px; width=auto;'>


# 2. Windows
One can follow the official instructions found at - https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/# - to install CUDA on windows. A summarized version of the same has been given below for the readers.

## 2.0 Installation via CONDA
The easisest way to get started with cuda on Windows is by having a working anaconda environment and running
```
conda install cuda -c nvidia
```
This will perform a basic install of all CUDA Toolkit components using Conda.

However the downside of this method is that CUDA is usable only in that single environment and the step needs to be repeated whenever a new environment is needed.
To alleviate that, one can follow the following instructions to install at a system level.

The setup to install CUDA development tools on a system level, running the appropriate version of Windows consists of a few simple steps:
- Verify the system has a CUDA-capable GPU.
- Download the NVIDIA CUDA Toolkit.
- Install the NVIDIA CUDA Toolkit.
- Test that the installed software runs correctly and communicates with the hardware.

## 2.1. Verify You Have a CUDA-Capable GPU

You can verify that you have a CUDA-capable GPU through the **Display Adapters** section in the **Windows Device Manager**. Here you will find the vendor name and model of your graphics card(s). If you have an NVIDIA card that is listed in [https://developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus), that GPU is CUDA-capable. The Release Notes for the CUDA Toolkit also contain a list of supported products.

The **Windows Device Manager** can be opened via the following steps:
1. Open a run window from the Start Menu
2. Run:
    control /name Microsoft.DeviceManager

## 2.2. Download the NVIDIA CUDA Toolkit

The NVIDIA CUDA Toolkit is available at [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads). Choose the platform you are using and one of the following installer formats:
1. Network Installer: A minimal installer which later downloads packages required for installation. Only the packages selected during the selection phase of the installer are downloaded. This installer is useful for users who want to minimize download time.
2. Full Installer: An installer which contains all the components of the CUDA Toolkit and does not require any further download. This installer is useful for systems which lack network access and for enterprise deployment.
It is recommened to use Full Installer.



## 2.3. Install the CUDA Software
Install the CUDA Software by executing the CUDA installer and following the on-screen prompts.

## 2.4 Test that CUDA is working by running the following commands
Finally you can check CUDA is installed by running
```
nvcc -V
```
and
```
nvidia-smi
```
