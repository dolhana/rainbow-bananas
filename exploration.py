#%% Imports
from unityagents import UnityEnvironment

#%% Creating a banana collector environment
env = UnityEnvironment('Banana_Windows_x86_64/Banana.exe')

#%% A Unity environment

#%% Cleaning up
env.close()
