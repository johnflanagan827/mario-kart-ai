# AI Mario Kart Agent via Reinforcement Learning

This repository hosts an AI project that teaches an agent to play Mario Kart Wii using reinforcement learning techniques. The project utilizes PyTorch and the Double Deep Q Network (Double DQN) algorithm to train a model capable of in-game decision making.

## Dependencies
- [Dolphin Emulator (Fork by Felk)](https://github.com/Felk/dolphin): A modified Dolphin Emulator for AI integration.
- [Dolphin Memory Engine](https://github.com/aldelaro5/Dolphin-memory-engine): A utility for game memory inspection and manipulation.

## Setup Guide

### Requirements
To get started, you will need:
- A legally obtained ISO file of Mario Kart Wii.
- Python 3.x with PyTorch and necessary libraries installed.

### Installation Steps

1. **Obtain Mario Kart Wii ISO:**
   You must have your own copy of Mario Kart Wii, which you can obtain using a tool like CleanRip.

2. **Emulator Setup:**
   Clone the [Dolphin Emulator (Fork by Felk)](https://github.com/Felk/dolphin) repository. This custom emulator allows for AI scripting.

3. **Memory Engine Setup:**
   Set up the [Dolphin Memory Engine](https://github.com/aldelaro5/Dolphin-memory-engine) as per the instructions in its repository.

4. **Python Dependencies:**
   Install PyTorch and any other dependencies:

   ```sh
   pip install torch torchvision torchaudio
   ```

5. **Script Integration:**
   Place `game_interaction_script.py` into the scripting directory of Dolphin's fork to facilitate game-AI interaction.

6. **AI Execution:**
   With Mario Kart Wii running in Dolphin, start `main.py` to initiate the AI training.

### How to Use

```sh
# Ensure Mario Kart Wii is running in Dolphin

# Execute the training script in a separate terminal
python main.py
```

The script will start the AI agent that begins to learn through the Double DQN method.

## License

This project is made available under the MIT License. Refer to the `LICENSE` file for more details.
