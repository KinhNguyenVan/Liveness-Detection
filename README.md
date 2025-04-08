# Liveness-Detection
## Project Structure

### Directory Descriptions:
* **`checkpoint/`**: Contains the trained checkpoint files of the models. Users can place pre-trained checkpoints here to run inference or continue training. The checkpoint files include:
    * `model_dino_lora.pt`: Checkpoint for the Dino model with LORA technique.
    * `model_dino_transfer.pt`: Checkpoint for the Dino model trained with transfer learning.
    * `model_vit_lora.pt`: Checkpoint for the Vision Transformer (ViT) model with LORA technique.
    * `model_vit_transfer.pt`: Checkpoint for the ViT model trained with transfer learning.

* **`dataset/`**: Contains the data for training and validation processes.
    * **`dev/`**: Directory containing validation data.
        * **`normal/`**: Contains real images used for evaluation.
        * **`spoof/`**: Contains spoofed images used for evaluation.
    * **`train/`**: Directory containing training data.
        * **`normal/`**: Contains real images used for training.
        * **`spoof/`**: Contains spoofed images used for training.

* **`src/`**: Contains the source code of the project.
    * `__init__.py`: Initializes the Python package.
    * `data_analize.ipynb`: Notebook for data analysis.
    * `dataloader.py`: Code for loading and processing data.
    * `demo.ipynb`: Notebook for running demos or experiments.
    * `model.py`: Defines the model architectures.
    * `train.py`: Script for training the models.

* **`venv/`**: (Ignored) Contains the Python virtual environment with necessary dependencies. Users should create their own virtual environment and install the requirements from the `requirements.txt` file.

* **`.gitignore`**: File specifying files and directories that Git should ignore.

* **`requirements.txt`**: Lists the necessary Python libraries to run the project. Users can install them using the command `pip install -r requirements.txt`.

* **`README.md`**: This file, providing an overview and usage instructions for the project.