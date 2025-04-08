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


## Getting Started - Running the Code

This section guides you through the steps to run the code, including downloading the image datasets, downloading the model checkpoints, and installing the necessary Python libraries.

### 1. Download Image Datasets

The image datasets for training and validation are organized into `normal` (real images) and `spoof` (fake/attack) images` categories within the `dataset/train/` and `dataset/dev/` directories, respectively. You can download the datasets using the following links:

* **Link download**: [Here](https://drive.google.com/file/d/14mQyIzvBOPIXippMtx7tNMxoMejjkkrg/view?usp=drive_link)

* **`dataset/`**: Contains the data for training and validation processes.
    * **`dev/`**: Directory containing validation data.
        * **`normal/`**: Contains real images used for evaluation.
        * **`spoof/`**: Contains spoofed images used for evaluation.
    * **`train/`**: Directory containing training data.
        * **`normal/`**: Contains real images used for training.
        * **`spoof/`**: Contains spoofed images used for training.


Please download these datasets and place the image files in the corresponding directories as shown in the [Directory Structure](#directory-structure) section.

### 2. Download Model Checkpoints

Pre-trained model checkpoints can be downloaded using the links below. These checkpoints can be used for inference, fine-tuning, or as a starting point for training.

* **Dino LORA Checkpoint**:[checkpoint/model_dino_lora.pt](https://drive.google.com/file/d/1awBvFExD4udTpYpjqcLPUOsXaT1XvZjn/view?usp=drive_link)
* **Dino Transfer Learning Checkpoint**: [checkpoint/model_dino_transfer.pt](https://drive.google.com/file/d/1HHRB1orugyQKBYFJuUve3Yp-n-U9F2WZ/view?usp=drive_link)
* **ViT LORA Checkpoint**: [checkpoint/model_vit_lora.pt](https://drive.google.com/file/d/13bt65I7mPeF2fkg8SZnEzK3DbuslsUlx/view?usp=drive_link)
* **ViT Transfer Learning Checkpoint**: [checkpoint/model_vit_transfer.pt](https://drive.google.com/file/d/171xELJC96AgyJ-gXOTyxUGRyQ0fPYjmO/view?usp=drive_link)

After downloading, place these checkpoint files in the `checkpoint/` directory.

### 3. Install Required Libraries

This project relies on several Python libraries. To install all the necessary dependencies, follow these steps:

1.  **Navigate to the project's root directory** in your terminal or command prompt.

    ```bash
    cd /path/to/your/project
    ```

2.  **Create a virtual environment** (recommended) to isolate the project dependencies.

    ```bash
    # For Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required libraries** using the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

    This command will install all the libraries listed in the `requirements.txt` file, including their specific versions if mentioned.

After completing these steps, you should have the necessary data, model checkpoints, and environment set up to run the code in this project. Refer to the specific scripts (e.g., `train.py`, `demo.ipynb`) for instructions on how to execute them.

### 4. Model Parameter Setup

This table provides a summary of the parameters associated with each provided model checkpoint.

| checkpoint                 | pretrained                  | lora  | lora_r | lora_alpha | projection_dim |
|----------------------------|-----------------------------|-------|--------|------------|----------------|
| model_vit_transfer.pt    | google/vit-base-patch16-224 | False | None   | None       | 768            |
| model_vit_lora.pt        | google/vit-base-patch16-224 | True  | 32     | 16         | 768            |
| model_dino_transfer.pt   | facebook/dinov2-base        | False | None   | None       | 1536           |
| model_dino_lora.pt       | facebook/dinov2-base        | True  | 128    | 64         | 768            |

**Explanation of Parameters:**

* **checkpoint:** The name of the model checkpoint file.
* **pretrained:** The name or path of the pretrained backbone model used.
* **lora:** A boolean indicating whether Low-Rank Adaptation (LORA) is applied to the model.
* **lora_r:** The rank (dimensionality) of the LORA matrices. This is only relevant if `lora` is `True`.
* **lora_alpha:** The scaling factor for the LORA matrices. This is only relevant if `lora` is `True`.
* **projection_dim:** The dimensionality of the projection layer in the model.

When running the training or evaluation scripts, you might need to specify these parameters to load the correct model configuration corresponding to the checkpoint you intend to use. Please refer to the script's documentation or command-line arguments for details on how to set these parameters.
