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

* **Dino LORA Checkpoint**:[checkpoint/model_dino_lora.pt](https://drive.google.com/file/d/11oNMwr4f3dsqzD-a3f0ETFwPafX6F-TZ/view?usp=drive_link)
* **Dino Transfer Learning Checkpoint**: [checkpoint/model_dino_transfer.pt](https://drive.google.com/file/d/1vOYgg_Ur-fkMNmvQdIQMpoMczUKwKI4m/view?usp=drive_link)
* **ViT LORA Checkpoint**: [checkpoint/model_vit_lora.pt](https://drive.google.com/file/d/1d-9I8LHrTSR7Qk36MusQqA-ImBSUeKsm/view?usp=drive_link)
* **ViT Transfer Learning Checkpoint**: [checkpoint/model_vit_transfer.pt](https://drive.google.com/file/d/1pRe3p54RE6IpLdQLbFsI6WltDLHpniWL/view?usp=drive_link)

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

 ### 5.Inference
```import torch
from model import LivenessModel, preprocessor
from inference import inference
import os

img_path = "D:/CAKE/dataset/dev/spoof/63_3.jpg"
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Determine device (GPU if available)
checkpoint = torch.load("D:/CAKE/checkpoint1/model_dino_lora.pt",
                        weights_only=False, # Load only the weights
                        map_location=torch.device(device))  # Load checkpoint weights
model = LivenessModel(checkpoint['args'])  # Initialize model on device

processor = preprocessor(model.args)  # Initialize processor with arguments

model.load_state_dict(checkpoint['model_state_dict'])  # Load weights into model

print("Inference: ", inference(model, processor,img_path, device))  # Perform inference on the image
```


### 6. Training 

```from train import train , evaluate_model
import torch
from model import LivenessModel
from dataloader import CustomImageDataset   
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = 'YOUR_DATASET_DIR'  # **Change this to your actual dataset path**  
train_dir = os.path.join(data_dir, "train")
dev_dir = os.path.join(data_dir, "dev")

args = set_args()
args.pretrained = "google/vit-base-patch16-224" # or "facebook/dino-vitbase16-224",...
args.projection_dim = 768 # depending on the pretrained model you choose
args.lora = True # or False, depending on your needs
args.lora_alpha = 32 # or any other value you prefer
args.lora_r = 64 # or any other value you prefer

model = LivenessModel(args).to(device)
train_dataset = CustomImageDataset(train_dir,modelname=args.pretrained)
dev_dataset = CustomImageDataset(dev_dir,modelname=args.pretrained,dev=True)

train(model, device=device, trainset=train_dataset, devset=dev_dataset, args=args)  # Train the model
evaluate_model(model, device=device, testset=dev_dataset, args=args)  # Evaluate the model

output_dir = "YOUR_OUTPUT_DIR"  # Specify your output directory
os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
torch.save({
    'args': args,
    'model_state_dict': model.state_dict()
}, os.path.join(output_dir, "model.pt"))  # Save the model state dictionary
```
