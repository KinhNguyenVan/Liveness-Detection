import torch
from PIL import Image
import argparse

def set_args():
    """
    Sets the script's arguments using argparse.

    Instead of parsing command-line arguments,
    we'll manually create an argument namespace with
    the default values. This simulates what would happen
    if the script was run from the command line with no
    additional arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train_epochs', default=7, type=int, help='number of train epoched')
    parser.add_argument('--model', default='LivenessModel', type=str, help='model name')
    parser.add_argument('--output_dir', default='/output/', type=str, help='output directory')
    parser.add_argument('--train_batch_size', default=32, type=int, help='batch size in train phase')
    parser.add_argument('--dev_batch_size', default=32, type=int, help='batch size in dev phase')
    parser.add_argument('--projection_dim', default=768, type=int, help='classifier layer hidden size')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--optimizer_name", type=str, default='adam',help="use which optimizer to train the model.")
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='dropout rate')
    parser.add_argument('--learning_rate', default=2e-4, type=float, help='learning rate')
    parser.add_argument('--num_classes',default=2,type=int,help='number of classes')
    parser.add_argument('--weight_decay',default=0.05,type=float,help='regularization')
    parser.add_argument('--lora',default=True,type=bool,help='train lora')
    parser.add_argument('--lora_alpha',default=32,type=int,help='lora alpha')
    parser.add_argument('--lora_r',default=64,type=int,help='lora rank')
    parser.add_argument('--pretrained',default=None,type=int,help='load pretrained model')

    args = parser.parse_args(args=[])
    return args

def inference(model, processor, image_path, device):
    """
    Perform inference on a single image.

    Args:
        model: The model to use for inference.
        processor: The processor to preprocess the image.
        image_path: Path to the image file.
        device: Device to perform inference on (e.g., 'cuda' or 'cpu').

    Returns:
        The predicted class label and the corresponding score.
    """
    model.eval()
    model.to(device)
    idx2label = {0: "spoof", 1: "normal"}
    processor.to(device)

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image).to(device)
    inputs = inputs['pixel_values']
    inputs = inputs.squeeze(1)
    # Perform inference
    with torch.no_grad():
        outputs = model(inputs)
        scores = torch.nn.functional.softmax(outputs, dim=-1)

    # Get the predicted class and score
    predicted_class = torch.argmax(scores, dim=-1).item()
    predicted_class = idx2label[predicted_class]

    return predicted_class