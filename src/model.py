import torch.nn as nn
import copy
from transformers import AutoImageProcessor, AutoModelForImageClassification
from peft import LoraConfig, get_peft_model

class LivenessModel(nn.Module):
    def __init__(self, args):
        super(LivenessModel, self).__init__()

        self.basemodel = AutoModelForImageClassification.from_pretrained(args.pretrained)
        self.basemodel.config.num_labels = args.num_classes
        self.basemodel.classifier = nn.Linear(args.projection_dim, args.num_classes, bias=True)

        if args.lora:
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.1,
                target_modules=["query", "value", "dense"],
                bias="none"
            )
            self.basemodel = get_peft_model(self.basemodel, lora_config)
            for param in self.basemodel.base_model.classifier.parameters():
                param.requires_grad = True
            self.basemodel.print_trainable_parameters()
        else:
            for param in self.basemodel.parameters():
                param.requires_grad = False
            for param in self.basemodel.classifier.parameters():
                param.requires_grad = True

    def forward(self,inputs):
        outputs = self.basemodel(inputs)
        logits = outputs.logits
        return logits
    
class Processor(nn.Module):
    def __init__(self,modelname:str):

        super(Processor,self).__init__()
        self.modelname = modelname
        self.process = AutoImageProcessor.from_pretrained(modelname)

    def forward(self,inputs):
        return self.process(inputs,return_tensors="pt")