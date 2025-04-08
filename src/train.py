from torch.utils.data import DataLoader
import torch
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR




def train(model,trainset,args,criterion,device):
    model.train()
    model.to(device)
    train_loader = DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True) 
    if args.optimizer_name == 'adam':
        from torch.optim import AdamW
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon,weight_decay=args.weight_decay,)
        print("Used Adamw optimizer")
    elif args.optimizer_name == 'sgd':
        from torch.optim import SGD
        optimizer = SGD(model.parameters(), lr=args.learning_rate,weight_decay=args.weight_decay,)
        print("Used SGD optimizer")

    T_max = args.num_train_epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=T_max)

    for epoch in range(args.num_train_epochs):
        sum_loss=0
        sum_step=0
        predict=None
        target=None

        for inputs,labels in train_loader:
            labels = torch.tensor(labels).to(device)
            inputs = inputs.to(device)
            inputs = inputs.squeeze(1)
            
            outputs = model(inputs)
            losses = criterion(outputs,labels)
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            scheduler.step()

            score = torch.nn.functional.softmax(outputs,dim=-1)
            sum_loss+=losses.item()
            sum_step+=1
            predictions=torch.argmax(score,dim=-1)
            
            if predict is None:
                predict=predictions
                target=labels
            else:
                predict=torch.cat((predict,predictions),dim=0)
                target=torch.cat((target,labels),dim=0)
        print("epoch: ",epoch,"loss: ",sum_loss/sum_step)
        print("f1_score: ",f1_score(target.cpu().numpy(),predict.cpu().numpy(),average="macro"))
        print("recall" ,recall_score(target.cpu().numpy(), predict.cpu().numpy(), average='macro'))
        print("precision: ",precision_score(target.cpu().numpy(), predict.cpu().numpy(), average='macro'))


def evaluate_model(model, device, testset, args):
    model.eval()
    model.to(device)
    test_loader = DataLoader(testset, batch_size=args.dev_batch_size, shuffle=False)
    all_preds = []
    all_true_labels = []
    idx2label = {0: "spoof", 1: "normal"}
    labels = [idx2label[i] for i in sorted(idx2label.keys())]  

    with torch.no_grad():
        for images, labels_batch in test_loader:
            images, labels_batch = images.to(device), labels_batch.to(device)
            images = images.squeeze(1)
            outputs = model(images)

            preds = torch.argmax(torch.nn.functional.softmax(outputs, dim=-1), dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_true_labels.extend(labels_batch.cpu().numpy())

    accuracy = accuracy_score(all_true_labels, all_preds)
    f1_macro = f1_score(all_true_labels, all_preds, average='macro')
    recall = recall_score(all_true_labels, all_preds, average='macro')
    precision = precision_score(all_true_labels, all_preds, average='macro')
    cm = confusion_matrix(all_true_labels, all_preds)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    print(f"Precision (Macro): {precision:.4f}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)  
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
    

