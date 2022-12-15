import torch
import torch.nn as nn
import numpy as np
import time
from transformers import BertTokenizer, BertModel, AlbertModel, BertConfig, AlbertConfig
import argparse
import os
import sys
from lego_data import make_lego_datasets
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', default=0, type=int, help='1: albert, 0: bert')
parser.add_argument('--repeat', default=0, type=int, help='repetition num')
args = parser.parse_args()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
n_var, n_train_var = 12, 6
n_train, n_test = n_var*10000, n_var*1000
batch_size = 400

trainloader, testloader = make_lego_datasets(tokenizer, n_var, n_train, n_test, batch_size)



def train(print_acc=False):
    total_loss = 0
    correct = [0]*len(train_var_pred)
    total = 0
    model.train()
    for batch, labels, order in trainloader:
    
        x = batch.cuda()
        y = labels.cuda()
        inv_order = order.permute(0, 2, 1).cuda()
        optimizer.zero_grad()
        pred = model(x)
        
        ordered_pred = torch.bmm(inv_order, pred[:, 1:-1:5, :])

        loss = 0
        for idx in train_var_pred:
            loss += criterion(ordered_pred[:, idx], y[:, idx]) / len(train_var_pred)
            total_loss += loss.item() / len(train_var_pred)
            correct[idx] += (ordered_pred[:, idx].max(1)[1] == y[:, idx]).float().mean().item()
        
        total += 1
    
        loss.backward()
        optimizer.step()
    
    train_acc = [corr/total for corr in correct]
    print("   Train Loss: %f" % (total_loss/total))
    if print_acc:
        for idx in train_var_pred:
            print("     %s: %f" % (idx, train_acc[idx]))
    
    return train_acc



def test():
    test_acc = []
    start = time.time()
    total_loss = 0
    correct = [0]*n_var
    total = 0
    model.eval()
    with torch.no_grad():
        for batch, labels, order in testloader:
            x = batch.cuda()
            y = labels.cuda()
            inv_order = order.permute(0, 2, 1).cuda()
            pred = model(x)
            ordered_pred = torch.bmm(inv_order, pred[:, 1:-1:5, :])
            for idx in test_var_pred:
                loss = criterion(ordered_pred[:, idx], y[:, idx])
                total_loss += loss.item() / len(test_var_pred)
                correct[idx] += (ordered_pred[:, idx].max(1)[1] == y[:, idx]).float().mean().item()           
            total += 1
        test_acc = [corr/total for corr in correct]
        print("   Test  Loss: %f" % (total_loss/total))
        for idx in test_var_pred:
            print("     %s: %f" % (idx, test_acc[idx]))
   
    return test_acc

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, base, d_model, tgt_vocab=1):
        super(Encoder, self).__init__()
        self.base = nn.Sequential(base.embeddings, base.encoder)
        self.classifier = nn.Linear(d_model, tgt_vocab)
        
    def forward(self, x, mask=None):
        h = self.base(x)
        #x = self.norm(x)
        out = self.classifier(h.last_hidden_state)
        return out

    
work_dir = './'
if args.model == 1:  
    config = AlbertConfig.from_pretrained("albert-base-v1")
    albert = AlbertModel(config)
    albert.load_state_dict(torch.load(work_dir + 'mimic_albert.pth'))
    base = albert
elif args.model == 0:
    config = BertConfig.from_pretrained("bert-base-uncased")
    bert = BertModel(config)
    bert.load_state_dict(torch.load(work_dir + '/mimic_bert.pth'))
    base = bert
else:
    print("unknown model!!!!!")
    
print(args)

model = nn.DataParallel(Encoder(base, d_model=768, tgt_vocab=6).cuda())
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
start = time.time()
train_var_pred = [i for i in range(n_var-6)]
test_var_pred = [i for i in range(n_var)]
test_acc = []


for epoch in range(200):
    start = time.time()
    
    
    print('Epoch %d, lr %f' % (epoch, optimizer.param_groups[0]['lr']))
    train(True)
    test_acc.append(test())
    
    scheduler.step()
    
    print('Time elapsed: %f s' %(time.time() - start))
    sys.stdout.flush()