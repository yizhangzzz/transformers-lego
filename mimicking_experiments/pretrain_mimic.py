import torch
import torch.nn as nn
import numpy as np
import time
from transformers import BertTokenizer, BertModel, AlbertModel, AlbertConfig, BertConfig
import argparse
import torch.nn.functional as F
from lego_data import generate_mimicking_data

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', default=0, type=int, help='1: albert, 0: bert')
args = parser.parse_args()


#tokenizer = AlbertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

batch_size = 200

#trainloader, testloader = make_lego_datasets(tokenizer, n_var=12, n_train=12000, n_test=1200, batch_size=batch_size)




def pre_train():
    total_conv_loss = 0
    total_asso_loss = 0
    total = 0
    model.train()
    for _ in range(100):
        batch, asso_maps, target_conv_attn = generate_mimicking_data(tokenizer, n_var=12, batch_size=batch_size)
        x = batch.cuda()
        target_asso_maps = asso_maps.cuda()
        
        optimizer.zero_grad()
        h = model(input_ids=x, output_attentions=True)
        
        conv_loss = 0
        asso_loss = 0
        
        for attn in h.attentions:
            map0 = (attn[:,0,:,:] + 1e-3) / (attn[:,0,:,:] + 1e-3).sum(-1, keepdim=True)
            map1 = (attn[:,1,:,:] + 1e-3) / (attn[:,1,:,:] + 1e-3).sum(-1, keepdim=True)
            conv_loss += (map0 * torch.log(map0 / target_conv_attn)).sum(-1).mean()
            asso_loss += (map1 * torch.log(map1 /target_asso_maps)).sum(-1).mean()

        
        total += 1
        loss = conv_loss + asso_loss
        loss.backward()
        optimizer.step()
        total_conv_loss += conv_loss.item()
        total_asso_loss += asso_loss.item()
    print("   Train Conv Loss: %f, Asso Loss: %f" % (total_conv_loss/total, total_asso_loss/total))

if args.model == 0:
    config = BertConfig.from_pretrained("bert-base-uncased")
    model = nn.DataParallel(BertModel(config).cuda())
    n_epoch = 10
elif args.model == 1:
    config = AlbertConfig.from_pretrained("albert-base-v1")
    model = nn.DataParallel(AlbertModel(config).cuda())
    n_epoch = 50
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)
start = time.time()

for epoch in range(n_epoch):
    start = time.time()

    pre_train()
    #pre_test()
    scheduler.step()
    print('Time elapsed: %f s' %(time.time() - start))

if args.model == 0:
    torch.save(model.module.state_dict(), './mimic_bert.pth')
elif args.model == 1:
    torch.save(model.module.state_dict(), './mimic_albert.pth')