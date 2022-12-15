import torch
import torch.nn as nn
import numpy as np
import os
import math

all_vars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def generate_mimicking_data(tokenizer, n_var=12, batch_size=100):
    n_var = 12
    seq_len = n_var*5+2
    v = torch.ones(1, seq_len).long()
    gauss_filt = torch.FloatTensor([1, 2, 4, 2, 1]).cuda()
    canvas = torch.zeros(66, 66).cuda()
    for i in range(62):
        canvas[i, i:i+5] = gauss_filt[:]
    target_conv_attn = canvas[None, 0:62, 2:64].cuda() + 1e-3
    target_conv_attn /= target_conv_attn.sum(-1, keepdim=True)
    target_conv_attn = target_conv_attn.expand(batch_size,62,62)
    
    batch = []
    asso_attn = []
    for _ in range(batch_size):
        values = np.random.randint(0, 2, (n_var,))
        var_idx = tuple(np.random.permutation(len(all_vars)))
        vars = [all_vars[i] for i in var_idx]

        # generate first sentence
        clauses = []
        clauses.append('%s = val %d , ' % (vars[0], values[0]))

        for i in range(1, n_var):
            modifier = 'val' if values[i] == values[i-1] else 'not'
            clauses.append(' %s = %s %s , ' % (vars[i], modifier, vars[i-1]))
            

        sent = ''
        label = []
        
        clause_idx = tuple(np.random.permutation(n_var))
        sent += ''.join([clauses[idx] for idx in clause_idx])
        label += [values[idx] for idx in clause_idx]
        
        
        order = torch.zeros(1, n_var, n_var)
        for i in range(n_var):
            order[0, i, clause_idx[i]] = 1
         
        tokens = tokenizer(sent, return_tensors='pt')['input_ids']
        asso_map = (v.T @ tokens**2 + (tokens**2).T @ v - 2*tokens.T@tokens == 0).float()
        idx = torch.arange(0, seq_len).long()
        asso_map[idx, idx] = 0
        asso_map += 1e-3
        asso_map /= asso_map.sum(-1, keepdim=True) 

        asso_attn.append(asso_map.unsqueeze(0))
        batch.append(tokens)

    return torch.cat(batch), torch.cat(asso_attn), target_conv_attn

def generate_data(tokenizer, n_var, batch_size=100):
    
    batch = []
    labels = []
    clause_order = []
    for _ in range(batch_size):
        values = np.random.randint(0, 2, (n_var,))
        var_idx = tuple(np.random.permutation(len(all_vars)))
        vars = [all_vars[i] for i in var_idx]

        # generate first sentence
        clauses = []
        clauses.append('%s = val %d , ' % (vars[0], values[0]))

        for i in range(1, n_var):
            modifier = 'val' if values[i] == values[i-1] else 'not'
            clauses.append(' %s = %s %s , ' % (vars[i], modifier, vars[i-1]))
            

        sent = ''
        label = []
        
        clause_idx = tuple(np.random.permutation(n_var))
        sent += ''.join([clauses[idx] for idx in clause_idx])
        label += [values[idx] for idx in clause_idx]
        
        
        order = torch.zeros(1, n_var, n_var)
        for i in range(n_var):
            order[0, i, clause_idx[i]] = 1
            
        batch.append(tokenizer(sent, return_tensors='pt')['input_ids'])
        labels.append(values)
        clause_order.append(order)
    return torch.cat(batch), torch.LongTensor(labels), torch.cat(clause_order)




def make_lego_datasets(tokenizer, n_var, n_train, n_test, batch_size):
    
    train_data = []
    train_labels = []
    train_order = []

    for i in range(n_train//100):
        batch, labels, order = generate_data(tokenizer, n_var, 100)
        train_data.append(batch)
        train_labels.append(labels)
        train_order.append(order)

    x_train = torch.cat(train_data)
    y_train = torch.cat(train_labels)
    order_train = torch.cat(train_order)
    
    trainset = torch.utils.data.TensorDataset(x_train, y_train, order_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    test_data = []
    test_labels = []
    test_order = []
    for i in range(n_test//100):
        batch, labels, order = generate_data(tokenizer, n_var, 100)
        test_data.append(batch)
        test_labels.append(labels)
        test_order.append(order)

    x_test = torch.cat(test_data)
    y_test = torch.cat(test_labels)
    order_test = torch.cat(test_order)

    testset = torch.utils.data.TensorDataset(x_test, y_test, order_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)
    
    return trainloader, testloader