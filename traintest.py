import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import time
import visdom
import config
from util import log, getdev, accuracy

PATH = './models'

test_metrics = [
    nn.CrossEntropyLoss(reduction = 'sum'),
    accuracy(reduction = 'sum')]

def train(train_model, dataloader, n_epochs, lr=None, criterion = None, opt = None, val_loader = None, dev = None, log_stats = False, save_checkpoints = False):
    if dev is None:
        dev = getdev()
    vis = config.vis

    if criterion is None:
        criterion = nn.CrossEntropyLoss(reduction = 'sum')

    if lr is None:
        lr = 0.002

    if opt is None:
        opt = optim.Adam

    log('Starting training', vis, win='training')
    start = time.time()
    loss_history = []
    optimizer = opt(train_model.parameters(), lr=lr, weight_decay = 0.001)

    for epoch in range(n_epochs):
        loss_train_running_total = 0
        count_train = 0
        for batch_num, (batch_in, batch_target) in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            batch_in, batch_target = batch_in.to(dev), batch_target.to(dev)

            # gradient step
            optimizer.zero_grad()
            output = train_model(batch_in)
            loss = criterion(output, batch_target)
            loss.backward()        
            optimizer.step()

            # keep track of loss for display (not used for backprop)
            loss_train_running_total += loss
            count_train += np.prod(batch_target.shape)

            if save_checkpoints:
                torch.save(train_model.state_dict(), PATH+f'/epoch{epoch+1}.pth')

            # print statistics
            if log_stats and batch_num % 50 == 0:
                loss_train = (loss_train_running_total/count_train).item()
                log(f'Epoch {epoch+1:3d}, Batch {batch_num+1:5d}, time: {time.time() - start:.2f} - loss: {loss_train:.3f}', vis, win='training')
                loss_history.append(loss_train)
                vis.line(X=np.array([epoch]), Y=np.array([loss_train]), win='loss', name='train_loss', update = 'append')
                # we reset here since this is just for display
                loss_train_running_total = 0
                count_train = 0

        # do validation if available
        if val_loader is not None:
            loss_val = evaluate(train_model, val_loader, test_metrics)
            loss_val = loss_val.flatten()
            vis.line(X=np.array([epoch]), Y=np.array([loss_val[0]]), win='loss', name='val_loss', update = 'append')
            vis.line(X=np.array([epoch]), Y=np.array([loss_val[1]]), win='acc', name='val_acc', update = 'append')
            
            log(f'Epoch {epoch+1:3d}, Batch {batch_num+1:5d}, time: {time.time() - start:.2f} - val loss: {loss_val[0]:.3f}, val acc: {loss_val[1]:.3f}', vis, win='training')

    return loss_history

# Testing functions

def evaluate(evaluation_models, dataloader, metrics, dev = None):
    if dev is None:
        dev = getdev()
    if not isinstance(evaluation_models, list):
        evaluation_models = [evaluation_models]
    for m in evaluation_models:
        m.eval()

    loss_eval = np.stack([np.zeros_like(metrics, dtype = np.float) for m in evaluation_models])
    count_eval = np.stack([np.zeros_like(metrics, dtype = np.float) for m in evaluation_models])

    with torch.no_grad():
        # loop through batches first, since I think it takes longer to create batches
        for batch_num, (batch_in, batch_target) in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            batch_in, batch_target = batch_in.to(dev), batch_target.to(dev)

            for model_num, model in enumerate(evaluation_models, 0):
                output = model(batch_in)

                for i, m in enumerate(metrics):
                    loss_eval[model_num, i] += m(output, batch_target)
                    count_eval[model_num, i] += np.prod(batch_target.shape)

    loss_eval = loss_eval/count_eval
    return loss_eval

def test(testing_models, test_loader, dev = None):
    if dev is None:
        dev = getdev()
    vis = config.vis

    if not isinstance(testing_models, list):
        evaluation_models = [testing_models]
    
    loss_test = evaluate(evaluation_models, test_loader, test_metrics)

    for model_loss in loss_test:
        # print("Test set results:",
        #       *[repr(m) + ": {:.6f} ".format(x.item()) for x, m in zip(model_loss, test_metrics)])
        log('Test set results: ' + ' '.join([repr(m) + ": {:.6f} ".format(x.item()) for x, m in zip(model_loss, test_metrics)]), vis, win='training')


