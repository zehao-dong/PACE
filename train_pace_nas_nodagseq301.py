import gc
import time
import torch
import numpy as np
#from parallel import DataParallelModel, DataParallelCriterion
from mylog import mylog
from parameterLoader import argLoader
from data_process import Dataset
from models.graphEncoder_pace_nodagseq301 import PairWiseLearning
from models import KLDivLoss
from utils.utils_pace import prepare_train as prepare_func
from utils import save_check_point
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

LOG = mylog(reset=False)

def train(config):
    # Model
    net = PairWiseLearning(config)
    lossFunc = KLDivLoss(config)
    device = torch.device("cuda:{}".format(config.device))
    #if torch.cuda.is_available():
    print(config.device)
    print(config.save_path)
    net = net.cuda(config.device)
    lossFunc = lossFunc.cuda(config.device)
    #net = net.to(device)
    #lossFunc = lossFunc.to(device)

        #if config.parallel:
        #    net = DataParallelModel(net)
        #    lossFunc = DataParallelCriterion(lossFunc)

    # Data
    trainSet = Dataset("train", config, LOG, prepare_func)
    validSet = Dataset("valid", config, LOG, prepare_func)
    print(len(trainSet), len(validSet))

    # Learning Parameters
    num_batches_per_epoch = len(trainSet)
    learning_rate = config.learning_rate

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {"params": [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=config.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=num_batches_per_epoch * config.max_epoch
    )
    optimizer.zero_grad()

    ticks = 0
    Q = []
    best_vloss = 1e99

    LOG.log("There are %d batches per epoch" % (len(trainSet)))
    comp_time = 0
    for epoch_idx in range(config.max_epoch):
        trainSet.batchShuffle()
        LOG.log("Batch Shuffled")
        time_start=time.time()
        for batch_idx, batch_data in enumerate(trainSet):
            net.train()
            # release memory
            if (ticks + 1) % 1000 == 0:
                gc.collect()

            start_time = time.time()
            ticks += 1
            # adjX, adjY are added
            X, maskX, maskX_, Y, maskY, maskY_, maskXY, labels, adjX, adjY, clabelX, clabelY= batch_data
            #print(adjX.shape)
            X, maskX, maskX_, Y, maskY, maskY_, maskXY, labels, adjX, adjY, clabelX, clabelY = X.to(device), maskX.to(device), maskX_.to(device), Y.to(device), maskY.to(device), maskY_.to(device), maskXY.to(device), labels.to(device), adjX.to(device), adjY.to(device), clabelX.to(device), clabelY.to(device)
            logits = net(X, maskX, maskX_, Y, maskY, maskY_, maskXY, adjX, adjY, clabelX, clabelY)
            n_token = int((labels.data != config.PAD).data.sum())
            loss = lossFunc(logits, labels, n_token).sum()

            Q.append(float(loss))
            if len(Q) > 200:
                Q.pop(0)
            loss_avg = sum(Q) / len(Q)

            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            LOG.log('Epoch %2d, Batch %6d, Loss %9.6f, Average Loss %9.6f, Time %9.6f' %
                    (epoch_idx + 1, batch_idx + 1, loss, loss_avg, time.time() - start_time))

            loss = None

            if (ticks >= config.check_point_min) and (ticks % config.check_point_freq == 0):
                gc.collect()
                vloss = 0
                total_tokens = 0
                with torch.no_grad():
                    net.eval()
                    for bid, batch_data in enumerate(validSet):
                        X, maskX, maskX_, Y, maskY, maskY_, maskXY, labels, adjX, adjY, clabelX, clabelY= batch_data
                        X, maskX, maskX_, Y, maskY, maskY_, maskXY, labels, adjX, adjY, clabelX, clabelY = X.to(device), maskX.to(device), maskX_.to(device), Y.to(device), maskY.to(device), maskY_.to(device), maskXY.to(device), labels.to(device), adjX.to(device), adjY.to(device), clabelX.to(device), clabelY.to(device)
                        logits = net(X, maskX, maskX_, Y, maskY, maskY_, maskXY, adjX, adjY, clabelX, clabelY)
                        nv_token = int((labels.data != config.PAD).data.sum())
                        total_tokens += nv_token
                        vloss += float(lossFunc(logits, labels, n_token).sum())

                vloss /= total_tokens
                is_best = vloss < best_vloss
                best_vloss = min(vloss, best_vloss)
                LOG.log('CheckPoint: Validation Loss %11.8f, Best Loss %11.8f' % (vloss, best_vloss))
                vloss = None

                if is_best:
                    LOG.log('Best Model Updated')
                    save_check_point({
                        'epoch': epoch_idx + 1,
                        'batch': batch_idx + 1,
                        'config': config,
                        'state_dict': net.state_dict(),
                        'best_vloss': best_vloss},
                        is_best,
                        path=config.save_path,
                        fileName=config.dataset + '_latest.pth.tar',
                        dataset=config.dataset
                    )
        time_end=time.time()
        epoch_time = time_end - time_start
        comp_time += epoch_time

        if config.save_each_epoch:
            LOG.log('Saving Model after %d-th Epoch.' % (epoch_idx + 1))
            save_check_point({
                'epoch': epoch_idx + 1,
                'batch': batch_idx + 1,
                'config': config,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_vloss': 1e99},
                False,
                path=config.save_path,
                fileName=config.dataset + '_checkpoint_Epoch' + str(epoch_idx + 1) + '.pth.tar',
                dataset=config.dataset
            )
            
        LOG.log('Epoch Finished.')
        gc.collect()
    print('total time:', comp_time)

if __name__ == '__main__':
    args = argLoader()
    print("Totally", torch.cuda.device_count(), "GPUs are available.")
    
    torch.cuda.set_device(args.device)
    print("Using #", args.device, "named", torch.cuda.get_device_name(args.device), (
                torch.cuda.get_device_properties(args.device).total_memory - torch.cuda.memory_allocated(
            args.device)) // 1000 // 1000 / 1000, "GB Memory available.")

    if args.do_train:
        train(args)