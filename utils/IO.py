import json
import pickle
import torch
import shutil

# IO
def loadFromJson(filename):
    f = open(filename, 'r', encoding='utf-8')
    data = json.load(f, strict=False)
    f.close()
    return data

def saveToJson(filename, data):
    f = open(filename, 'w', encoding='utf-8')
    json.dump(data, f, indent=4)
    f.close()
    return True

def saveToPKL(filename, data):
    with open(filename, 'wb')as f:
        pickle.dump(data, f)
    return

def loadFromPKL(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def writeFile(filename, massage):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(massage)
    return True

def save_check_point(state, is_best, path='.model', fileName='latest.pth.tar', dataset='darts'):
    torch.save(state, path + '/' + fileName)
    if is_best:
        shutil.copyfile(path + '/' + fileName, path + '/' + dataset + '_model_best.pth.tar')
        shutil.copyfile(path + '/' + fileName, path + '/' + dataset + '_model_best_epoch_' + str(state['epoch']) + '.pth.tar')