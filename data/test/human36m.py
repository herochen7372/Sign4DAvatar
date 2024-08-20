import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(os.path.join(cwd, 'main'))

from data.Human36M.Human36M import Human36M

if __name__ == '__main__':
    from torchvision import transforms
    # trainset3d_loader.append(eval(cfg.trainset_3d[i])(transforms.ToTensor(), "train"))
    
    trainset3d_loader = Human36M(transforms.ToTensor(), "train")
    for data in trainset3d_loader:
        print(data)
        
    

