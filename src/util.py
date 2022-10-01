import torch
import numpy as np
import matplotlib.pyplot as plt

class accuracy:
    """Returns count of correct classes.
    
    output: predicted probability of each class
    target: ground truth class label (index value of the correct class)
    """
    
    def __init__(self, reduction = None):
        super().__init__()
        self.reduction = reduction


    def __repr__(self):
        return 'Accuracy'

    def __str__(self):
        return 'Accuracy'
    
    def __call__(self, output, target_classes):
        _, predicted_class = torch.max(output.data, -1)
        if (target_classes.shape != predicted_class.shape):
            print('Warning: predicted_class shape does not match target_class shape')
            print('predicted_class shape:', predicted_class.shape, ' target_classes shape:', target_classes.shape)
        if (self.reduction == 'sum'):
            return (predicted_class == target_classes).sum()
        else:
            return (predicted_class == target_classes).sum() / torch.numel(target_classes)

class binary_accuracy:
    """Returns count of correct classes.
    
    output: logit (pre-sigmoid output) of each class
    target: ground truth class label (index value of the correct class)
    """
    def __init__(self, reduction = None):
        super().__init__()
        self.reduction = reduction

    def __repr__(self):
        return 'Binary Accuracy'

    def __str__(self):
        return 'Binary Accuracy'
    
    def __call__(self, output, target_classes):    
        predicted_class = torch.sigmoid(output.data)>0.5
        if (self.reduction == 'sum'):
            return (predicted_class == target_classes).sum()
        else:
            return (predicted_class == target_classes).sum() / torch.numel(target_classes)



def log(message, vis = None, win = None):
    if vis is not None and win is not None:
        if not vis.win_exists(win):
            vis.text('Starting', win=win)
        vis.text(str(message), win=win, append=True)
    else:
        print(str(message))


def getdev():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    return dev

# from https://github.com/noagarcia/visdom-tutorial
class LinePlot(object):
    """Plots to Visdom"""
    def __init__(self, vis, env_name='main'):
        self.vis = vis
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.vis.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.vis.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

def imshow(img, labels, vis):
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
#    plt.show()
    vis.matplot(plt, win='img')
    log(' '.join('%5s' % classes[labels[j]] for j in range(6)), vis)

def load(model, name):
    if name is not None:
        log(f'Loading model {name}')
        model.load_state_dict(torch.load(name))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)