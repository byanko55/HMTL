import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np


# Training module for a single dataset
class Trainer():
    def __init__(
        self, 
        data:torch.utils.data.DataLoader, 
        model:torch.nn.Module,
        batch_size:int = 128,
        learning_rate:float = 1e-3,
        train_ratio:float = 0.7,
        val_ratio:float = 0.1,
        cuda:bool = True
    ) -> None:
        cudnn.benchmark = True
        self.cuda = cuda
        
        # load data
        train_batch, eval_batch, test_batch = data.buildbatch(batch_size, train_ratio, val_ratio)

        self.dloader = {
            'train': train_batch,
            'eval': eval_batch,
            'test': test_batch
        }
        
        # build model
        self.md = model
        self.cf = torch.nn.CrossEntropyLoss()
        self.ot = optim.Adam(model.parameters(), lr=learning_rate)

        if cuda:
            self.md = self.md.cuda()
            self.cf = self.cf.cuda()

    """
    Train a model

    # In
      - epochs:
    # Out
      - acc_train:
      - acc_eval: 
    """
    def train(self, epochs:int = 10) -> tuple:
        self.md.train()
        num_train = len(self.dloader['train'].dataset)
        num_eval = len(self.dloader['eval'].dataset)
        
        acc_train = []; acc_eval = []
        
        for epoch in range(epochs):
            correct_all = 0
            loss_all = 0
            
            for _, (x_b, y_b) in enumerate(self.dloader['train']):
                x_b, y_b = self.applycuda(x_b, y_b)

                # Forward
                self.ot.zero_grad()
                y_hat = self.md(x_b)
                
                loss = self.cf(y_hat, y_b)
                loss_all += loss.item()*len(y_b)

                # Inference
                pred = y_hat.argmax(dim=1, keepdim=True)
                correct = pred.eq(y_b.view_as(pred)).sum().item()
                correct_all += correct
                
                # Back-propagation
                loss.backward()
                self.ot.step()
                
            print("epoch: {:3d}, train-loss: {:2.3f} train-accuracy: {:2.3f}".format(
                    epoch+1, 
                    loss_all/num_train, 
                    correct_all/num_train
                ), end=' '
            )

            acc_train.append(correct_all/num_train)

            with torch.no_grad():
                correct_all = 0
                loss_all = 0
                
                for x_b, y_b in self.dloader['eval']:
                    x_b, y_b = self.applycuda(x_b, y_b)

                    # Forward/Inference
                    y_hat = self.md(x_b)
                    loss = self.cf(y_hat, y_b)
                    loss_all += loss.item()*len(y_b)

                    pred = y_hat.argmax(dim=1, keepdim=True)
                    correct = pred.eq(y_b.view_as(pred)).sum().item()
                    correct_all += correct
                
                acc_eval.append(correct_all/num_eval)

                print("eval-loss: {:2.3f} eval-accuracy: {:2.3f}".format(
                        loss_all/num_eval, 
                        correct_all/num_eval
                    )
                )
                
        return acc_train, acc_eval

    """
    Inference the the hypothesis for the test samples.
    You can choose the default test dataloader("self.dloader['test']") or any other input tensor.

    # In
      - x: target dataset
      - label: if 'False', it means true labels are not given
    # Out
      - y_real/y_pred :
    """
    def predict(self, x:torch.utils.data.DataLoader = None, label:bool = True) -> list:
        self.md.eval()
        test_loss = 0
        correct = 0

        data_loader = x

        if data_loader == None :
            data_loader = self.dloader['test']

        num_test = len(data_loader.dataset)

        with torch.no_grad():
            y_real = []; y_pred = []

            if label :
                for x_b, y_b in data_loader:
                    x_b, y_b = self.applycuda(x_b, y_b)
                    
                    y_hat = self.md(x_b)
                    test_loss += self.cf(y_hat, y_b).item()*len(y_b)
                    pred = y_hat.argmax(dim=1, keepdim=True)
                    correct += pred.eq(y_b.view_as(pred)).sum().item()
                    
                    y_real += [i.cpu().numpy() for i in y_b]
                    y_pred += [i.cpu().numpy() for i in pred]
                    
                print('\nTest set: Average loss: {:.3f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
                            test_loss/num_test, 
                            correct, 
                            num_test, 
                            100. * correct / num_test
                        )
                    )

                return np.array(y_real), np.array(y_pred)
            else :
                res_pred = []

                for x_b in data_loader:
                    x_b = self.applycuda(x_b)

                    y_hat = self.md(x_b)
                    pred = y_hat.argmax(dim=1, keepdim=True)
                    y_pred += [i.cpu().numpy() for i in pred]

                return None, np.array(y_pred)

    """
    Transfers a tensor from CPU to GPU when the 'cuda' flag is on

    # In
      - *args: list of original tensors
    # Out
      - list of cuda tensors
    """
    def applycuda(self, *args):
        if self.cuda :
            return [d.cuda() for d in args]
        else:
            return args