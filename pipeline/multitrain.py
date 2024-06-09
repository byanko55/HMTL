import os
import torch
import torch.utils.data
import torch.optim as optim
import torch.backends.cudnn as cudnn

from model.multimodel import MultiEncoder, IndepClassifier
from dataloader.imgset import ImgLoader
from dataloader.multiset import MultiSet


def inter_class_distance(z, batch_size):
    """ Inter Class Distance Loss """
    sigma = torch.std(z)
    delta = torch.sum(torch.exp(-torch.cdist(z, z, p=2)/sigma))/pow(batch_size, 2)

    loss = delta/(1+delta)
    return loss


# Training module for fusnet
class HeteroTrainer():
    """ Training module for multi-encoder """
    def __init__(self, imgsets, train_opt, cuda=True, **kargs):
        model_path = os.path.join('FEmodels', '_model_epoch_' + str(train_opt['epoch']) + '.pth')
        cudnn.benchmark = True
        
        self.cuda = cuda
        self.topt = train_opt
        
        """ load data """
        multiset = MultiSet(imgsets)
        num_samples = multiset.get_num_samples()
        total_images = sum(num_samples)
        train_loader, test_loader = ImgLoader(
            multiset,
            batch_size=train_opt['batch_size'],
            train_ratio=train_opt['train_ratio']
        )

        self.tw = num_samples/total_images # Compute weight (dataset size) parameter α_1, α_2, ... 
        self.dloader = {'train': train_loader, 'test': test_loader}
        
        """ build model """
        self.fe = MultiEncoder()
        self.ic = IndepClassifier()
        self.cf = torch.nn.CrossEntropyLoss()
        self.ot = optim.Adam(list(self.fe.parameters()) + list(self.ic.parameters()), lr=train_opt['learning_rate'])
        
        if cuda:
            self.fe = self.fe.cuda()
            self.ic = self.ic.cuda()
            self.cf = self.cf.cuda()
        
    def train(self):
        """ training """
        self.fe.train()
        self.ic.train()
        batch_size = self.topt['batch_size']
                               
        beta = self.topt['beta']
        
        for epoch in range(self.topt['epoch']):
            for index, (x_o, y_o) in enumerate(self.dloader['train']):
                x_a, y_a = self.reshape(x_o[0], y_o[0], batch_size, dim=[batch_size, 1, 28, 28])
                x_b, y_b = self.reshape(x_o[1], y_o[1], batch_size, dim=[batch_size, 3, 32, 32])
                x_c, y_c = self.reshape(x_o[2], y_o[2], batch_size, dim=[batch_size, 3, 28, 28])               
                
                self.ot.zero_grad()
                z_a, z_b, z_c, e_a, e_b, e_c = self.fe(x_a, x_b, x_c)
                               
                e = torch.stack([e_a, e_b, e_c], 0)
                
                y_hat_a, y_hat_b, y_hat_c = self.ic(z_a, z_b, z_c)
                               
                loss_cl = self.tw[0]*self.cf(y_hat_a, y_a) + \
                          self.tw[1]*self.cf(y_hat_b, y_b) + \
                          self.tw[2]*self.cf(y_hat_c, y_c)
                loss_fe = inter_class_distance(e, batch_size)
                loss = loss_cl + beta*loss_fe
                               
                pred_a = y_hat_a.argmax(dim=1, keepdim=True)
                pred_b = y_hat_b.argmax(dim=1, keepdim=True)
                pred_c = y_hat_c.argmax(dim=1, keepdim=True)
                               
                acc_a = pred_a.eq(y_a.view_as(pred_a)).sum().item()/batch_size
                acc_b = pred_b.eq(y_b.view_as(pred_b)).sum().item()/batch_size
                acc_c = pred_c.eq(y_c.view_as(pred_c)).sum().item()/batch_size
                loss.backward()
                self.ot.step()

                if index == 0:
                    print("epoch: {}, loss(FE): {:.3f}, loss(IC): {:.3f}, \
                          accuracy(MNIST): {:.3f}, accuracy(SVHN): {:.3f}, accuracy(MNISTM): {:.3f} \
                    ".format(epoch+1, loss_cl.item(), loss_fe.item(), acc_a, acc_b, acc_c))

    def predict(self):
        """ eval """
        self.fe.eval()
        self.ic.eval()
                               
        test_loss = 0
        correct = [0, 0, 0]
        num_testdata = [0, 0, 0]
        beta = self.topt['beta']
        num_batches = len(self.dloader['test'])
        batch_size = self.topt['batch_size']
        
        with torch.no_grad():
            for x_o, y_o in self.dloader['test']:
                x_a, y_a = self.reshape(x_o[0], y_o[0], batch_size, dim=[batch_size, 1, 28, 28])
                x_b, y_b = self.reshape(x_o[1], y_o[1], batch_size, dim=[batch_size, 3, 32, 32])
                x_c, y_c = self.reshape(x_o[2], y_o[2], batch_size, dim=[batch_size, 3, 28, 28])
                               
                z_a, z_b, z_c, e_a, e_b, e_c = self.fe(x_a, x_b, x_c)
                               
                e = torch.stack([e_a, e_b, e_c], 0)
                               
                y_hat_a, y_hat_b, y_hat_c = self.ic(z_a, z_b, z_c)  
                               
                loss_cl = self.tw[0]*self.cf(y_hat_a, y_a) + \
                          self.tw[1]*self.cf(y_hat_b, y_b) + \
                          self.tw[2]*self.cf(y_hat_c, y_c)
                loss_fe = inter_class_distance(e, self.topt['batch_size'])
                loss = loss_cl + beta*loss_fe
                
                test_loss += loss.item()
                pred_a = y_hat_a.argmax(dim=1, keepdim=True)
                pred_b = y_hat_b.argmax(dim=1, keepdim=True)
                pred_c = y_hat_c.argmax(dim=1, keepdim=True)
                               
                correct[0] += pred_a.eq(y_a.view_as(pred_a)).sum().item()
                correct[1] += pred_b.eq(y_b.view_as(pred_b)).sum().item()
                correct[2] += pred_c.eq(y_c.view_as(pred_c)).sum().item()
                
                num_testdata[0] += len(y_a)
                num_testdata[1] += len(y_b)
                num_testdata[2] += len(y_c)
                
            print('\nTest set: Average loss: {:.4f}, \
                    Accuracy(MNIST): {}/{} ({:.0f}%), \
                    Accuracy(SVHN): {}/{} ({:.0f}%), \
                    Accuracy(MNISTM): {}/{} ({:.0f}%)\n'.format(
                    test_loss/num_batches,
                    correct[0], num_testdata[0], 100. * correct[0] / num_testdata[0],
                    correct[1], num_testdata[1], 100. * correct[1] / num_testdata[1],
                    correct[2], num_testdata[2], 100. * correct[2] / num_testdata[2]))
            
        return [correct[0] / num_testdata[0], correct[1] / num_testdata[1], correct[2] / num_testdata[2]]
                               
    def reshape(self, x_o, y_o, batch_size, dim):
        """ transfers a tensor from CPU to GPU """
        x_b = torch.FloatTensor(dim)
        y_b = torch.LongTensor(batch_size)
        
        if self.cuda:
            x_o = x_o.cuda()
            y_o = y_o.cuda()
            x_b = x_b.cuda()
            y_b = y_b.cuda()
            
        x_b.resize_as_(x_o).copy_(x_o)
        y_b.resize_as_(y_o).copy_(y_o)
        
        return x_b, y_b