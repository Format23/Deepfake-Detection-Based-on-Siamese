#-*-coding:GBK -*-



import time
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import cv2
from torchvision import datasets, models, transforms
from netall.xce import *
from createdata.create import *
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
 
 
def main():
     args = parse.parse_args()
     name = args.name
     train_path = args.train_path
     val_path = args.val_path
     continue_train = args.continue_train
     epoches = args.epoches
     batch_size = args.batch_size
     model_name = args.model_name
     model_path = args.model_path
     output_path = os.path.join('./outputnew', name)
     if not os.path.exists(output_path):
         os.mkdir(output_path)
     torch.backends.cudnn.benchmark = True
 
     # Creat the model
     model = Xception()
     if continue_train:
         model.load_state_dict(torch.load(model_path))
     model = model.cuda()
     #print(model)
     #Loss
     #criterion = nn.BCEWithLogitsLoss()        #ContrastiveLoss()
     #criterion = BCE_CELoss()
     criterion = nn.CrossEntropyLoss()
     # optimizer
     #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)
     optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001)
     scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
 
     # Train the model using multiple GPUs
 
     best_model_wts = model.state_dict()
     
     best_acc = 0.0
     iteration = 0
     
     data_len = create_pair_train(train_path)
     print(data_len)
     max_iters = data_len // batch_size 
     data_len = max_iters*batch_size
     print(data_len)
     
     test_data_len = create_pair_test(val_path)
     print(test_data_len)
     test_max_iters = test_data_len // batch_size
     test_data_len = test_max_iters*batch_size
     print(test_data_len)
        
     #画图准备
     counter = []
     train_loss_history = []
     val_loss_history = []
     train_acc = []
     val_acc = []
     
     #training
     for epoch in range(epoches):
         adjust_learning_rate(optimizer, epoch)
         batch_time = AverageMeter()
         data_time = AverageMeter()
         print('Epoch {}/{}'.format(epoch + 1, epoches))
         print('-' * 10)
         model = model.train()
         end = time.time()
         print('learning rate:{}'.format(optimizer.param_groups[0]['lr']))
         train_loss = 0.0
         train_corrects = 0.0
         val_loss = 0.0
         val_corrects = 0.0
         num0 = 0.0
         num = 0.0
         for i in range(max_iters):
             data_time.update(time.time() - end)
             
             image_b,image_f,label = read_pair_train(batch_size)
             label = label.long()
             iter_loss = 0.0
             iter_corrects = 0.0
             
             image_b = image_b.cuda()
             image_f = image_f.cuda()
             label = label.cuda()
             
             optimizer.zero_grad()
             outputs = model(image_b,image_f)
             loss = criterion(outputs, label)
             loss.backward()
             optimizer.step()
             outputs = torch.nn.functional.softmax(outputs,dim=1)
             _, preds = torch.max(outputs.data, 1)
             
             iter_loss = loss.data.item()
             train_loss += iter_loss
             iter_corrects = torch.sum(preds == label.data).to(torch.float32)
             train_corrects += iter_corrects
             iteration += 1
             num0 = torch.sum(0 == label.data).to(torch.float32)
             num +=num0
             if not (iteration % 200):
                 print('iteration {} train loss: {:.4f} Acc: {:.4f}'.format(iteration, iter_loss, iter_corrects / len(outputs)))
                    
         epoch_loss = train_loss / max_iters
         epoch_acc = train_corrects / data_len
 
         batch_time.update(time.time() - end)
         end = time.time()
 
         print('Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
               'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
               .format(batch_time=batch_time, data_time=data_time))
         print('epoch train loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
         print('伪造图片一共：{}'.format(num))
         counter.append(epoch)
         train_loss_history.append(epoch_loss) 
         train_acc.append(epoch_acc)
         
         #val
         model.eval()
         with torch.no_grad():
             for i in range(test_max_iters):
                 image_b,image_f,label = read_pair_test(batch_size)
                 label = label.long()
                 image_f = image_f.cuda()
                 image_b = image_b.cuda()
                 label = label.cuda()

                 outputs = model(image_b,image_f)
                 
                 loss = criterion(outputs,label)
                 outputs = torch.nn.functional.softmax(outputs,dim=1)
                 _, preds = torch.max(outputs.data, 1)
                 val_loss += loss.data.item()

                 
                 
                 val_corrects += torch.sum(preds == label.data).to(torch.float32)
             
             epoch_loss = val_loss / test_max_iters
             epoch_acc = val_corrects / test_data_len
             val_loss_history.append(epoch_loss)
             val_acc.append(epoch_acc)
             print('epoch val loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
             
             if epoch_acc > best_acc:
                 best_acc = epoch_acc
                 best_model_wts = model.state_dict()
                 best_model = model
         #scheduler.step()
         if not (epoch % 10):
             # Save the model trained with multiple gpu
             # torch.save(model.module.state_dict(), os.path.join(output_path, str(epoch) + '_' + model_name))
             torch.save(model.state_dict(), os.path.join(output_path, str(epoch) + '_' + model_name))
         print('Best val Acc: {:.4f}'.format(best_acc))
         model.load_state_dict(best_model_wts)
         # torch.save(model.module.state_dict(), os.path.join(output_path, "best.pkl"))
         # torch.save(model.state_dict(), os.path.join(output_path, "best.pkl"))
         torch.save(model.state_dict(), os.path.join(output_path, "best.h5"))
     
     #torch.save(best_model, os.path.join(output_path, "best_{:.4f}model.pth".format(best_acc)))
     torch.save(model.state_dict(), os.path.join(output_path, "best_{:.4f}.h5".format(best_acc)))
     show_plot(counter,train_acc,val_acc,'acc')    
     show_plot(counter,train_loss_history,val_loss_history,'loss')
 
def adjust_learning_rate(optimizer, epoch):
     """
      For AlexNet, the lr starts from 0.05, and is divided by 10 at 90 and 120 epochs
     """
     args = parse.parse_args()
     warmup = 10
     #if epoch < warmup:
     #    #lr = (args.lr - 0.000001) * (epoch / warmup) + 0.000001'''
     if epoch < warmup:
         lr = args.lr  * ((epoch+1) / warmup)
     elif epoch < 20:
         lr = args.lr 
     elif epoch < 25:
         lr = args.lr*0.1
     elif epoch < 30:
         lr = args.lr * 0.01
     '''if epoch < warmup:
         lr = args.lr  * ((epoch+1) / warmup)
     elif epoch < 10:
         lr = args.lr 
     elif epoch < 15:
         lr = args.lr*0.1
     elif epoch < 20:
         lr = args.lr * 0.01  '''

 
     """scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0, last_epoch=-1)"""

     for param_group in optimizer.param_groups:
         param_group['lr'] = lr
 
 
class AverageMeter(object):
     """Computes and stores the average and current value"""
 
     def __init__(self):
         self.reset()
 
     def reset(self):
         self.val = 0
         self.avg = 0
         self.sum = 0
         self.count = 0
 
     def update(self, val, n=1):
         self.val = val
         self.sum += val * n
         self.count += n
         self.avg = self.sum / self.count
         
def show_plot(iteration,loss1,loss2,name):
     plt.plot(iteration,loss1,'b',iteration,loss2,'r')
     plt.savefig('./charm/'+name+'.png')
     plt.show()
 
if __name__ == '__main__':
    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR',help='initial learning rate')#0.0001
    parse.add_argument('--name', '-n', type=str, default='Mesonet')
    parse.add_argument('--train_path', '-tp', type=str, default='./deepfake_database/train')
    parse.add_argument('--val_path', '-vp', type=str, default='./deepfake_database/val')
    parse.add_argument('--batch_size', '-bz', type=int, default=64)
    parse.add_argument('--epoches', '-e', type=int, default='50')
    parse.add_argument('--model_name', '-mn', type=str, default='meso4.h5')
    parse.add_argument('--continue_train','-ct', type=bool, default=False)
    # parse.add_argument('--model_path', '-mp', type=str, default='./output/Mesonet/best.pkl')
    parse.add_argument('--model_path', '-mp', type=str, default='./output/best.h5')
    main()