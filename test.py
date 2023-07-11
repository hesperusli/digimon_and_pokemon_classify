import torch
import sklearn.metrics
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def evaluate(test_loader,model,criterion):
#     #设置为验证模型
#     model.eval()
#     # 初始化准确度和损失
#     running_loss=0.0
#     running_acc=0.0
#     #加载数据到测试集
#     for inputs,targets in test_loader:
#         inputs=inputs.to(device)
#         targets=targets.to(device)
#         # 加载inputs到模型得到output
#         outputs=model(inputs)
#         # 计算损失
#         loss=criterion(outputs,targets)
#         #获得预测
#         _,preds=torch.max(outputs,1)
#         #计算准确度
#         acc=sklearn.metrics.accuracy_score(targets.cpu(),preds.cpu())
#         #更新损失和准确率
#         running_loss+=loss.item()*inputs.size(0)
#         running_acc+=acc*inputs.size(0)
#         #计算平均损失和平均准确率
#         avg_loss=running_loss/len(test_loader.dataset)
#         avg_acc=running_acc/len(test_loader.dataset)
#
#         return avg_loss,avg_acc
def evaluate(test_loader, model, criterion):
  # set the model to evaluation mode
  model.eval()

  # initialize the running loss and accuracy
  running_loss = 0.0
  running_acc = 0.0

  # loop through the test data loader
  for inputs, targets in test_loader:
    # move the inputs and targets to the device
    inputs = inputs.to(device)
    targets = targets.to(device)

    # forward pass and get the output logits
    outputs = model(inputs)
    # compute the loss
    loss = criterion(outputs, targets)

    # get the predictions
    _, preds = torch.max(outputs, 1)
    # compute the accuracy
    acc = sklearn.metrics.accuracy_score(targets.cpu(), preds.cpu())

    # update the running loss and accuracy
    running_loss += loss.item() * inputs.size(0)
    running_acc += acc * inputs.size(0)

  # calculate the average loss and accuracy
  avg_loss = running_loss / len(test_loader.dataset)
  avg_acc = running_acc / len(test_loader.dataset)

  return avg_loss, avg_acc