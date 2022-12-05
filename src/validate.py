from basic import *
from models import *
from matplotlib import pyplot as plt
import numpy as np

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
model_path = '../model_save/0.pt'
model = torch.load(model_path)['model']

score = 0
with torch.no_grad():
    for X, Y in val_loader:
        output = model.forward(X)
        # print(output)
        output = output.squeeze(0)
        y_predict = np.array(output.round().nonzero().view(-1,))
        # y_predict = output.argsort()[:3]
        # y_predict = np.array(y_predict)

        y_true = Y.squeeze(0)
        y_true = y_true.nonzero().view(-1, )
        y_true = np.array(y_true)

        # 严格版本
        if np.setdiff1d(y_true, y_predict).size == 0 and y_true.size==y_predict.size:
            print(y_true, y_predict)
            score += 1
        # if np.intersect1d(y_true, y_predict).size > 0:
        #     # print(y_true, y_predict)
        #     score += 1

print(f"正确个数：{score} \t得分:{(score/len(val_dataset))*100 : .3f}/100")

# val_loss = torch.load(model_path)['val_loss']
# train_loss = torch.load(model_path)['train_loss']
# plt.plot(range(len(val_loss)),val_loss, label='val loss')
# plt.plot(range(len(train_loss)),train_loss, label='train loss')
# plt.legend()
# plt.show()
