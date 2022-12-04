from basic import *
from Train_and_Evaualtion import *
from models import *

import torch.optim as optim
import random

EPOCHS = 50
CLIP = 1
input_dim = len(vocabulary)
embedding_dim = 2048
hidden_dim = 1024
output_dim = 17
model = SimpleNN(input_dim, embedding_dim, hidden_dim, output_dim)
save_path = '../model_save/0.pt'
optimizer = optim.Adam(model.parameters())
loss_fn = nn.BCELoss()

best_valid_loss = float('inf')
train_losses = []
valid_losses = []
for epoch in range(EPOCHS):
    random_ids = list(train_df['id']).copy()
    random.shuffle(random_ids)
    k = 10
    fold_len = train_df.shape[0] // k

    train_loss = 0
    valid_loss = 0

    for i in tqdm(range(k)):
        test_ids = random_ids[i*fold_len:(i+1)*fold_len]
        train_ids = random_ids[:i*fold_len] + random_ids[(i+1)*fold_len:]

        train_dataset = TextDataset(train_df[train_df['id'].isin(train_ids)])
        test_dataset = TextDataset(train_df[train_df['id'].isin(test_ids)])

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=8)

        train_loss += train(model, train_loader, loss_fn=loss_fn, optimizer=optimizer, clip=CLIP)
        valid_loss += evaluate(model, test_loader, loss_fn=loss_fn)

    valid_loss /= k
    train_loss /= k

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    if valid_loss < best_valid_loss:
        torch.save({'val_loss':valid_losses,
                    'train_loss':train_losses,
                    'model':model,
                    }, save_path)
    print(f'Epoch: {epoch + 1:02}')
    print(f'\tTrain Loss: {train_loss/k:.6f}')
    print(f'\t Val. Loss: {valid_loss/k:.6f}')