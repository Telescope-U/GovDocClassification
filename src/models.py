from torch import nn
import torch
from torch.nn import functional as F

class SimpleNN(nn.Module):
    """
    一个简单的神经网络模型，得到初始结果。
    embeddingBag + 2*linear
    """
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.EmbeddingBag(input_dim, embedding_dim)
        self.linear = nn.Linear(embedding_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, src):
        x = self.embedding(src)
        x = F.relu(self.linear(x))
        prediction = torch.sigmoid(self.out(x))
        return prediction

class GCNN(nn.Module):
    def __init__(self, input_dim, embedding_dim = 64, num_class = 18):
        super(GCNN, self).__init__()#对父类初始化

        self.embedding_table = nn.Embedding(input_dim, embedding_dim)
        nn.init.xavier_uniform_(self.embedding_table.weight)
        #对文本使用一维卷积,embedding_dim作为输入通道数，64输出通道，15是kernel
        #步长接近kernel的一半，每次滑动会有一半的交叉
        self.conv_A_1 = nn.Conv1d(embedding_dim, 64, 15, stride = 7)
        self.conv_B_1 = nn.Conv1d(embedding_dim, 64, 15, stride = 7)

        self.conv_A_2 = nn.Conv1d(64, 64, 15, stride = 7)
        self.conv_B_2 = nn.Conv1d(64, 64, 15, stride = 7)

        self.output_linear1 = nn.Linear(64, 128)
        self.output_linear2 = nn.Linear(128, num_class)

    def forward(self, word_index):

        #1.得到word_embedding
        #word_index shape:[bs, max_seq_len]
        word_embedding = self.embedding_table(word_index) #[bs, max_seq_len, embedding_dim]
        # print("word_embedding:",word_embedding.shape)
        #2 第一层1D门卷积模块
        word_embedding = word_embedding.transpose(1, 2) #[bs, embedding_dim, max_seq_len]
        A = self.conv_A_1(word_embedding)  #(max_seq_len - kernel) // stride + 1，这里长度对吗？对的
        # A_shape [16, 64, 713]
        # print("A_embedding:",A.shape)
        B = self.conv_B_1(word_embedding)
        # B_shape [16, 64, 713]
        # print('B_embedding:', B.shape)
        H = torch.bmm(A , torch.sigmoid(B).transpose(1, 2))  #[bs, embedding_dim, embedding_dim]
        # print("H_embedding:",H.shape)

        A = self.conv_A_2(H)
        # print("A2_embedding:",A.shape)
        B = self.conv_B_2(H) # [16, 64, 8]
        # print('B_embedding:', B.shape)
        H = torch.bmm(A, torch.sigmoid(B).transpose(1, 2)) #[bs, 64, max_seq_len]
        # [16,64, 64]
        # print("H2_embedding:",H.shape)

        #3. 池化并经过全连接层
        pool_output = torch.mean(H, dim = -1) #平均池化，得到[bs, 64]
        linear1_output = self.output_linear1(pool_output)
        logits = self.output_linear2(linear1_output) #[bs,18]

        return torch.sigmoid(logits)