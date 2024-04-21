import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device1:", device)

# 数据载入和预处理
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=False,
                   transform=transforms.Compose([
                       transforms.Resize((28, 28)),
                       transforms.ToTensor(),
                       transforms.Normalize((-0.5,), (1.0,))
                   ])),
    batch_size=256, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=False,
                   transform=transforms.Compose([
                       transforms.Resize((28, 28)),
                       transforms.ToTensor(),
                       transforms.Normalize((-0.5,), (1.0,))
                   ])),
    batch_size=256, shuffle=False)

# 定义多层感知机（MLP）模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )
    def forward(self, x):
        x = self.layers(x)
        return x


# 初始化模型和优化器，添加L2正则化
weight_decay = 1e-5  # 调整正则化项的权重
# 预热和余弦退火的总epoch数
warmup_epochs = 10
momentum = 0.9
cosine_annealing_epochs = 40
lr=0.001
model = MLP().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=weight_decay)
# optimizer = optim.Adam(model.parameters(), lr=0.001)


# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 保存训练损失、测试误差和准确率
train_losses = []
train_accs = []
test_losses = []
test_accs = []

# 训练模型
for epoch in range(50):
    model.train()
    train_loss = 0
    correct_train = 0
    total_train = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # scheduler_warmup.step()
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct_train += pred.eq(target.view_as(pred)).sum().item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    train_acc = 100. * correct_train / len(train_loader.dataset)
    train_accs.append(train_acc)

    model.eval()
    correct_test = 0
    test_loss = 0
    total_test = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct_test += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    test_acc = 100. * correct_test / len(test_loader.dataset)
    test_accs.append(test_acc)


    print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f},Train Acc: {train_acc:.2f}, Test Acc: {test_acc:.2f}")


# 绘制损失和准确率曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accs) + 1), train_accs, label='Train Acc')
plt.plot(range(1, len(test_accs) + 1), test_accs, label='Test Acc')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig("shiyanyi_l2.png")
plt.show()
