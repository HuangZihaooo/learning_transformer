import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 网络定义
net = nn.Sequential(
    nn.Linear(28 * 28, 256),
    nn.Sigmoid(),
    nn.Linear(256, 10)
)

# 权重初始化
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

# 训练配置
batch_size, lr, num_epochs = 256, 0.1, 10
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

# 数据加载
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 用于记录训练过程的列表
train_losses = []
train_accuracies = []
test_accuracies = []

def calculate_accuracy(loader):
    """计算准确率"""
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.view(data.size(0), -1)
            outputs = net(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total

def train():
    net.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(data.size(0), -1)
            
            # 前向传播
            output = net(data)
            loss = loss_fn(output, target)
            
            # 反向传播
            optimizer.zero_grad() # 清空梯度
            loss.backward() # 计算梯度
            optimizer.step() # 更新参数
            
            # 统计
            epoch_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # 记录每个epoch的结果
        avg_loss = epoch_loss / len(train_loader)
        train_acc = 100 * correct / total
        test_acc = calculate_accuracy(test_loader)
        
        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        print(f'Epoch {epoch}: Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%')

def plot_training_curves():
    """绘制训练曲线"""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 左侧y轴 - 损失
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    line1 = ax1.plot(range(num_epochs), train_losses, color=color, linewidth=2, 
                     marker='o', markersize=4, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # 右侧y轴 - 准确率
    ax2 = ax1.twinx()
    color_train = 'tab:blue'
    color_test = 'tab:green'
    ax2.set_ylabel('Accuracy (%)', color='black')
    line2 = ax2.plot(range(num_epochs), train_accuracies, color=color_train, 
                     linewidth=2, marker='s', markersize=4, label='Train Accuracy')
    line3 = ax2.plot(range(num_epochs), test_accuracies, color=color_test, 
                     linewidth=2, marker='^', markersize=4, label='Test Accuracy')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim([80, 100])
    
    # 添加图例
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    # 设置标题
    plt.title('MLP Training Progress: Loss and Accuracy', fontsize=14, fontweight='bold', pad=20)
    
    # 添加最终结果文本
    final_text = f'Final Results: Train Acc={train_accuracies[-1]:.2f}%, Test Acc={test_accuracies[-1]:.2f}%, Loss={train_losses[-1]:.4f}'
    plt.figtext(0.5, 0.02, final_text, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('./results/simple_mlp.png')

# 执行训练和可视化
if __name__ == "__main__":
    print("开始训练...")
    train()
    print("\n生成训练曲线...")
    plot_training_curves()
    print("完成！")
