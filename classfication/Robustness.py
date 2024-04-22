import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from classfication.Modle import Net
import torch.nn.functional as F

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

test_dataset = datasets.MNIST(root='D:\\project\\pyproject\\djangoProject\\pythonProject\\dataset\\mnist\\', train=False, download=False, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

model = Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

model.load_state_dict(torch.load('C:\\Users\\25086\\Desktop\\test\\test.txt'))

# FGSM attack code
def fgsm_attack(model, image, epsilon, target):
    image.requires_grad = True
    output = model(image)
    loss = F.nll_loss(output, target)  # 使用负对数似然损失作为损失函数
    model.zero_grad()
    loss.backward()
    perturbed_image = image + epsilon * image.grad.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def test():
    correct = 0
    total = 0
    correct_r = 0
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        perturbed_images = fgsm_attack(model, images, 0.5, labels)
        perturbed_output = model(perturbed_images)
        _,perturbed_pridicted = torch.max(perturbed_output.data, dim=1)
        correct_r +=(perturbed_pridicted == labels).sum().item()
    print('accuracy on test set: %d %% ' % (100 * correct / total))
    print('accuracy on Robustness_test set: %d %% ' % (100 * correct_r / total))


if __name__ == '__main__':
    test()


