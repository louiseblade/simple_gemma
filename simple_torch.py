import torch
import torch.nn as nn
from torchsummary import summary

class SimpleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc(x)
        return torch.softmax(out, dim=1)

# Example usage:
# Create an instance of the SimpleClassifier
input_size = 10  # Size of input features
num_classes = 3  # Number of classes for classification
model = SimpleClassifier(input_size, num_classes)

# Generate some random input data
batch_size = 4
input_data = torch.randn(batch_size, input_size)

summary(model, input_size=(input_size,))

# Perform forward pass
output = model(input_data)

# Print the output probabilities
print(output)