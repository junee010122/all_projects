
import os
import torch
import numpy as np
import torch.optim as optim


from PIL import Image
from tqdm import tqdm

from main import CustomLSTM 


# if __name__ == "__main__":
# 
#     input_size = 400 * 400  # Adjust based on your preprocessing
#     hidden_size = 128  # Arbitrary choice, adjust as needed
#     output_size = 5  # Desired output size
# 
# 
#     model = CustomLSTM(input_size, hidden_size, output_size)
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
# 
#     optimizer.zero_grad()
# 
#     x = torch.rand(2, 3, input_size)
#     y = model(x)


class Dataset:

    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def load_and_format(self, all_files):

        sequence = []
        for current_file in all_files:
            image = Image.open(current_file).convert("L")
            size = int(image.size[0] * 0.10)
            image = np.asarray(image.resize((size, size))).reshape(-1)
            sequence.append(image)

        sequence = np.asarray(sequence)

        return sequence.astype(np.float32)

    def __getitem__(self, index):

        sample_seq_files, label_seq_files = self.samples[index], self.labels[index]
        sample = self.load_and_format(sample_seq_files)
        label = self.load_and_format(label_seq_files)

        return sample, label

    def __len__(self):

        return len(self.samples)


def update_plot(i, ax, data):

    ax.imshow(data[i], cmap="gray")


def plot_video(path, data, figsize=(14, 8), fontsize=24):

    fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(data[0], cmap="gray")
    ax.set_title("CNT Video", fontsize=fontsize)
    fig.tight_layout()

    fps = 2
    num_frames = data.shape[0]

    plots = [ax, data]
    ani = animation.FuncAnimation(fig, update_plot,
                                  frames=num_frames, fargs=(plots))

    writer = PillowWriter(fps=fps)
    ani.save(path, writer=writer)

    plt.close()


def convert_m2m(data, sequence):

    in_size = sequence["in"]
    out_size = sequence["out"]

    all_samples, all_labels = [], []

    for i in range(len(data)):

        if i > len(data) - out_size - in_size:
            continue

        all_samples.append(data[i:i+in_size])
        all_labels.append(data[i+in_size:i + in_size + out_size])

    return Dataset(all_samples, all_labels)


def load_folder(path):

    all_files = [ele for ele in os.listdir(path)
                 if ".png" in ele][:20]

    all_samples = []
    for current_file in tqdm(all_files, desc="Loading"):
        path_file = os.path.join(path, current_file)

        # image = np.asarray(Image.open(path_file).convert("L"))
        # size = int(image.shape[0] * 0.10)
        # image = cv2.resize(image, (size, size))
        # image = np.expand_dims(image, axis=0)
        # all_samples.append(image)

        all_samples.append(path_file)

    return all_samples


def example(path, sequence, batch_size, num_workers):

    # Load: All Data

    data = load_folder(path)

    # plot_video("test.gif", np.squeeze(data))

    # Convert: Data --> M2M Sequence

    data = convert_m2m(data, sequence)

    # Convert: Pytorch DataLoader

    data = torch.utils.data.DataLoader(data, shuffle=True,
                                       batch_size=batch_size,
                                       num_workers=num_workers)

    
    return data

if __name__ == "__main__":

    path = "/Users/june/Documents/results/CNT"
    sequence = {"in": 3, "out": 5}
    num_workers = 1
    batch_size = 2

    data_loader=example(path, sequence, batch_size, num_workers)
    from IPython import embed
    embed()

    # Model setup

    input_size = 400 * 400  # Adjust based on your preprocessing
    hidden_size = 128  # Arbitrary choice, adjust as needed
    output_size = 5  # Desired output size
    model = CustomLSTM(input_size, hidden_size, output_size)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        for sequences, labels in data_loader:

            # sequences = sequences.view(batch_size, sequence['in'], -1)

            optimizer.zero_grad()
            outputs = model(x)

            loss = criterion(outputs, labels.view(batch_size, -1))
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluation and visualization
    model.eval()
    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences = sequences.view(batch_size, sequence['in'], -1)
            outputs = model(sequences)
            visualize_predictions(labels.numpy(), outputs.numpy(), sequence['out'])
            break  # Visualize the first batch

