from model import dataset

data_loader = dataset.create_dataloader_v1(
           txt = "Why don't we do it in the road?",
           batch_size = 1,
           max_length = 4,
           stride = 4,
        )

print("len:", len(data_loader))

data_iter = iter(data_loader)
inputs, targets = next(data_iter)
print("Inputs: \n", inputs)
print("Targets: \n", targets)


