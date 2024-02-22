num_epochs = 5
lr = 0.001
momentum = 0.9
opt = optim.SGD(model_ft.parameters(), lr = lr, momentum = momentum)
sch = lr_scheduler.StepLR(opt, step_size=7, gamma=0.1)
ls = []

for i in range(num_epochs):
  total_loss = 0
  correct_predictions = 0
  total_predictions = 0
  for batch_idx, (images,labels), in enumerate (train_loader):
    images = images.to(device)
    labels = labels.to(device)

    outputs = efficientnet(images)
    loss = get_loss(outputs, labels)

    opt.zero_grad()
    loss.backward()
    opt.step()
    total_loss+=loss.item()

    _, predicted = torch.max(outputs.data, 1)
    total_predictions += labels.size(0)
    correct_predictions += (predicted == labels).sum().item()


    if (batch_idx + 1) % 100 == 0:
      batch_loss = total_loss / 100
      batch_accuracy = correct_predictions / total_predictions * 100
      print(f"Epoch [{i+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {batch_loss:.4f}, Accuracy: {batch_accuracy:.2f}%")
      total_loss = 0.0
      correct_predictions = 0
      total_predictions = 0


  ls.append(total_loss/len(train_loader))
  sch.step()

plt.plot(ls)
