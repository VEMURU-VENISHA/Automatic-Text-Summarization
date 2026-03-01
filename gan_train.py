import torch
import torch.optim as optim
from datasets import load_dataset
from gan_model import generator,discriminator,tokenizer,generate_summary,device

criterion=torch.nn.BCELoss()
optD=optim.Adam(discriminator.parameters(),lr=0.001)
optG=optim.Adam(generator.parameters(),lr=1e-5)

dataset=load_dataset("cnn_dailymail","3.0.0",split="train[:20]")

for epoch in range(1):

    for item in dataset:

        article=item["article"]
        real_summary=item["highlights"]

        fake_summary,fake_ids=generate_summary(article)

        real_ids=tokenizer(real_summary,return_tensors="pt",truncation=True,max_length=100)["input_ids"].to(device)
        fake_ids=fake_ids.to(device)

        # train D
        optD.zero_grad()

        real_labels=torch.ones(1,1).to(device)
        fake_labels=torch.zeros(1,1).to(device)

        d_loss=criterion(discriminator(real_ids),real_labels)+\
               criterion(discriminator(fake_ids),fake_labels)

        d_loss.backward()
        optD.step()

        # train G
        optG.zero_grad()

        g_loss=criterion(discriminator(fake_ids),real_labels)

        g_loss.backward()
        optG.step()

    print("GAN training done")