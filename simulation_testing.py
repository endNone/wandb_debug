import random
import wandb
import os
import time
os.environ["WANDB_API_KEY"] = ''
wandb.init(project="test_project", entity="endNone")
for step in range(50):
    accuracy = random.random()  
    loss = random.random()  
    wandb.log({"accuracy": accuracy, "loss": loss})
    print("Record successful")
    time.sleep(120*accuracy)
wandb.finish()