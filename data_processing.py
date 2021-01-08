import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer
from IPython.display import clear_output
from BertModules import BertClassifier
from pytorch_transformers import BertConfig
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn

config = BertConfig.from_pretrained('albert_config.json')

# Create our custom BERTClassifier model object
model = BertClassifier(config)
tokenizer = BertTokenizer(vocab_file='bert-base-uncased-vocab.txt')
BATCH_SIZE = 16
device = torch.device("cuda:0")
print("device:", device)
model = model.to(device)
criterion = nn.CrossEntropyLoss()



def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    

    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None

    tokens_tensors = pad_sequence(tokens_tensors, 
                                  batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, 
                                    batch_first=True)
    

    masks_tensors = torch.zeros(tokens_tensors.shape, 
                                dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(
        tokens_tensors != 0, 1)
    
    return tokens_tensors, segments_tensors, masks_tensors, label_ids



class FakeNewsDataset(Dataset):
    def __init__(self, mode, tokenizer):
        self.mode = mode
        self.df = pd.read_csv("youtube_done_reduce2.csv",encoding='ISO-8859-15', sep=",").fillna("")
        self.len = len(self.df)
        self.label_map = {'veryhigh': 0, 'high': 1, 'low': 2, 'verylow': 3}
        self.tokenizer = tokenizer  

    def __getitem__(self, idx):
        if self.mode == "test":
            title = self.df.iloc[idx, :2].values
            view_tensor = None
            like_tensor = None
            dislike_tensor = None
        else:
            channel, title, category, views, like, dislike, link = self.df.iloc[idx].values
            view_id = self.label_map[views]
            view_tensor = torch.tensor(view_id)
            like_id = self.label_map[like]
            like_tensor = torch.tensor(like_id)
            dislike_id = self.label_map[dislike]
            dislike_tensor = torch.tensor(dislike_id)
        word_pieces = ["[CLS]"]
        tokens_a = self.tokenizer.tokenize(title)
        word_pieces += tokens_a + ["[SEP]"]
        len_a = len(word_pieces)
 
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        

        segments_tensor = torch.tensor([0] * len_a, 
                                        dtype=torch.long)
        
        return (tokens_tensor, segments_tensor, view_tensor)
    
    def __len__(self):
        return self.len
    
    
def get_predictions(model, dataloader, compute_acc=False):
    predictions = None
    correct = 0
    total = 0
      
    with torch.no_grad():
        # 遍巡整個資料集
        for data in dataloader:
            # 將所有 tensors 移到 GPU 上
            if next(model.parameters()).is_cuda:
                data = [t.to(device) for t in data if t is not None]
            
            
            # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
            # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors)
            
            logits = outputs
            _, pred = torch.max(logits.data, 1)
            # 用來計算訓練集的分類準確率
            if compute_acc:
                labels = data[3]
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                
            # 將當前 batch 記錄下來
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))
    
    if compute_acc:
        acc = correct / total
        return predictions, acc
    return predictions

trainset = FakeNewsDataset("train", tokenizer=tokenizer)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, 
                         collate_fn=create_mini_batch)

clear_output()

data = next(iter(trainloader))

tokens_tensors, segments_tensors, \
    masks_tensors, label_ids = data

print(f"""
tokens_tensors.shape   = {tokens_tensors.shape} 
{tokens_tensors}
------------------------
segments_tensors.shape = {segments_tensors.shape}
{segments_tensors}
------------------------
masks_tensors.shape    = {masks_tensors.shape}
{masks_tensors}
------------------------
label_ids.shape        = {label_ids.shape}
{label_ids}
""")



# 訓練模式
model.train()

# 使用 Adam Optim 更新整個分類模型的參數
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
cou = 0

_, acc = get_predictions(model, trainloader, compute_acc=True)
print('[epoch %d] , acc: %.3f' %
      (0, acc))

EPOCHS = 60  # 幸運數字
for epoch in range(EPOCHS):
    
    running_loss = 0.0
    for data in trainloader:
        
        tokens_tensors, segments_tensors, \
        masks_tensors, labels = [t.to(device) for t in data]

        # 將參數梯度歸零
        optimizer.zero_grad()
        
        # forward pass
        outputs = model(input_ids=tokens_tensors, 
                        token_type_ids=segments_tensors, 
                        attention_mask=masks_tensors)
        loss = criterion(outputs, labels)
        # backward
        loss.backward()
        optimizer.step()


        # 紀錄當前 batch loss
        running_loss += loss.item()
    #print("test")
    # 計算分類準確率
    _, acc = get_predictions(model, trainloader, compute_acc=True)
    torch.save(model,'save_%d.pt' % epoch)
    print('[epoch %d] loss: %.3f, acc: %.3f' %
          (epoch + 1, running_loss, acc))
pred, acc = get_predictions(model, trainloader, compute_acc=True)
print("classification acc:", acc)