import argparse
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AlbertTokenizer, AlbertForSequenceClassification
    
class FakeNewsDataset(Dataset):
    def __init__(self, mode, tokenizer):
        assert mode in ["train", "test"]
        self.mode = mode
        self.df = pd.read_csv(mode + ".tsv", sep="\t").fillna("")
        self.len = len(self.df)
        self.label_map = {'veryhigh': 0, 'high': 1, 'low': 2, 'verylow': 3}
        self.tokenizer = tokenizer 
    
    def __getitem__(self, idx):
        text_a, text_b, label = self.df.iloc[idx, :].values
        label_id = self.label_map[label]
        label_tensor = torch.tensor(label_id)
            
        word_pieces = ["[CLS]"]
        tokens_a = self.tokenizer.tokenize(text_a)
        word_pieces += tokens_a + ["[SEP]"]
        len_a = len(word_pieces)
        
        tokens_b = self.tokenizer.tokenize(text_b)
        word_pieces += tokens_b + ["[SEP]"]
        len_b = len(word_pieces) - len_a
        
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        segments_tensor = torch.tensor([0] * len_a + [1] * len_b, dtype=torch.long)
        
        return (tokens_tensor, segments_tensor, label_tensor)
    
    def __len__(self):
        return self.len

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

def get_predictions(model, dataloader, compute_acc=False):
    predictions = None
    correct = 0
    total = 0
      
    with torch.no_grad():
        for data in dataloader:
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]
            
            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors)
            
            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)
            
            if compute_acc:
                labels = data[3]
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))
    
    if compute_acc:
        acc = correct / total
        return predictions, acc
    return predictions

class BERTHandler:
    def __init__(self, batch_size, epoch_num, model_name, is_test):
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epoch_num
        self.NUM_LABELS = 4
        self.model_name = model_name

        if self.model_name == "bert":
            self.model_version = 'bert-base-cased'
            self.tokenizer = BertTokenizer.from_pretrained(self.model_version)
            if is_test:
                self.model = BertForSequenceClassification.from_pretrained(
                    model_name+"_model", num_labels=self.NUM_LABELS)
            else:
                self.model = BertForSequenceClassification.from_pretrained(
                    self.model_version, num_labels=self.NUM_LABELS)
        elif self.model_name == "robert":
            self.model_version = 'roberta-base'   
            self.tokenizer = RobertaTokenizer.from_pretrained(self.model_version) 
            if is_test:
                self.model = RobertaForSequenceClassification.from_pretrained(
                    model_name+"_model", num_labels=self.NUM_LABELS)
            else:
                self.model = RobertaForSequenceClassification.from_pretrained(
                    self.model_version, num_labels=self.NUM_LABELS)
        elif self.model_name == "albert":
            self.model_version = 'albert-base-v2'   
            self.tokenizer = AlbertTokenizer.from_pretrained(self.model_version) 
            if is_test:
                self.model = AlbertForSequenceClassification.from_pretrained(
                    model_name+"_model", num_labels=self.NUM_LABELS)
            else:
                self.model = AlbertForSequenceClassification.from_pretrained(
                    self.model_version, num_labels=self.NUM_LABELS)

        if is_test:
            self.testset = FakeNewsDataset("test", tokenizer=self.tokenizer)
            self.testloader = DataLoader(self.testset, batch_size=self.BATCH_SIZE, 
                                    collate_fn=create_mini_batch)
        else:
            self.trainset = FakeNewsDataset("train", tokenizer=self.tokenizer)   
            self.trainloader = DataLoader(self.trainset, batch_size=self.BATCH_SIZE, 
                                    collate_fn=create_mini_batch)
            self.model.train()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self):
        for epoch in range(self.EPOCHS):
    
            running_loss = 0.0
            for data in self.trainloader:
                
                tokens_tensors, segments_tensors, \
                masks_tensors, labels = [t.to(self.device) for t in data]

                self.optimizer.zero_grad()
                
                # forward pass
                outputs = self.model(input_ids=tokens_tensors, 
                                token_type_ids=segments_tensors, 
                                attention_mask=masks_tensors, 
                                labels=labels)

                loss = outputs[0]
                # backward
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                
            _, acc = get_predictions(self.model, self.trainloader, compute_acc=True)

            print('[epoch %d] loss: %.3f, acc: %.3f' %
                (epoch + 1, running_loss, acc))

        self.model.save_pretrained(self.model_name+"_model")
    
    def test(self):
        pred, acc = get_predictions(self.model, self.testloader, compute_acc=True)

        print('test acc: %.3f' %(acc))
        print(pred)

def main():
    parser = argparse.ArgumentParser(description="Train and test a BERT model")
    parser.add_argument("-b", "--batch", type=int, default=4,
                        help="Batch size (default: 4)")
    parser.add_argument("-e", "--epoch", type=int, default=40,
                        help="Epoch number (default: 6)")
    parser.add_argument("-m", "--model", type=str, default="bert",
                        help="Model name (default: bert)")
    parser.add_argument("-t", "--test", default=False, action="store_true", 
                        help="Test the trained model.")                  
    args = parser.parse_args()

    bert = BERTHandler(args.batch, args.epoch, args.model, args.test)

    if args.test:
        bert.test()
    else:
        bert.train()

if __name__ == '__main__':
    main()