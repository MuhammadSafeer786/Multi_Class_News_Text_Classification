import torch
import pytest
from transformers import BertForSequenceClassification, BertTokenizer

@pytest.fixture
def model_setup():
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
    
    for param in model.bert.encoder.layer[:6].parameters():
        param.requires_grad = False
    return model

@pytest.fixture
def tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased')

def test_output_shape(model_setup, tokenizer):
    """Verifies output dimensions match the 4 AG News categories."""
    text = "Wall St. Bears Claw Back [SEP] Short-sellers are winning."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    outputs = model_setup(**inputs)
    assert outputs.logits.shape == (1, 4), f"Expected (1, 4), got {outputs.logits.shape}"

def test_layer_freezing_logic(model_setup):
    """Ensures specific layers are frozen as per project requirements."""
    assert not next(model_setup.bert.encoder.layer[0].parameters()).requires_grad
    assert next(model_setup.bert.encoder.layer[11].parameters()).requires_grad

def test_label_mapping_range(model_setup):
    """Checks if model can compute loss with indices 0-3 (AG News 1-4 shifted)."""
    batch_size = 4
    dummy_input = torch.randint(0, 30522, (batch_size, 64))
    # AG News labels 1,2,3,4 are mapped to 0,1,2,3
    labels = torch.tensor([0, 1, 2, 3]) 
    
    outputs = model_setup(dummy_input, labels=labels)
    assert outputs.loss is not None
    assert outputs.loss > 0
