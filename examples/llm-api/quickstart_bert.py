#!/usr/bin/env python3
"""
Direct BERT inference using BertForSequenceClassification without LLM API boilerplate.

This script demonstrates how to directly use the BertForSequenceClassification class
from TensorRT-LLM's PyTorch backend for encoder-only models like BERT and RoBERTa.

Example usage:
    # Run with BERT base model
    python quickstart_bert.py --model-name bert-base-uncased
    
    # Run with RoBERTa
    python quickstart_bert.py --model-name roberta-base
    
    # Run with a fine-tuned sentiment analysis model
    python quickstart_bert.py --model-name nlptown/bert-base-multilingual-uncased-sentiment
    
    # Run with custom texts
    python quickstart_bert.py --model-name roberta-base \
        --texts "I love this!" "This is bad" "Neutral statement"
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from typing import List, Dict, Optional
import argparse

# TensorRT-LLM imports
from tensorrt_llm._torch.models.modeling_bert import BertForSequenceClassification
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.attention_backend import AttentionMetadata, TrtllmAttentionMetadata
from tensorrt_llm.mapping import Mapping


class SimpleBERTRunner:
    """
    Simple runner for BERT models without the full LLM API.
    
    This class demonstrates how to:
    1. Create a ModelConfig from a HuggingFace BERT/RoBERTa config
    2. Initialize BertForSequenceClassification from TensorRT-LLM
    3. Load weights from a HuggingFace checkpoint
    4. Create minimal AttentionMetadata for encoder-only inference
    5. Run inference directly without the full LLM API pipeline
    
    Key differences from full LLM API:
    - No KV cache needed (encoder-only model)
    - Simpler AttentionMetadata setup
    - Direct weight loading from HuggingFace
    - No generation/decoding logic needed
    """
    
    def __init__(self, 
                 model_name_or_path: str = "bert-base-uncased",
                 device: str = "cuda",
                 cuda_graph_enabled: bool = False):
        """
        Initialize the BERT runner.
        
        Args:
            model_name_or_path: HuggingFace model identifier or path
                Examples: "bert-base-uncased", "roberta-base", "bert-large-cased"
            device: Device to run the model on
        """
        self.device = torch.device(device)
        self.cuda_graph_enabled = cuda_graph_enabled
        
        # Load HuggingFace config and tokenizer
        print(f"Loading model config and tokenizer from: {model_name_or_path}")
        self.hf_config = AutoConfig.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        # Set torch dtype if not specified
        if not hasattr(self.hf_config, 'torch_dtype') or self.hf_config.torch_dtype is None:
            self.hf_config.torch_dtype = torch.float16
        
        # Store max position embeddings from config
        self.max_position_embeddings = getattr(self.hf_config, 'max_position_embeddings', 512)
        
        # Create ModelConfig for TensorRT-LLM
        self.model_config = ModelConfig(
            pretrained_config=self.hf_config,
            mapping=Mapping(),  # Single GPU mapping
        )
        
        # Initialize the model
        print("Initializing BertForSequenceClassification model...")
        self.model = torch.compile(BertForSequenceClassification(self.model_config), dynamic=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load weights from HuggingFace model
        print("Loading weights from HuggingFace model...")
        self._load_weights_from_hf(model_name_or_path)
        
        print("Model ready for inference!")
    
    def _load_weights_from_hf(self, model_name_or_path: str):
        """Load weights from HuggingFace model."""
        # Load HuggingFace model
        hf_model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            torch_dtype=self.hf_config.torch_dtype
        )
        hf_model.eval()
        
        # Get state dict and load into our model
        state_dict = hf_model.state_dict()
        self.model.load_weights(state_dict)
        
        # Clean up HuggingFace model
        del hf_model
        torch.cuda.empty_cache()
    
    def create_attention_metadata(self, 
                                 batch_size: int,
                                 seq_lengths: List[int],
                                 max_seq_len: Optional[int] = None) -> TrtllmAttentionMetadata:
        """
        Create TrtllmAttentionMetadata for BERT inference.
        
        BERT is an encoder-only model, so we create a simplified metadata:
        - No KV cache needed (encoder processes all tokens at once)
        - All sequences are in "context" phase (no generation)
        - No need for beam search or sampling parameters
        
        Args:
            batch_size: Number of sequences in the batch
            seq_lengths: Length of each sequence
            max_seq_len: Maximum sequence length (uses model's max_position_embeddings if not provided)
            
        Returns:
            TrtllmAttentionMetadata object configured for encoder-only inference
        """
        # Use model's max_position_embeddings if max_seq_len not provided
        if max_seq_len is None:
            max_seq_len = self.max_position_embeddings
        
        # Create seq_lens tensor
        seq_lens_tensor = torch.tensor(seq_lengths, dtype=torch.int32)
        
        # Create TrtllmAttentionMetadata with necessary parameters
        # When kv_cache_manager is None, max_seq_len must be provided
        attn_metadata = TrtllmAttentionMetadata(
            max_num_requests=batch_size,
            max_num_tokens=sum(seq_lengths),  # Total tokens across all sequences
            kv_cache_manager=None,  # No KV cache for encoder models
            seq_lens=seq_lens_tensor,
            num_contexts=batch_size,  # All sequences are in context phase for BERT
            max_seq_len=max_seq_len,  # Required when kv_cache_manager is None
            _max_seq_len_storage=max_seq_len,
            request_ids=[2051],
        )
        
        # Prepare the metadata (this sets up internal tensors)
        attn_metadata.prepare()
        
        return attn_metadata
    
    @torch.no_grad()
    def forward(self, 
                texts: List[str],
                max_length: int = 512) -> torch.Tensor:
        """
        Run inference on a batch of texts.
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Tokenize inputs
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded.get('attention_mask', torch.ones_like(input_ids)).to(self.device)
        
        # RoBERTa models don't use token_type_ids, so handle both cases
        if 'token_type_ids' in encoded:
            token_type_ids = encoded['token_type_ids'].to(self.device)
        else:
            # Create zeros for models like RoBERTa
            token_type_ids = torch.zeros_like(input_ids).to(self.device)
        
        batch_size, seq_len = input_ids.shape
        
        # Create position ids
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Calculate actual sequence lengths (excluding padding)
        seq_lengths = attention_mask.sum(dim=1).cpu().tolist()
        
        # Create attention metadata with max_seq_len
        attn_metadata = self.create_attention_metadata(batch_size, seq_lengths, max_seq_len=max_length)
        
        # Flatten inputs for packed format (BERT expects packed inputs)
        # Only keep non-padded tokens
        flat_input_ids = []
        flat_token_type_ids = []
        flat_position_ids = []
        
        for i in range(batch_size):
            length = seq_lengths[i]
            flat_input_ids.append(input_ids[i, :length])
            flat_token_type_ids.append(token_type_ids[i, :length])
            flat_position_ids.append(position_ids[i, :length])
        
        flat_input_ids = torch.cat(flat_input_ids, dim=0)
        flat_token_type_ids = torch.cat(flat_token_type_ids, dim=0)
        flat_position_ids = torch.cat(flat_position_ids, dim=0)
        
        # Original forward pass
        if not self.cuda_graph_enabled:
            # Run forward pass
            for i in range(21):
                if i == 20:
                    torch.cuda.cudart().cudaProfilerStart()
                logits = self.model(
                    attn_metadata=attn_metadata,
                    input_ids=flat_input_ids,
                    token_type_ids=flat_token_type_ids,
                    position_ids=flat_position_ids,
                )
                if i == 20:
                    torch.cuda.cudart().cudaProfilerStop()
        else:
            # Warm up
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for i in range(10):
                    logits = self.model(
                        attn_metadata=attn_metadata,
                        input_ids=flat_input_ids,
                        token_type_ids=flat_token_type_ids,
                        position_ids=flat_position_ids,
                    )
            torch.cuda.current_stream().wait_stream(s)

            # Capture with CUDA Graph
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                logits = self.model(
                    attn_metadata=attn_metadata,
                    input_ids=flat_input_ids,
                    token_type_ids=flat_token_type_ids,
                    position_ids=flat_position_ids,
                )
            # Replay and profile
            for i in range(21):
                if i == 20:
                    torch.cuda.cudart().cudaProfilerStart()
                g.replay()
                if i == 20:
                    torch.cuda.cudart().cudaProfilerStop()
        
        return logits
    
    def classify(self, texts: List[str]) -> List[Dict]:
        """
        Classify texts and return predictions.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of dictionaries with predictions
        """
        logits = self.forward(texts)
        
        # Get predictions
        predictions = torch.argmax(logits, dim=-1)
        probabilities = torch.softmax(logits, dim=-1)
        
        results = []
        for i, text in enumerate(texts):
            pred_label = predictions[i].item()
            pred_probs = probabilities[i].cpu().numpy()
            
            # Get label name if available
            label_name = pred_label
            if hasattr(self.hf_config, 'id2label') and self.hf_config.id2label:
                label_name = self.hf_config.id2label.get(pred_label, pred_label)
            
            results.append({
                'text': text,
                'predicted_label': label_name,
                'predicted_label_id': pred_label,
                'probabilities': pred_probs.tolist(),
                'confidence': float(pred_probs[pred_label])
            })
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Direct BERT inference without LLM API boilerplate'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='bert-base-uncased',
        help='HuggingFace model name or path (e.g., bert-base-uncased, roberta-base)'
    )
    parser.add_argument(
        '--texts',
        type=str,
        nargs='+',
        default=[
            "This movie is fantastic! I really enjoyed it.",
            "This movie is bad! I really did not enjoy it.",
            "I am sad today.",
        ],
        help='Texts to classify'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run on (cuda or cpu)'
    )
    parser.add_argument(
        '--cuda-graph-enabled',
        help='Enable CUDA Graph',
        action='store_true'
    )
    
    args = parser.parse_args()
    
    # Initialize runner
    print(f"\n{'='*60}")
    print(f"Initializing BERT Runner")
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"CUDA Graph Enabled: {args.cuda_graph_enabled}")
    print(f"{'='*60}\n")
    
    runner = SimpleBERTRunner(
        model_name_or_path=args.model_name,
        device=args.device,
        cuda_graph_enabled=args.cuda_graph_enabled
    )
    
    # Run classification
    print(f"\n{'='*60}")
    print(f"Running Classification")
    print(f"{'='*60}\n")
    
    results = runner.classify(args.texts)
    
    # Display results
    for i, result in enumerate(results):
        print(f"\nText {i+1}: \"{result['text']}\"")
        print(f"Predicted Label: {result['predicted_label']} (ID: {result['predicted_label_id']})")
        print(f"Confidence: {result['confidence']:.4f}")
        if len(result['probabilities']) <= 5:  # Show all probabilities if few classes
            print(f"All Probabilities: {result['probabilities']}")
    
    print(f"\n{'='*60}")
    print("Inference Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main() # use --model-name textattack/bert-base-uncased-yelp-polarity
