import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryScorer(nn.Module):
    def __init__(self, feature_dim, mem_dim, num_heads):
        super(MemoryScorer, self).__init__()
        self.feature_dim = feature_dim
        self.mem_dim = mem_dim
        self.num_heads = num_heads
        assert feature_dim % num_heads == 0, "Feature dimension must be divisible by the number of heads"
        self.head_dim = feature_dim // num_heads

        # Linear layers for Q, K
        self.query_proj = nn.Linear(feature_dim, mem_dim)
        self.key_proj = nn.Linear(mem_dim, mem_dim)

        # Scaling factor for attention
        self.scale = self.head_dim ** 0.5

    def generate_temporal_positional_embeddings(self, num_memory, feature_dim):
        """
        Generate sinusoidal positional embeddings for temporal indices.
        """
        position = torch.arange(0, num_memory).unsqueeze(1)  # Shape: (num_memory, 1)
        div_term = torch.exp(torch.arange(0, feature_dim, 2) * -(torch.log(torch.tensor(10000.0)) / feature_dim))
        pos_embedding = torch.zeros(num_memory, feature_dim)
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        return pos_embedding

    def forward(self, target_frame, memory_frames, training=False):
        batch_size = target_frame.shape[1] # technically always 2 for number of feature maps
        num_memory = memory_frames.shape[0]
        n_obj = memory_frames.shape[2]
        target_frame_flat = target_frame.reshape(batch_size, -1, self.feature_dim)
        memory_frames_flat = memory_frames.reshape(num_memory, n_obj, -1, self.mem_dim)


        # Generate and add temporal positional embeddings
        temporal_pos_embeddings = self.generate_temporal_positional_embeddings(num_memory, self.mem_dim).to(memory_frames.device)
        memory_frames_flat = memory_frames_flat + temporal_pos_embeddings.unsqueeze(1).unsqueeze(1)  # Broadcast to (num_memory, H*W, C)

        # Linear projections and reshape for multiple heads
        query = self.query_proj(target_frame_flat).reshape(batch_size, -1, self.num_heads, self.head_dim)  # (1, H*W, num_heads, head_dim)
        keys = self.key_proj(memory_frames_flat).reshape(num_memory, n_obj, -1, self.num_heads, self.head_dim)  # (num_memory, H*W, num_heads, head_dim)

        # Compute attention scores
        attention_scores = torch.einsum("bhnd,mohnd->mobn", query, keys) / self.scale

        # Normalize attention scores to probabilities
        attention_scores = torch.sum(attention_scores, dim=1, keepdim=True)
        similarity_scores = None
        print("Attention:", attention_scores.squeeze())
        if training:
            similarity_scores = F.softmax(attention_scores, dim=0) # n, 1, o, 1
        elif num_memory > 7:
            attention_scores, _ = torch.max(attention_scores, dim=2)
            values, _ = torch.topk(attention_scores.flatten(), 7)
            similarity_scores = (attention_scores > torch.min(values) - 1) / 7.0
            similarity_scores[0] = 1
        else:
            similarity_scores = torch.ones(num_memory) / num_memory
        print("Similarity:", similarity_scores.squeeze())
        return similarity_scores

