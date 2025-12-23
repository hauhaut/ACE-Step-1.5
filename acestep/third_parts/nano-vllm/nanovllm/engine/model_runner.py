import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model

import socket


def find_available_port(start_port: int = 2333, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port.
    
    Args:
        start_port: The starting port number to check
        max_attempts: Maximum number of ports to try
        
    Returns:
        An available port number
        
    Raises:
        RuntimeError: If no available port is found within max_attempts
    """
    for i in range(max_attempts):
        port = start_port + i
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('localhost', port))
                return port
        except OSError:
            # Port is in use, try next one
            continue
    raise RuntimeError(f"Could not find an available port starting from {start_port} after {max_attempts} attempts")


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        dist_port = find_available_port()
        print(f"[debug]dist_port: {dist_port}")
        dist.init_process_group("nccl", f"tcp://localhost:{dist_port}", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence], is_cfg_batch: bool = False):
        """Prepare sampling parameters. For CFG batch, only return parameters for conditional sequences."""
        if is_cfg_batch:
            # For CFG batch, seqs contains [cond_seq1, cond_seq2, ..., uncond_seq1, uncond_seq2, ...]
            # We only need parameters for conditional sequences (first half)
            num_cond = len(seqs) // 2
            temperatures = []
            cfg_scales = []
            top_ks = []
            top_ps = []
            repetition_penalties = []
            for seq in seqs[:num_cond]:
                temperatures.append(seq.temperature)
                cfg_scales.append(seq.cfg_scale)
                top_ks.append(seq.top_k if seq.top_k is not None else 0)
                top_ps.append(seq.top_p if seq.top_p is not None else 1.0)
                repetition_penalties.append(seq.repetition_penalty)
        else:
            temperatures = []
            cfg_scales = []
            top_ks = []
            top_ps = []
            repetition_penalties = []
            for seq in seqs:
                temperatures.append(seq.temperature)
                cfg_scales.append(seq.cfg_scale)
                top_ks.append(seq.top_k if seq.top_k is not None else 0)
                top_ps.append(seq.top_p if seq.top_p is not None else 1.0)
                repetition_penalties.append(seq.repetition_penalty)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        cfg_scales = torch.tensor(cfg_scales, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        top_ks = torch.tensor(top_ks, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        top_ps = torch.tensor(top_ps, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        repetition_penalties = torch.tensor(repetition_penalties, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures, cfg_scales, top_ks, top_ps, repetition_penalties

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """Run model forward and sampling. For CFG sequences, batch is structured as:
        [cond_seq1, cond_seq2, ..., uncond_seq1, uncond_seq2, ...]
        where uncond_seqi is the paired unconditional sequence of cond_seqi."""
        # Check if this is a CFG batch (contains paired conditional and unconditional sequences)
        is_cfg_batch = False
        if len(seqs) > 0:
            # CFG batch if first sequence has cfg_scale > 1.0 and paired_seq
            if seqs[0].cfg_scale > 1.0 and seqs[0].paired_seq is not None:
                is_cfg_batch = True
                # Verify batch structure: first half conditional, second half unconditional
                num_cond = len(seqs) // 2
                for i in range(num_cond):
                    if seqs[i].is_unconditional or seqs[i + num_cond].is_unconditional == False:
                        is_cfg_batch = False
                        break
        
        if is_cfg_batch:
            # CFG batch: seqs = [cond_seq1, cond_seq2, ..., uncond_seq1, uncond_seq2, ...]
            num_cond = len(seqs) // 2
            cond_seqs = seqs[:num_cond]
            uncond_seqs = seqs[num_cond:]
            
            # Prepare inputs for both conditional and unconditional (they're already in the batch)
            input_ids, positions = (self.prepare_prefill(seqs) if is_prefill 
                                   else self.prepare_decode(seqs))
            sample_params = self.prepare_sample(seqs, is_cfg_batch=True) if self.rank == 0 else None
            if sample_params is not None:
                temperatures, cfg_scales, top_ks, top_ps, repetition_penalties = sample_params
            else:
                temperatures = cfg_scales = top_ks = top_ps = repetition_penalties = None
            
            # Run model forward (processes entire batch: cond + uncond)
            logits_all = self.run_model(input_ids, positions, is_prefill)
            reset_context()
            
            if self.rank == 0:
                # Split logits: first half is conditional, second half is unconditional
                logits_cond = logits_all[:num_cond]
                logits_uncond = logits_all[num_cond:]
                
                # Apply repetition penalty to conditional logits (before CFG)
                if repetition_penalties is not None:
                    for i, seq in enumerate(cond_seqs):
                        penalty = repetition_penalties[i].item()
                        if penalty != 1.0:
                            # Only penalize completion tokens (not prompt tokens)
                            completion_tokens = torch.tensor(seq.completion_token_ids, device=logits_cond.device)
                            if len(completion_tokens) > 0:
                                # Create token mask: mark tokens that appeared in completion
                                token_mask = torch.zeros(logits_cond.shape[1], dtype=torch.bool, device=logits_cond.device)
                                token_mask[completion_tokens] = True
                                
                                # Apply standard repetition penalty formula (matching transformers implementation):
                                # For tokens in completion: if score < 0 then score * penalty, else score / penalty
                                penalty_scores = torch.where(
                                    logits_cond[i] < 0,
                                    logits_cond[i] * penalty,
                                    logits_cond[i] / penalty
                                )
                                # Only apply penalty to tokens that appeared in completion
                                logits_cond[i] = torch.where(token_mask, penalty_scores, logits_cond[i])
                
                # Apply CFG formula: logits_cfg = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
                cfg_scales_tensor = cfg_scales.unsqueeze(1)  # [num_cond, 1]
                logits_cfg = logits_uncond + cfg_scales_tensor * (logits_cond - logits_uncond)
                
                # Prepare input_ids for sampler (for repetition penalty, though we already applied it)
                cond_input_ids = torch.tensor([seq.token_ids for seq in cond_seqs], device=logits_cfg.device)
                
                # Sample from CFG logits
                token_ids_cfg = self.sampler(
                    logits_cfg, 
                    temperatures,
                    top_ks=top_ks if top_ks is not None else None,
                    top_ps=top_ps if top_ps is not None else None,
                    repetition_penalties=None,  # Already applied above
                    input_ids=cond_input_ids,
                ).tolist()
                
                # Return token_ids (will be applied to both conditional and unconditional sequences)
                return token_ids_cfg
            else:
                return None
        else:
            # Normal batch (non-CFG)
            input_ids, positions = (self.prepare_prefill(seqs) if is_prefill 
                                   else self.prepare_decode(seqs))
            sample_params = self.prepare_sample(seqs, is_cfg_batch=False) if self.rank == 0 else None
            if sample_params is not None:
                temperatures, cfg_scales, top_ks, top_ps, repetition_penalties = sample_params
            else:
                temperatures = cfg_scales = top_ks = top_ps = repetition_penalties = None
            logits = self.run_model(input_ids, positions, is_prefill)
            reset_context()
            
            if self.rank == 0:
                # Apply repetition penalty to logits
                if repetition_penalties is not None:
                    for i, seq in enumerate(seqs):
                        penalty = repetition_penalties[i].item()
                        if penalty != 1.0:
                            # Only penalize completion tokens (not prompt tokens)
                            completion_tokens = torch.tensor(seq.completion_token_ids, device=logits.device)
                            if len(completion_tokens) > 0:
                                # Create token mask: mark tokens that appeared in completion
                                token_mask = torch.zeros(logits.shape[1], dtype=torch.bool, device=logits.device)
                                token_mask[completion_tokens] = True
                                
                                # Apply standard repetition penalty formula (matching transformers implementation):
                                # For tokens in completion: if score < 0 then score * penalty, else score / penalty
                                penalty_scores = torch.where(
                                    logits[i] < 0,
                                    logits[i] * penalty,
                                    logits[i] / penalty
                                )
                                # Only apply penalty to tokens that appeared in completion
                                logits[i] = torch.where(token_mask, penalty_scores, logits[i])
                
                # Prepare input_ids for sampler
                seq_input_ids = torch.tensor([seq.token_ids for seq in seqs], device=logits.device)
                
                token_ids = self.sampler(
                    logits, 
                    temperatures,
                    top_ks=top_ks if top_ks is not None else None,
                    top_ps=top_ps if top_ps is not None else None,
                    repetition_penalties=None,  # Already applied above
                    input_ids=seq_input_ids,
                ).tolist()
                return token_ids
            else:
                return None

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
