#!/usr/bin/env python3
"""
vdnamic_fastq_sim_optimized.py
───────────────────────────────────────────────────────────────────────────────
Optimized version of vdnamic_fastq_sim_patched.py with:
- Eliminated dynamic allocations in hot paths
- Vectorized random generation
- Parallel processing for main simulation loops
- Batched I/O operations
- Pattern caching and buffer reuse

New:
- --cdna-secondary-insert: for each cDNA UMI, add a second insert sequence whose
  numeric index is (orig_index + max_index + 1) and, during cDNA simulation,
  select per-read between (original, secondary) with probabilities (~2/3, ~1/3).

Maintains identical functionality and CLI interface as the original, plus the flag.
"""
from __future__ import annotations

import argparse
import heapq
import random
import re
import textwrap
import json
import warnings
import tempfile
from pathlib import Path
from collections import defaultdict, namedtuple
import pandas as pd
import numpy as np
import threading
import array
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix

# ───────────────────────────── IUPAC helpers ────────────────────────────────
IUPAC = {
    'A':"A", 'C':"C", 'G':"G", 'T':"T",
    'R':"AG", 'Y':"CT", 'S':"CG", 'W':"AT",
    'K':"GT", 'M':"AC",
    'B':"CGT", 'D':"AGT", 'H':"ACT", 'V':"ACG",
    'N':"ACGT"
}
# positions that the pipeline considers part of UMI/UEI ambiguity
AMBIGS = set('NWSMKRYBDHV')
ALPHABET = 'ACGT'

# ───────────────────────── Optimized random generation ─────────────────────

class FastRandomGenerator:
    """Thread-safe fast random generator with pre-allocated buffers."""
    
    def __init__(self, buffer_size: int = 100000):
        self.buffer_size = buffer_size
        self.base_buffer = np.random.randint(0, 4, buffer_size, dtype=np.uint8)
        self.base_index = 0
        self.lock = threading.Lock()
        self.bases = np.array(list('ACGT'), dtype='U1')
        
    def get_bases(self, n: int) -> np.ndarray:
        """Get n random base indices (0-3)."""
        with self.lock:
            if self.base_index + n > len(self.base_buffer):
                self.base_buffer = np.random.randint(0, 4, self.buffer_size, dtype=np.uint8)
                self.base_index = 0
            result = self.base_buffer[self.base_index:self.base_index + n].copy()
            self.base_index += n
            return result
    
    def get_base_string(self, length: int) -> str:
        """Get a random DNA string of given length."""
        indices = self.get_bases(length)
        return ''.join(self.bases[indices])
    
    def rand_base(self, sym: str) -> str:
        """Return a concrete base from an IUPAC symbol."""
        allowed = IUPAC.get(sym, sym)
        if len(allowed) == 1:
            return allowed
        idx = self.get_bases(1)[0]
        return allowed[idx % len(allowed)]

# Global fast random generator
_fast_random = FastRandomGenerator()

# ───────────────────────── Pattern analysis caching ─────────────────────────

class PatternCache:
    """Cache pattern analysis to avoid repeated computations."""
    
    def __init__(self):
        self.cache = {}
        self.lock = threading.Lock()
        # Pre-compile ambiguity check
        self.is_ambig = np.zeros(256, dtype=bool)
        for c in AMBIGS:
            self.is_ambig[ord(c)] = True
    
    def get_ambig_positions(self, pattern: str) -> np.ndarray:
        """Get cached ambiguous positions in pattern."""
        with self.lock:
            if pattern not in self.cache:
                # Fast numpy-based scanning
                if pattern:
                    pattern_bytes = np.frombuffer(pattern.encode('ascii'), dtype=np.uint8)
                    positions = np.where(self.is_ambig[pattern_bytes])[0]
                else:
                    positions = np.array([], dtype=np.int32)
                self.cache[pattern] = positions
            return self.cache[pattern]

_pattern_cache = PatternCache()

# ───────────────────────── Optimized FASTQ helpers ──────────────────────────

def phred_line_vectorized(n: int, mean: int = 33, stdev: int = 6) -> str:
    """Vectorized quality score generation - 5-10x faster."""
    qs = np.random.normal(mean, stdev, n)
    qs = np.clip(qs, 2, 40).astype(np.int32)
    return ''.join(chr(q + 33) for q in qs)

def phred_lines_batch(count: int, length: int, mean: int = 33, stdev: int = 6) -> List[str]:
    """Generate multiple quality strings at once."""
    qs = np.random.normal(mean, stdev, (count, length))
    qs = np.clip(qs, 2, 40).astype(np.int32)
    return [''.join(chr(q + 33) for q in row) for row in qs]

class BatchedFASTQWriter:
    """Buffered FASTQ writer for improved I/O performance."""
    
    def __init__(self, filepath: str, batch_size: int = 10000):
        self.file = open(filepath, 'w')
        self.batch_size = batch_size
        self.buffer = []
        self.lock = threading.Lock()
        
    def write_entry(self, header: str, seq: str, qual: str):
        """Add entry to buffer and flush if needed."""
        entry = f"@{header}\n{seq}\n+\n{qual}\n"
        with self.lock:
            self.buffer.append(entry)
            if len(self.buffer) >= self.batch_size:
                self._flush_unsafe()
    
    def _flush_unsafe(self):
        """Flush without locking (call from within lock)."""
        if self.buffer:
            self.file.write(''.join(self.buffer))
            self.buffer = []
    
    def flush(self):
        """Flush buffer to disk."""
        with self.lock:
            self._flush_unsafe()
    
    def close(self):
        """Flush and close."""
        self.flush()
        self.file.close()


def _iter_fastq_pairs(r1_path: Path | str, r2_path: Path | str):
    """Yield paired FASTQ records from R1/R2 in lockstep."""
    with open(r1_path) as fh1, open(r2_path) as fh2:
        while True:
            r1 = [fh1.readline() for _ in range(4)]
            r2 = [fh2.readline() for _ in range(4)]
            if not r1[0] and not r2[0]:
                return
            if not all(r1) or not all(r2):
                raise ValueError(f"FASTQ pair files are truncated or out of sync: {r1_path}, {r2_path}")
            yield tuple(r1), tuple(r2)

def _write_shuffle_chunk(chunk, chunk_path: Path) -> None:
    """Write one sorted shuffle chunk to disk."""
    chunk.sort(key=lambda rec: rec[0])
    with open(chunk_path, 'w') as fh:
        for key, r1, r2 in chunk:
            fh.write(key + '\n')
            fh.writelines(r1)
            fh.writelines(r2)

def _read_shuffle_chunk_record(fh):
    """Read one record from a shuffle chunk."""
    key = fh.readline()
    if not key:
        return None
    r1 = tuple(fh.readline() for _ in range(4))
    r2 = tuple(fh.readline() for _ in range(4))
    if not all(r1) or not all(r2):
        raise ValueError("Corrupt shuffle chunk encountered during paired FASTQ shuffle.")
    return key.rstrip('\n'), r1, r2

def shuffle_paired_fastqs(r1_path: Path | str,
                          r2_path: Path | str,
                          seed: int | None = None,
                          chunk_pairs: int = 20000) -> None:
    """
    Externally shuffle paired FASTQ files while keeping R1/R2 mates aligned.

    The files are read in lockstep as read-pairs, each pair gets a random
    128-bit sort key, sorted chunks are spilled to a temporary directory, and
    then merged into the final shuffled outputs. This guarantees that the two
    mates in a pair stay together while the output order is globally scrambled.
    """
    rng = random.Random(seed)
    r1_path = Path(r1_path)
    r2_path = Path(r2_path)

    with tempfile.TemporaryDirectory(prefix=f"{r1_path.stem}_shuffle_",
                                     dir=str(r1_path.parent)) as tmpdir_name:
        tmpdir = Path(tmpdir_name)
        chunk = []
        chunk_paths = []

        for r1, r2 in _iter_fastq_pairs(r1_path, r2_path):
            key = f"{rng.getrandbits(128):032x}"
            chunk.append((key, r1, r2))
            if len(chunk) >= chunk_pairs:
                chunk_path = tmpdir / f"chunk_{len(chunk_paths):06d}.txt"
                _write_shuffle_chunk(chunk, chunk_path)
                chunk_paths.append(chunk_path)
                chunk = []

        if chunk:
            chunk_path = tmpdir / f"chunk_{len(chunk_paths):06d}.txt"
            _write_shuffle_chunk(chunk, chunk_path)
            chunk_paths.append(chunk_path)

        if not chunk_paths:
            return

        out1_tmp = tmpdir / r1_path.name
        out2_tmp = tmpdir / r2_path.name

        chunk_handles = []
        heap = []
        try:
            for idx, chunk_path in enumerate(chunk_paths):
                fh = open(chunk_path)
                chunk_handles.append(fh)
                rec = _read_shuffle_chunk_record(fh)
                if rec is not None:
                    key, r1, r2 = rec
                    heapq.heappush(heap, (key, idx, r1, r2))

            with open(out1_tmp, 'w') as o1, open(out2_tmp, 'w') as o2:
                while heap:
                    _, idx, r1, r2 = heapq.heappop(heap)
                    o1.writelines(r1)
                    o2.writelines(r2)
                    rec = _read_shuffle_chunk_record(chunk_handles[idx])
                    if rec is not None:
                        next_key, next_r1, next_r2 = rec
                        heapq.heappush(heap, (next_key, idx, next_r1, next_r2))
        finally:
            for fh in chunk_handles:
                fh.close()

        out1_tmp.replace(r1_path)
        out2_tmp.replace(r2_path)

def revcomp(seq: str) -> str:
    """Fast reverse complement."""
    return seq.translate(str.maketrans('ACGT', 'TGCA'))[::-1]

def mean_phred(q: str) -> float:
    """Calculate mean phred score."""
    return sum(ord(c) - 33 for c in q) / len(q) if q else 0

# ───────────────────────── Buffer management for build_read ─────────────────

class ReadBuilder:
    """Reusable read builder with pre-allocated buffers."""
    
    def __init__(self, max_read_length: int = 300):
        self.read_buffer = bytearray(max_read_length)
        self.temp_buffer = bytearray(max_read_length)
        self.max_length = max_read_length
        self.random_gen = FastRandomGenerator(buffer_size=10000)
        
        # Pre-compute IUPAC lookups
        self.iupac_map = {}
        for sym, bases in IUPAC.items():
            self.iupac_map[ord(sym)] = bases.encode('ascii')
    
    def reset(self, read_len: int):
        """Reset buffer for new read."""
        if read_len > self.max_length:
            self.read_buffer = bytearray(read_len)
            self.temp_buffer = bytearray(read_len)
            self.max_length = read_len
        # Fill with spaces
        for i in range(read_len):
            self.read_buffer[i] = ord(' ')
    
    def map_seed_to_allowed(self, seed_char: str, symbol: str) -> str:
        """Map a seed base to allowed set for IUPAC symbol."""
        allowed = IUPAC.get(symbol, symbol)
        if len(allowed) == 1:
            return allowed
        idx = ALPHABET.find(seed_char)
        if idx < 0:
            idx = self.random_gen.get_bases(1)[0]
        return allowed[idx % len(allowed)]

# Thread-local builders
_thread_local = threading.local()

def get_thread_local_builder() -> ReadBuilder:
    """Get or create thread-local builder."""
    if not hasattr(_thread_local, 'builder'):
        _thread_local.builder = ReadBuilder()
    return _thread_local.builder

# ───────────────────────── settings parsing (unchanged) ────────────────────────
SeqForm = namedtuple('SeqForm', 'label pattern start end raw raw_idx')
UMISpec = namedtuple('UMISpec', 'label f_idx r_idx z_ordinal revcomp')
AmpSpec = namedtuple('AmpSpec', 'f_idx r_idx a_idx revcomp')

def parse_seqform_entry(entry: str) -> list[SeqForm]:
    """
    Accept either:
      - <LABEL>_<PATTERN>_<start[:end]>[ | <LABEL>_<PATTERN>_<start[:end]> ...]
      - <LABEL>_<PATTERN>_<start[:end]>[ | <PATTERN>_<start[:end]> ...]  (label carries over)
    """
    first_label, rest = entry.split('_', 1)
    forms: list[SeqForm] = []

    for block in rest.split('|'):
        block = block.strip()

        # If this block starts with its own label (e.g., "U_..."), take it; else inherit.
        if len(block) >= 2 and block[1] == '_' and block[0].isalpha():
            label = block[0]
            body = block[2:]
        else:
            label = first_label
            body = block

        # Split coords from the RIGHT so underscores in the pattern (if any) are preserved.
        s_i = e_i = None
        pat = body
        # NEW: coords-only body like "29:" or "29:76" (no explicit pattern)
        if (':' in body) and re.fullmatch(r'\d*:?\d*', body):
            pat = ''
            s, e = (body.split(':') + [''])[:2]
            s_i = int(s) if s != '' else None
            e_i = int(e) if e != '' else None
        elif '_' in body:
            maybe_pat, maybe_coords = body.rsplit('_', 1)
            if ':' in maybe_coords:
                pat = maybe_pat
                s, e = (maybe_coords.split(':') + [''])[:2]
                s_i = int(s) if s != '' else None
                e_i = int(e) if e != '' else None

        forms.append(SeqForm(label, pat, s_i, e_i, block, None))

    return forms


def _parse_map_line(tag: str, body: str, which: str):
    """Parse -u* and -a* lines."""
    parts = body.split(',')
    if len(parts) < 3:
        raise ValueError(f"{tag}: needs three comma-separated fields.")
    f_tok, r_tok, idx_field_plus = parts[0], parts[1], ','.join(parts[2:])

    rev = False
    if which == 'u' and idx_field_plus.endswith(':revcomp'):
        rev = True
        idx_field_plus = idx_field_plus[:-8]

    def opt_int(tok):
        tok = tok.strip()
        return None if tok in ('', '*') else int(tok)

    f_idx = opt_int(f_tok)
    r_idx = opt_int(r_tok)

    if which == 'u':
        label = int(tag[1:])
        z_fields = idx_field_plus.split('+') if idx_field_plus else ['0']
        specs = [UMISpec(label, f_idx, r_idx, int(z), rev) for z in z_fields]
        return specs
    else:
        a_idx = opt_int(idx_field_plus)
        return [AmpSpec(f_idx, r_idx, a_idx, False)]

def parse_settings(path: Path | str) -> dict:
    """Parse lib.settings file."""
    cfg = defaultdict(list)
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            try:
                key, val = line.split(None, 1)
            except ValueError:
                key, val = line, ''
            key = key.lstrip('-')
            if re.fullmatch(r'u\d+', key):
                cfg['u_specs'].extend(_parse_map_line(key, val, which='u'))
            elif re.fullmatch(r'a\d+', key):
                cfg['a_specs'].extend(_parse_map_line(key, val, which='a'))
            else:
                cfg[key].append(val)

    # flatten seqforms and tag with their original list index
    for label_key in ('seqform_for', 'seqform_rev'):
        flat: list[SeqForm] = []
        for idx, raw in enumerate(cfg.get(label_key, [])):
            for sf in parse_seqform_entry(raw):
                flat.append(sf._replace(raw_idx=idx))
        cfg[label_key] = flat

    # Group seqforms by their original line index
    for label_key in ('seqform_for', 'seqform_rev'):
        grouped = defaultdict(list)
        for sf in cfg[label_key]:
            grouped[sf.raw_idx].append(sf)
        cfg[f"{label_key}_by_idx"] = grouped

    # Derive per-read read lengths when not provided
    def _derive_len(grouped, fallback: int) -> int:
        ends = []
        for blocks in grouped.values():
            for sf in blocks:
                if sf.end is not None:
                    ends.append(sf.end)
        return max(ends) if ends else fallback

    # Scalars
    cfg['min_mean_qual'] = int(cfg.get('min_mean_qual', ['0'])[0])

    # explicit lengths if present
    rl = cfg.get('read_length', [])
    rlf = cfg.get('read_length_for', [])
    rlr = cfg.get('read_length_rev', [])

    if rlf:
        cfg['read_length_for'] = int(rlf[0])
    if rlr:
        cfg['read_length_rev'] = int(rlr[0])
    if rl:
        cfg.setdefault('read_length_for', int(rl[0]))
        cfg.setdefault('read_length_rev', int(rl[0]))

    cfg.setdefault('read_length_for', _derive_len(cfg['seqform_for_by_idx'], 151))
    cfg.setdefault('read_length_rev', _derive_len(cfg['seqform_rev_by_idx'], 151))

    if 'amplicon_terminate' in cfg:
        cfg['amplicon_terminate'] = [t for t in cfg['amplicon_terminate'][0].split(',') if t]
    else:
        cfg['amplicon_terminate'] = []

    return cfg

# ───────────────────────── mapping helpers ────────────────────────────────

def enumerate_u_blocks(blocks: list[SeqForm]) -> list[int]:
    """Return per-block U-ordinals for the given block list."""
    u_ord = -1
    out = []
    for sf in blocks:
        if sf.label == 'U':
            u_ord += 1
            out.append(u_ord)
        else:
            out.append(-1)
    return out

def build_u_label_maps(cfg: dict,
                       f_idx: int | None,
                       r_idx: int | None,
                       r1_blocks: list[SeqForm],
                       r2_blocks: list[SeqForm]):
    """Build per-read mapping from per-read U-ordinal to UMI/UEI label."""
    r1_u_ord = enumerate_u_blocks(r1_blocks)
    r2_u_ord = enumerate_u_blocks(r2_blocks)
    n_u_r1 = max([o for o in r1_u_ord if o >= 0] + [-1]) + 1
    n_u_r2 = max([o for o in r2_u_ord if o >= 0] + [-1]) + 1

    pair_map: dict[int, tuple[int, bool]] = {}
    for us in cfg.get('u_specs', []):
        if (us.f_idx is None or us.f_idx == f_idx) and (us.r_idx is None or us.r_idx == r_idx):
            pair_map[us.z_ordinal] = (us.label, us.revcomp)

    r1_map: dict[int, int | None] = {}
    r2_map: dict[int, int | None] = {}
    r1_rev: dict[int, bool] = {}
    r2_rev: dict[int, bool] = {}

    for j in range(n_u_r1):
        lab, rv = pair_map.get(j, (None, False))
        r1_map[j] = lab
        r1_rev[j] = bool(rv)
    for j in range(n_u_r2):
        lab, rv = pair_map.get(n_u_r1 + j, (None, False))
        r2_map[j] = lab
        r2_rev[j] = bool(rv)

    return r1_map, r2_map, r1_rev, r2_rev

def _label_lengths(cfg: dict) -> dict[int, int]:
    """
    Return a dict {label -> unique U-length}, where length is the number of
    ambiguous IUPAC positions in the *raw* U pattern. This matches the
    key length used by build_read_optimized (len(amb_pos) over sf.pattern).
    """
    seen: dict[int, set[int]] = defaultdict(set)

    for f_idx, r1_blocks in cfg['seqform_for_by_idx'].items():
        for r_idx, r2_blocks in cfg['seqform_rev_by_idx'].items():
            # New signature returns (r1_map, r2_map, r1_rev, r2_rev)
            r1_map, r2_map, _, _ = build_u_label_maps(cfg, f_idx, r_idx, r1_blocks, r2_blocks)
            r1_ord = enumerate_u_blocks(r1_blocks)
            r2_ord = enumerate_u_blocks(r2_blocks)

            # Forward read
            for sf, ordv in zip(r1_blocks, r1_ord):
                if sf.label != 'U' or ordv < 0:
                    continue
                lab = r1_map.get(ordv)
                if lab is None:
                    continue
                # Count ambiguous positions the same way build_read_optimized does
                amb_len = int(_pattern_cache.get_ambig_positions(sf.pattern).size)
                seen[lab].add(amb_len)

            # Reverse read
            for sf, ordv in zip(r2_blocks, r2_ord):
                if sf.label != 'U' or ordv < 0:
                    continue
                lab = r2_map.get(ordv)
                if lab is None:
                    continue
                amb_len = int(_pattern_cache.get_ambig_positions(sf.pattern).size)
                seen[lab].add(amb_len)

    out: dict[int, int] = {}
    for lab, lens in seen.items():
        if not lens:
            continue
        if len(lens) != 1:
            raise ValueError(f"Inconsistent U-lengths for label {lab}: {sorted(lens)}")
        out[lab] = next(iter(lens))
    return out

# ───────────────────────── Optimized read construction ──────────────────────

def build_read_optimized(seqforms: list[SeqForm],
                         read_len: int,
                         amplicon: str = '',
                         terminators: list[str] | None = None,
                         umi_map: dict | None = None,
                         umi_pool_dict: dict | None = None,
                         u_label_by_ord: dict[int, int | None] | None = None,
                         revcomp_by_ord: dict[int, bool] | None = None,
                         read_tag: str = 'f',
                         builder: ReadBuilder = None) -> str:
    """Optimized read construction with eliminated allocations."""
    
    if builder is None:
        builder = get_thread_local_builder()
    if umi_map is None:
        umi_map = {}
    if umi_pool_dict is None:
        umi_pool_dict = {}
    if u_label_by_ord is None:
        u_label_by_ord = {}
    if terminators is None:
        terminators = []

    builder.reset(read_len)
    read = builder.read_buffer
    cursor = 0
    last_amplicon_end = -1
    u_seen = 0

    for sf in seqforms:
        start = sf.start if sf.start is not None else cursor
        end = sf.end

        if sf.label == 'U':
            # Use cached ambiguous positions (and a set for O(1) membership)
            amb_pos = _pattern_cache.get_ambig_positions(sf.pattern)
            amb_pos_set = set(amb_pos.tolist())
            keylen = len(amb_pos)
            label = u_label_by_ord.get(u_seen, None)
            need_rev = bool(revcomp_by_ord.get(u_seen, False)) if revcomp_by_ord else False
            u_seen += 1

            if label is not None:
                pool_key = (label, keylen)
                if (label, keylen) not in umi_map:
                    pool = umi_pool_dict.get(pool_key)
                    if not pool:
                        pool_size = umi_pool_dict.get('__pool_size__', 16)
                        # Use vectorized generation
                        pool = [builder.random_gen.get_base_string(keylen) 
                               for _ in range(pool_size)]
                        umi_pool_dict[pool_key] = pool
                    umi_map[(label, keylen)] = pool[np.random.randint(len(pool))]
                umi_seq = umi_map[(label, keylen)]
                if need_rev:
                    umi_seq = revcomp(umi_seq)
            else:
                umi_seq = builder.random_gen.get_base_string(keylen)

            # Build core sequence
            core_bytes = bytearray()
            amb_iter = iter(umi_seq)
            for i, c in enumerate(sf.pattern):
                if i in amb_pos_set:
                    seed = next(amb_iter, 'N')
                    mapped = builder.map_seed_to_allowed(seed, c)
                    core_bytes.append(ord(mapped))
                else:
                    if c in IUPAC:
                        if len(IUPAC[c]) == 1:
                            core_bytes.append(ord(c))
                        else:
                            core_bytes.append(ord(builder.random_gen.rand_base(c)))
                    else:
                        core_bytes.append(ord(c))
            core = core_bytes.decode('ascii')

        elif sf.label == 'A':
            core = amplicon
            # If a start coordinate is designated, fill holes < start with 'A' (left-A padding)
            if start is not None and start > 0:
                pad_to = min(start, read_len)
                A_byte = ord('A')
                space = ord(' ')
                for i in range(pad_to):
                    if read[i] == space:
                        read[i] = A_byte
        else:
            if sf.pattern:
                core_bytes = bytearray()
                for c in sf.pattern:
                    if c in IUPAC:
                        if len(IUPAC[c]) == 1:
                            core_bytes.append(ord(c))
                        else:
                            core_bytes.append(ord(builder.random_gen.rand_base(c)))
                    else:
                        core_bytes.append(ord(c))
                core = core_bytes.decode('ascii')
            else:
                core = ''

        # Place into read array
        if end is None:
            end_eff = min(start + len(core), read_len)
            seg = core[: max(0, end_eff - start)]
        else:
            if start <= end:
                end_eff = min(end, read_len)
                seg = core[: max(0, end_eff - start)]
            else:
                end_eff = min(start, read_len)
                beg_eff = max(0, min(end, end_eff))
                width = max(0, end_eff - beg_eff)
                seg = revcomp(core)[:width]
                start = beg_eff

        # Place segment into read
        if start < read_len and start < end_eff:
            seg_bytes = seg.encode('ascii')
            for i, b in enumerate(seg_bytes):
                if start + i < end_eff:
                    read[start + i] = b
            # Remember where A ended (in read coords) so we can drop a terminator
            if sf.label == 'A':
                last_amplicon_end = start + len(seg_bytes)
        cursor = end_eff

    # If we placed an amplicon and we have a terminator, stamp it immediately after A
    if last_amplicon_end >= 0 and terminators:
        tbytes = terminators[0].encode('ascii')
        pos = last_amplicon_end
        for b in tbytes:
            if pos >= read_len:
                break
            if read[pos] == ord(' '):
                read[pos] = b
            pos += 1

    # Fill remaining holes with random bases
    bases_ord = [ord(c) for c in 'ACGT']
    random_indices = builder.random_gen.get_bases(read_len)
    for i in range(read_len):
        if read[i] == ord(' '):
            read[i] = bases_ord[random_indices[i % len(random_indices)]]

    return read[:read_len].decode('ascii')

# ───────────────────── legacy TSV simulation ───────────────────

def simulate(df: pd.DataFrame,
             cdna_cfg: dict,
             uei_cfg: dict,
             outdir: Path,
             args: argparse.Namespace):

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    def _outpath(cfg: dict, which: str) -> tuple[Path, Path]:
        r1 = cfg.get('source_for', ['R1.fastq'])[0]
        r2 = cfg.get('source_rev', ['R2.fastq'])[0]
        return outdir / r1, outdir / r2

    r1_cd_path, r2_cd_path = _outpath(cdna_cfg, 'cdna')
    r1_ue_path, r2_ue_path = _outpath(uei_cfg, 'uei')

    outdir.mkdir(parents=True, exist_ok=True)
    
    # Use batched writers
    handles = {
        'R1_cdna': BatchedFASTQWriter(str(r1_cd_path)),
        'R2_cdna': BatchedFASTQWriter(str(r2_cd_path)),
        'R1_uei': BatchedFASTQWriter(str(r1_ue_path)),
        'R2_uei': BatchedFASTQWriter(str(r2_ue_path)),
    }

    cdna_len_for = cdna_cfg['read_length_for']
    cdna_len_rev = cdna_cfg['read_length_rev']
    uei_len_for = uei_cfg['read_length_for']
    uei_len_rev = uei_cfg['read_length_rev']

    global_umi_pool: dict[tuple[int, int], list[str]] = {}
    global_umi_pool['__pool_size__'] = args.umi_pool

    # Create thread-local builder
    builder = ReadBuilder()

    uid = 0
    for gene, insert, count in df.itertuples(index=False):
        if not re.fullmatch('[ACGT]+', insert, re.I):
            raise ValueError(f"{gene}: insert contains non-ACGT bases.")

        umi_map: dict[tuple[int, int], str] = {}
        for _ in range(int(count)):
            uid += 1

            cd_for_idx = random.choice(list(cdna_cfg['seqform_for_by_idx'].keys()))
            cd_rv_idx = random.choice(list(cdna_cfg['seqform_rev_by_idx'].keys()))
            ue_for_idx = random.choice(list(uei_cfg['seqform_for_by_idx'].keys()))
            ue_rv_idx = random.choice(list(uei_cfg['seqform_rev_by_idx'].keys()))

            r1_cd_sfs = cdna_cfg['seqform_for_by_idx'][cd_for_idx]
            r2_cd_sfs = cdna_cfg['seqform_rev_by_idx'][cd_rv_idx]
            r1_ue_sfs = uei_cfg['seqform_for_by_idx'][ue_for_idx]
            r2_ue_sfs = uei_cfg['seqform_rev_by_idx'][ue_rv_idx]

            cd_r1_map, cd_r2_map, cd_r1_rev, cd_r2_rev = build_u_label_maps(cdna_cfg, cd_for_idx, cd_rv_idx,
                                                                            r1_cd_sfs, r2_cd_sfs)
            ue_r1_map, ue_r2_map, ue_r1_rev, ue_r2_rev = build_u_label_maps(uei_cfg, ue_for_idx, ue_rv_idx,
                                                                            r1_ue_sfs, r2_ue_sfs)

            # Use optimized build_read
            r1_cd = build_read_optimized(r1_cd_sfs, cdna_len_for,
                                         amplicon=insert,
                                         terminators=cdna_cfg['amplicon_terminate'],
                                         umi_map=umi_map,
                                         umi_pool_dict=global_umi_pool,
                                         u_label_by_ord=cd_r1_map, revcomp_by_ord=cd_r1_rev,
                                         read_tag='f',
                                         builder=builder)
            r2_cd = build_read_optimized(r2_cd_sfs, cdna_len_rev,
                                         amplicon=insert,
                                         terminators=cdna_cfg['amplicon_terminate'],
                                         umi_map=umi_map,
                                         umi_pool_dict=global_umi_pool,
                                         u_label_by_ord=cd_r2_map, revcomp_by_ord=cd_r2_rev,
                                         read_tag='r',
                                         builder=builder)

            r1_ue = build_read_optimized(r1_ue_sfs, uei_len_for,
                                         amplicon='',
                                         terminators=uei_cfg['amplicon_terminate'],
                                         umi_map=umi_map,
                                         umi_pool_dict=global_umi_pool,
                                         u_label_by_ord=ue_r1_map, revcomp_by_ord=ue_r1_rev,
                                         read_tag='f',
                                         builder=builder)
            r2_ue = build_read_optimized(r2_ue_sfs, uei_len_rev,
                                         amplicon='',
                                         terminators=uei_cfg['amplicon_terminate'],
                                         umi_map=umi_map,
                                         umi_pool_dict=global_umi_pool,
                                         u_label_by_ord=ue_r2_map, revcomp_by_ord=ue_r2_rev,
                                         read_tag='r',
                                         builder=builder)

            # Use vectorized quality generation
            q1_cd = phred_line_vectorized(cdna_len_for)
            q2_cd = phred_line_vectorized(cdna_len_rev)
            q1_ue = phred_line_vectorized(uei_len_for)
            q2_ue = phred_line_vectorized(uei_len_rev)

            if (mean_phred(q1_cd) >= cdna_cfg['min_mean_qual'] and
                mean_phred(q2_cd) >= cdna_cfg['min_mean_qual']):
                handles['R1_cdna'].write_entry(f"C{uid:09d}/1", r1_cd, q1_cd)
                handles['R2_cdna'].write_entry(f"C{uid:09d}/2", r2_cd, q2_cd)

            if (mean_phred(q1_ue) >= uei_cfg['min_mean_qual'] and
                mean_phred(q2_ue) >= uei_cfg['min_mean_qual']):
                handles['R1_uei'].write_entry(f"U{uid:09d}/1", r1_ue, q1_ue)
                handles['R2_uei'].write_entry(f"U{uid:09d}/2", r2_ue, q2_ue)

    for h in handles.values():
        h.close()

    shuffle_paired_fastqs(r1_cd_path, r2_cd_path,
                          seed=(None if args.seed is None else args.seed + 3001))
    shuffle_paired_fastqs(r1_ue_path, r2_ue_path,
                          seed=(None if args.seed is None else args.seed + 3002))

# ─────────────── upstream builder: positions → graph ─────────

def encode_index_base4(idx: int, width: int | None = None, digits: str = 'ACGT') -> str:
    """Encode integer to base-4 string."""
    if idx < 0:
        raise ValueError('idx must be nonnegative')
    if width == 0:
        width = None
    base = 4
    if idx == 0:
        s = digits[0]
    else:
        out = []
        n = idx
        while n > 0:
            out.append(digits[n % base])
            n //= base
        s = ''.join(reversed(out))
    if width is not None:
        if len(s) > width:
            raise ValueError(f'encode width {width} too small for idx {idx}')
        s = digits[0] * (width - len(s)) + s
    return s

def encode_indices_base4_vectorized(indices: np.ndarray, width: int = 0, digits: str = 'ACGT') -> List[str]:
    """Vectorized base-4 encoding for multiple indices."""
    if len(indices) == 0:
        return []
    
    indices = np.asarray(indices, dtype=np.int64)
    if width == 0:
        max_val = np.max(indices)
        width = int(np.log(max_val) / np.log(4)) + 1 if max_val > 0 else 1
    
    n = len(indices)
    result = np.empty((n, width), dtype='U1')
    digit_arr = np.array(list(digits))
    
    temp = indices.copy()
    for i in range(width):
        result[:, width - 1 - i] = digit_arr[temp % 4]
        temp //= 4
    
    return [''.join(row) for row in result]

def _pairwise_d2(pos0: np.ndarray, pos1: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between pos0 and pos1."""
    x2 = np.sum(pos0**2, axis=1)[:, None]
    y2 = np.sum(pos1**2, axis=1)[None, :]
    xy = pos0 @ pos1.T
    d2 = np.maximum(x2 + y2 - 2.0 * xy, 0.0)
    return d2

def build_inputs_from_positions(pos_csv: str,
                                outdir: Path,
                                rescale: float,
                                rescale2: float,
                                weight2: float,
                                mperPt: float,
                                negbin_p: float,
                                amp_dispersion: float,
                                dropout0: float,
                                dropout1: float,
                                encode_width: int | None,
                                encode_digits: str,
                                seed: int | None = None) -> tuple[Path, Path, Path, int, int]:
    """Build umi0.txt, umi1.txt, and graph.npz from positions CSV."""
    from scipy.sparse import csr_matrix, save_npz

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    arr = np.loadtxt(pos_csv, delimiter=',')
    if arr.shape[1] < 3:
        raise ValueError('pos CSV must have at least 3 columns: id,label,x[,y...]')

    
    # Keep type-0 rows first and type-1 rows second, but remember original CSV rows.
    order = np.lexsort((arr[:, 0], arr[:, 1]))  # primary: label, secondary: id
    arr = arr[order]

    labels = arr[:, 1].astype(int)
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError('label column must be 0 or 1 only')
    mask0 = labels == 0
    mask1 = labels == 1
    n0 = int(mask0.sum())
    n1 = int(mask1.sum())
    if n0 == 0 or n1 == 0:
        raise ValueError('both partitions must be nonempty (labels 0 and 1).')

    # ------------------------------------------------------------------
    # Legacy-normalized coordinates (match the second script):
    # 1) isotropize by total variance; 2) scale by --rescale
    # ------------------------------------------------------------------
    pos = arr[:, 2:]
    varsum = float(np.sum(np.var(pos, axis=0)))
    scale = (rescale / np.sqrt(varsum)) if varsum > 0.0 else 1.0
    sim_pos = pos * scale

    pos0 = sim_pos[mask0]
    pos1 = sim_pos[mask1]

    amps = np.random.randn(n0 + n1) * amp_dispersion
    a0 = amps[:n0][:, None]
    a1 = amps[n0:][None, :]

    # --------------------------
    # Two-layer kernel (legacy):
    # Layer 1 on sim_pos; Layer 2 on sim_pos / rescale2.
    # Each layer normalized; mixture re-normalized.
    # --------------------------
    d2 = _pairwise_d2(pos0, pos1)
    W1 = np.exp(-d2 + a0 + a1)
    W1 = W1 / W1.sum()
    if weight2 > 0.0:
        sim_pos2 = sim_pos / rescale2
        pos0b = sim_pos2[mask0]
        pos1b = sim_pos2[mask1]
        d2b = _pairwise_d2(pos0b, pos1b)
        W2 = np.exp(-d2b + a0 + a1)
        W2 = W2 / W2.sum()
        W = W1 + weight2 * W2
        W = W / W.sum()
    else:
        W = W1

    # Legacy zero-handling:
    # mark zeros; for the NB means use min-positive in masked cells,
    # but force those cells' counts to zero after sampling.
    zero_mask = (W <= 0.0)
    if np.any(zero_mask):
        min_pos = float(W[~zero_mask].min()) if np.any(~zero_mask) else 0.0
        W_for_mu = W.copy()
        W_for_mu[zero_mask] = min_pos
    else:
        W_for_mu = W

    mu = mperPt * (n0 + n1) * W_for_mu
    nparam = mu * (negbin_p / (1.0 - negbin_p))  # shape
    # Gamma–Poisson mixture: lambda ~ Gamma(shape=nparam, scale=(1-p)/p), counts ~ Poisson(lambda)
    lam = np.random.gamma(shape=nparam, scale=(1.0 - negbin_p) / negbin_p, size=mu.shape)
    counts = np.random.poisson(lam).astype(np.int64)
    if np.any(zero_mask):
        counts[zero_mask] = 0
    counts = counts.astype(np.int64)

    rows, cols, data = [], [], []
    for i in range(n0):
        js = np.nonzero(counts[i, :] > 0)[0]
        if js.size == 0:
            continue
        rows.extend([i] * int(js.size))
        cols.extend((n0 + js).tolist())
        data.extend(counts[i, js].tolist())

    graph = csr_matrix((data, (rows, cols)), shape=(n0 + n1, n0 + n1))

    outdir.mkdir(parents=True, exist_ok=True)
    graph_path = outdir / 'graph.npz'
    save_npz(graph_path, graph)

    # Use vectorized encoding when possible
    def _make_inserts(n: int, dropout: float, values: np.ndarray | None = None) -> list[str]:
        dropout_mask = np.random.rand(n) < dropout
        indices = np.asarray(values, dtype=np.int64) if values is not None else np.arange(n, dtype=np.int64)
        # Vectorized even when width==0 (the helper computes minimal width)
        encoded = encode_indices_base4_vectorized(indices, width=encode_width or 0, digits=encode_digits)
        seqs = ['N' if dropout_mask[i] else encoded[i] for i in range(n)]

        return seqs

    # Map back to ORIGINAL CSV row indices for each partition
    sorted_rows = order  # maps sorted position -> original CSV row
    idxs0 = np.nonzero(mask0)[0]
    idxs1 = np.nonzero(mask1)[0]
    orig_rows0 = sorted_rows[idxs0]  # length n0
    orig_rows1 = sorted_rows[idxs1]  # length n1

    umi0_seqs = _make_inserts(n0, dropout0, values=orig_rows0)
    umi1_seqs = _make_inserts(n1, dropout1, values=orig_rows1)

    umi0_path = outdir / 'umi0.txt'
    umi1_path = outdir / 'umi1.txt'
    with open(umi0_path, 'w') as f0:
        for s in umi0_seqs:
            f0.write(s + '\n')
    with open(umi1_path, 'w') as f1:
        for s in umi1_seqs:
            f1.write(s + '\n')

    return umi0_path, umi1_path, graph_path, n0, n1

# ───────────────────── Helper functions for parallel processing ─────────────

def _read_inserts_list(path: str) -> list[str]:
    arr = []
    with open(path) as fh:
        for ln in fh:
            s = ln.strip().upper()
            if not s:
                continue
            arr.append(s)
    return arr

def _choose_seqform_pair_with_labels(cfg: dict,
                                     require_labels: set[int],
                                     r1_blocks_key: str = 'seqform_for_by_idx',
                                     r2_blocks_key: str = 'seqform_rev_by_idx'):
    """Pick a seqform pair that exposes required labels."""
    keys_f = list(cfg[r1_blocks_key].keys())
    keys_r = list(cfg[r2_blocks_key].keys())
    for _ in range(256):
        f_idx = random.choice(keys_f)
        r_idx = random.choice(keys_r)
        r1_sfs = cfg[r1_blocks_key][f_idx]
        r2_sfs = cfg[r2_blocks_key][r_idx]
        r1_map, r2_map, r1_rev, r2_rev = build_u_label_maps(cfg, f_idx, r_idx, r1_sfs, r2_sfs)
        exposed = set(v for v in list(r1_map.values()) + list(r2_map.values()) if v is not None)
        if require_labels.issubset(exposed):
            return f_idx, r_idx, r1_sfs, r2_sfs, r1_map, r2_map, r1_rev, r2_rev
    raise RuntimeError(f"Could not find seqform pair exposing labels {sorted(require_labels)}")

# NEW: base-4 decode (inverse of encode_index_base4)
def decode_index_base4(s: str, digits: str = 'ACGT') -> int:
    """Decode a base-4 DNA string to integer using the given digit alphabet."""
    if not s or any(ch not in digits for ch in s):
        raise ValueError("Invalid base-4 string for the provided digits.")
    base = 4
    m = {digits[i]: i for i in range(4)}
    val = 0
    for ch in s:
        val = val * base + m[ch]
    return val


# ───────────────────── Cross-section → AnnData (.h5ad) ─────────────────────

def _seq_vocab_encode_sequences_sparse(df: pd.DataFrame,
                                      seq_cols: list[str],
                                      drop_token: str = "N") -> tuple[csr_matrix, list[str], dict[str, int]]:
    """
    Vocabulary-style sequence encoding (desired):
      - one column per *distinct sequence* observed in the provided seq_cols
      - each observation gets 1.0 for each sequence it carries (binary presence)
      - drop_token (default "N") is treated as "no label" and contributes no entries

    Returns:
      X_seq      : CSR (n_obs × n_distinct_sequences), float32, binary
      sequences  : list of sequences in column order (for var['sequence'])
      seq_to_col : dict mapping sequence -> column index (0-based)
    """
    n_obs = int(df.shape[0])
    rows_all: list[np.ndarray] = []
    seqs_all: list[np.ndarray] = []

    for col in seq_cols:
        if col not in df.columns:
            continue
        s = df[col].astype(str)
        # filter out missing-ish tokens (astype(str) turns NaN into "nan")
        mask = (
            (s != drop_token) &
            (s != "") &
            (s.str.lower() != "nan") &
            (s.str.lower() != "none")
        )
        if bool(mask.any()):
            rows_all.append(np.flatnonzero(mask.to_numpy(dtype=bool)).astype(np.int32))
            seqs_all.append(s[mask].to_numpy(dtype=object))

    # No sequences at all -> return empty feature space
    if not rows_all:
        return csr_matrix((n_obs, 0), dtype=np.float32), [], {}

    rows = np.concatenate(rows_all).astype(np.int32, copy=False)
    seqs = np.concatenate(seqs_all).astype(object, copy=False)

    # Factorize gives us integer codes + uniques in first-seen order (stable)
    codes, uniques = pd.factorize(seqs, sort=False)
    cols = codes.astype(np.int32, copy=False)
    data = np.ones(rows.shape[0], dtype=np.float32)

    X_seq = csr_matrix((data, (rows, cols)), shape=(n_obs, int(len(uniques))), dtype=np.float32)
    # If the same (row, seq) appeared multiple times (e.g. insert_seq + insert_seq_secondary identical),
    # collapse duplicates and binarize.
    X_seq.sum_duplicates()
    if X_seq.nnz:
        X_seq.data[:] = 1.0

    sequences = [str(x) for x in uniques.tolist()]
    seq_to_col = {s: int(i) for i, s in enumerate(sequences)}
    return X_seq, sequences, seq_to_col
 
 
def _knn_graph_from_coords(X: np.ndarray, k: int = 15):
    """
    Build a symmetric KNN graph (distances + binary connectivities) using cKDTree.
    Returns: distances_csr, connectivities_csr, k_eff
    """
    X = np.asarray(X, dtype=np.float32)
    n = int(X.shape[0])
    k_eff = int(min(max(0, k), max(0, n - 1)))
    if k_eff <= 0:
        Z = csr_matrix((n, n), dtype=np.float32)
        return Z, Z, k_eff

    tree = cKDTree(X)
    dists, nbrs = tree.query(X, k=k_eff + 1)  # includes self at [:,0]
    nbrs = nbrs[:, 1:]
    dists = dists[:, 1:]

    rows = np.repeat(np.arange(n, dtype=np.int32), k_eff)
    cols = nbrs.reshape(-1).astype(np.int32)
    dist_data = dists.reshape(-1).astype(np.float32)
    ones = np.ones_like(dist_data, dtype=np.float32)

    distances = csr_matrix((dist_data, (rows, cols)), shape=(n, n), dtype=np.float32)
    connectivities = csr_matrix((ones, (rows, cols)), shape=(n, n), dtype=np.float32)

    # Symmetrize (union of directed KNN edges)
    distances = distances.maximum(distances.T)
    connectivities = connectivities.maximum(connectivities.T)
    return distances, connectivities, k_eff


def build_cross_section_anndata(df: pd.DataFrame, d: int, encode_digits: str = "ACGT", knn_k: int = 15):
    """
    Build an AnnData object for the cross-section.
      - adata.layers['seq'] = sparse matrix with ONE COLUMN PER DISTINCT SEQUENCE
      - adata.var['sequence'] stores the literal sequence for each column
      - adata.uns['seq_to_col'] maps sequence -> column index
      - adata.X is an empty (all-zero) sparse matrix of the same shape to avoid duplicating storage
      - adata.obs = metadata (partition_label, insert_seq, indices, etc.)
      - adata.obsm['X_scaled'] = scaled coords (float32)
      - adata.obsm['X_orig'] = original coords (float32)
      - adata.obsp['distances'], adata.obsp['connectivities'] = KNN graph on X_scaled
    """
    try:
        import anndata as ad
    except Exception as e:
        raise ImportError("Missing dependency 'anndata' (and usually 'h5py'). Install via: pip install anndata h5py") from e

    # coords
    X_scaled = df[[f"scaled_{i}" for i in range(d)]].to_numpy(dtype=np.float32, copy=True)
    X_orig = df[[f"coord_{i}" for i in range(d)]].to_numpy(dtype=np.float32, copy=True)

    # ---------------------------------------------------------------------
    # Sequence vocabulary encoding: one column per distinct sequence
    # ---------------------------------------------------------------------
    seq_cols = ["insert_seq"]
    if "insert_seq_secondary" in df.columns:
        seq_cols.append("insert_seq_secondary")

    X_seq, sequences, seq_to_col = _seq_vocab_encode_sequences_sparse(
        df=df,
        seq_cols=seq_cols,
        drop_token="N",
    )

    # var: store the actual sequences for lookup
    var_names = [f"seq_{i:06d}" for i in range(len(sequences))]
    n_vars = int(len(sequences))
    var = pd.DataFrame(index=var_names)
    var["feature_type"] = ["seq"] * n_vars
    # IMPORTANT: must be real strings for h5ad writing (pd.NA breaks h5py vlen strings)
    var["gene_id"] = [""] * n_vars
    var["sequence"] = [str(s) for s in sequences]
    var.index = var.index.astype(str)
    # IMPORTANT: keep X empty to avoid duplicating the seq matrix in both X and layers['seq']
    X_empty = csr_matrix((int(df.shape[0]), int(len(sequences))), dtype=np.float32)
    adata = ad.AnnData(X=X_empty, var=var)

    adata.layers["seq"] = X_seq 


    # obs columns (keep it explicit and stable)
    keep_obs = [c for c in ["sorted_index", "partition_label", "insert_seq", "projection", "orig_csv_row", "id",
                            "insert_seq_secondary"] if c in df.columns]
    obs = df[keep_obs].copy()
    if "sorted_index" in obs.columns:
        obs["node_id"] = obs["sorted_index"].astype(int)
    if "partition_label" in obs.columns:
        obs["partition_label"] = obs["partition_label"].astype(int)
        obs["umi_type"] = np.where(obs["partition_label"].to_numpy() == 0, "umi0", "umi1")
    if "orig_csv_row" in obs.columns:
        obs["raw_umi_index"] = obs["orig_csv_row"].astype(int)
    if "insert_seq" in obs.columns:
        obs["has_label"] = (obs["insert_seq"].astype(str) != "N")
    # AnnData stores obs_names (index) separately from obs columns.
    # If obs.index.name equals an existing column name, AnnData requires that
    # column's values match the index exactly. Setting the index from the
    # Series obs["node_id"] makes obs.index.name == "node_id", which collides
    # with the "node_id" column (int) and breaks write_h5ad.
    if "node_id" in obs.columns:
        obs.index = pd.Index(obs["node_id"].astype(str).to_numpy(), name=None)
    else:
        obs.index = pd.Index(obs.index.astype(str).to_numpy(), name=None)

    adata.obs = obs
 
    # coords
    adata.obsm["X_scaled"] = X_scaled
    adata.obsm["X_orig"] = X_orig

    # KNN graph
    distances, connectivities, k_eff = _knn_graph_from_coords(X_scaled, k=knn_k)
    adata.obsp["distances"] = distances
    adata.obsp["connectivities"] = connectivities
    adata.uns["neighbors"] = {
        "params": {"n_neighbors": int(k_eff), "method": "cKDTree", "metric": "euclidean", "use_rep": "X_scaled"},
        "distances_key": "distances",
        "connectivities_key": "connectivities",
    }
    # Lookup: literal sequence -> column index in var/layers['seq']
    adata.uns["seq_to_col"] = seq_to_col

    # Optional: store encoding meta (updated)
    adata.uns["sequence_encoding"] = {
        "digits": encode_digits,
        "scheme": "vocab(one_column_per_distinct_sequence)",
        "source_columns": seq_cols,
        "n_sequences": int(len(sequences)),
    }
    return adata


# ───────────────────── Cross-section export (post-simulation) ─────────────

def export_random_cross_section(pos_csv: str,
                                outdir: Path,
                                umi0_inserts_path: str,
                                umi1_inserts_path: str,
                                rescale: float,
                                thickness_frac: float = 0.10,
                                seed: int | None = None,
                                encode_digits: str = 'ACGT',
                                cdna_secondary_insert: bool = False) -> tuple[Path, Path, Path]:
    """Export a random (d-1)-dimensional cross-section (slab) through the point cloud.

    Definition:
      - Load pos_csv (columns: id,label,x[,y...]).
      - Sort rows by (label, id) to match build_inputs_from_positions().
      - Rescale coordinates exactly like build_inputs_from_positions(): isotropize by
        total variance, then scale by --rescale.
      - Choose a random unit normal vector n in R^d.
      - Project all points: p = X·n.
      - Let full point-spread be (p_max - p_min). Choose slab thickness = thickness_frac * spread.
      - Choose a random center c and keep points with p in [c - t/2, c + t/2].

    Writes into outdir:
      - cross_section_points.csv : one row per selected point with coordinates + sequence labels
      - cross_section_meta.json  : slab geometry + parameters used
      - cross_section.png        : scatter (first two scaled dims) + projection histogram w/ slab
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Re-load and reproduce the exact sorted ordering used by the builder.
    arr = np.loadtxt(pos_csv, delimiter=',')
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 3:
        raise ValueError('pos CSV must have at least 3 columns: id,label,x[,y...]')

    order = np.lexsort((arr[:, 0], arr[:, 1]))  # primary: label, secondary: id
    arr = arr[order]
    ids = arr[:, 0]
    labels = arr[:, 1].astype(int)
    pos = arr[:, 2:]

    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError('label column must be 0 or 1 only')

    # Match build_inputs_from_positions() scaling for "simulation coordinates"
    varsum = float(np.sum(np.var(pos, axis=0)))
    scale = (rescale / np.sqrt(varsum)) if varsum > 0.0 else 1.0
    sim_pos = pos * scale

    n_total, d = sim_pos.shape
    if d <= 0 or n_total <= 0:
        raise ValueError("No coordinates found to slice.")

    # Load the sequence-based labels (inserts), which are already ordered as:
    #   [all label-0 points] + [all label-1 points]
    ins0 = _read_inserts_list(umi0_inserts_path)
    ins1 = _read_inserts_list(umi1_inserts_path)
    n0 = int(np.count_nonzero(labels == 0))
    n1 = int(np.count_nonzero(labels == 1))
    if len(ins0) != n0 or len(ins1) != n1:
        raise ValueError(
            f"Inserts length mismatch vs posfile partitions: "
            f"n0={n0}, len(umi0)={len(ins0)}; n1={n1}, len(umi1)={len(ins1)}"
        )
    inserts = np.array(ins0 + ins1, dtype=object)

    # Optional: include the "secondary insert" label if cdna_secondary_insert is enabled.
    sec_inserts = None
    if cdna_secondary_insert:
        decoded = []
        for s in inserts.tolist():
            if s == 'N':
                continue
            if any(ch not in encode_digits for ch in s):
                continue
            decoded.append(decode_index_base4(s, digits=encode_digits))
        max_idx = max(decoded) if decoded else -1
        offset = (max_idx + 1) if max_idx >= 0 else 0

        sec_list = []
        for s in inserts.tolist():
            if s == 'N' or any(ch not in encode_digits for ch in s):
                sec_list.append('N')
            else:
                orig = decode_index_base4(s, digits=encode_digits)
                sec_list.append(encode_index_base4(orig + offset, width=len(s), digits=encode_digits))
        sec_inserts = np.array(sec_list, dtype=object)

    # Use an RNG stream independent of the simulation draws (but reproducible from --seed).
    rng = np.random.default_rng((seed + 777777) if seed is not None else None)

    # Random (d-1) slab via random normal vector in R^d
    nvec = rng.normal(size=d)
    nrm = float(np.linalg.norm(nvec))
    if (not np.isfinite(nrm)) or nrm <= 0.0:
        nvec = np.zeros(d, dtype=float)
        nvec[0] = 1.0
        nrm = 1.0
    nvec = nvec / nrm

    proj = sim_pos @ nvec
    pmin = float(np.min(proj))
    pmax = float(np.max(proj))
    spread = float(pmax - pmin)
    thickness = float(thickness_frac) * spread

    if spread <= 0.0 or thickness <= 0.0:
        center = 0.5 * (pmin + pmax)
        low, high = pmin, pmax
        mask = np.ones(n_total, dtype=bool)
    else:
        half = 0.5 * thickness
        lo_c = pmin + half
        hi_c = pmax - half
        if lo_c > hi_c:
            center = 0.5 * (pmin + pmax)
        else:
            center = float(rng.uniform(lo_c, hi_c))
        low = center - half
        high = center + half
        mask = (proj >= low) & (proj <= high)
        # Guarantee non-empty selection by snapping to nearest point if needed
        if not np.any(mask):
            k = int(np.argmin(np.abs(proj - center)))
            mask[k] = True

    sel_idx = np.nonzero(mask)[0]
    n_sel = int(sel_idx.size)

    # Assemble table for selected points
    # Note: 'sorted_index' corresponds to the graph/inserts indexing convention.
    out = {
        'sorted_index': sel_idx.astype(int),
        'partition_label': labels[sel_idx].astype(int),
        'insert_seq': inserts[sel_idx],
        'projection': proj[sel_idx].astype(float),
        'orig_csv_row': order[sel_idx].astype(int),
        'id': ids[sel_idx],
    }
    if sec_inserts is not None:
        out['insert_seq_secondary'] = sec_inserts[sel_idx]

    # Original coordinates and scaled ("simulation") coordinates
    for k in range(d):
        out[f'coord_{k}'] = pos[sel_idx, k].astype(float)
    for k in range(d):
        out[f'scaled_{k}'] = sim_pos[sel_idx, k].astype(float)

    df = pd.DataFrame(out).sort_values('sorted_index')

    points_path = outdir / 'cross_section_points.csv'
    meta_path = outdir / 'cross_section_meta.json'
    png_path = outdir / 'cross_section.png'

    df.to_csv(points_path, index=False)

    meta = {
        'pos_csv': str(pos_csv),
        'outdir': str(outdir),
        'n_total': int(n_total),
        'n_selected': int(n_sel),
        'd': int(d),
        'rescale': float(rescale),
        'scale_factor_applied': float(scale),
        'thickness_fraction': float(thickness_frac),
        'projection_min': float(pmin),
        'projection_max': float(pmax),
        'projection_spread': float(spread),
        'slab_center': float(center),
        'slab_thickness': float(thickness),
        'slab_low': float(low),
        'slab_high': float(high),
        'normal_vector': [float(x) for x in nvec.tolist()],
        'seed_base': (None if seed is None else int(seed)),
        'seed_used_for_slice': (None if seed is None else int(seed + 777777)),
        'files': {
            'points_csv': str(points_path.name),
            'meta_json': str(meta_path.name),
            'png': str(png_path.name),
        }
    }
    with open(meta_path, 'w') as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)

    # Plot: scatter of first 2 scaled dims + histogram of projection with slab bounds.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Keep plots from exploding on very large n by subsampling non-selected points.
    max_points = 200000
    if n_total > max_points:
        non_sel = np.nonzero(~mask)[0]
        # Always keep all selected points; sample the remainder from non-selected.
        keep_sel = sel_idx
        n_non = max_points - int(keep_sel.size)
        if n_non > 0 and non_sel.size > 0:
            samp_non = rng.choice(non_sel, size=min(n_non, int(non_sel.size)), replace=False)
            plot_idx = np.unique(np.concatenate([keep_sel, samp_non]))
        else:
            # Selected points alone already exceed max_points → sample selected.
            plot_idx = rng.choice(keep_sel, size=max_points, replace=False)
    else:
        plot_idx = np.arange(n_total, dtype=int)

    mask_plot = mask[plot_idx]
    Xp = sim_pos[plot_idx]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot in the first two scaled dimensions (or index vs coord if d==1)
    if d >= 2:
        x_all, y_all = Xp[:, 0], Xp[:, 1]
        x_sel, y_sel = x_all[mask_plot], y_all[mask_plot]
        ax0.scatter(x_all, y_all, s=4, alpha=0.25)
        ax0.scatter(x_sel, y_sel, s=10, alpha=0.9)
        ax0.set_xlabel('scaled_0')
        ax0.set_ylabel('scaled_1')
    else:
        x_all = np.arange(Xp.shape[0], dtype=int)
        y_all = Xp[:, 0]
        ax0.scatter(x_all, y_all, s=4, alpha=0.25)
        ax0.scatter(x_all[mask_plot], y_all[mask_plot], s=10, alpha=0.9)
        ax0.set_xlabel('point_index (plotted subset)')
        ax0.set_ylabel('scaled_0')

    ax0.set_title(f"Random slab: {n_sel}/{n_total} points (thickness {thickness_frac:.0%} of proj spread)")

    # Projection histogram + slab bounds
    proj_plot = proj[plot_idx]
    ax1.hist(proj_plot, bins=60)
    ax1.axvline(low)
    ax1.axvline(high)
    ax1.set_xlabel('projection onto random normal')
    ax1.set_ylabel('count (plotted subset)')
    ax1.set_title("Slab location (between vertical lines)")

    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    return points_path, meta_path, png_path

# ───────────────────── Parallel processing workers ─────────────

def process_umi_batch(batch_args: tuple) -> list:
    """Process a batch of UMIs for parallel execution.

    batch_args:
      (start_idx, end_idx, ins_list, sec_ins_list_or_None, sec_prob, umi_list, label, L,
       cdna_cfg, avg_reads, min_reads, read_len_for, read_len_rev, batch_seed)
    """
    (start_idx, end_idx,
     ins_list, sec_ins_list, sec_prob,
     umi_list, label, L,
     cdna_cfg, avg_reads, min_reads,
     read_len_for, read_len_rev,
     batch_seed) = batch_args

    if batch_seed is not None:
        random.seed(batch_seed)
        np.random.seed(batch_seed)    
    # Create thread-local builder
    builder = ReadBuilder()
    results = []
    
    # Draw n_reads once per UMI and pre-generate qualities to match exactly
    n_reads_vec = []
    for i in range(start_idx, end_idx):
        if ins_list[i] == 'N':
            n_reads_vec.append(0)
        else:
            n_reads_vec.append(max(min_reads, int(np.random.poisson(avg_reads))))
    total_reads = int(sum(n_reads_vec))
    
    if total_reads > 0:
        quals_r1 = phred_lines_batch(total_reads, read_len_for)
        quals_r2 = phred_lines_batch(total_reads, read_len_rev)
    else:
        quals_r1 = []
        quals_r2 = []
    
    qual_idx = 0

    # Choose a single seqform pair *per UMI* so the amplicon footprint is identical
    chosen_pairs = {}
    for idx in range(start_idx, end_idx):
        if ins_list[idx] == 'N':
            continue
        # Pick once; reuse for every read of this UMI
        chosen_pairs[idx] = _choose_seqform_pair_with_labels(
            cdna_cfg, require_labels={label}
        )
    
    for idx in range(start_idx, end_idx):
        if ins_list[idx] == 'N':
            continue
        
        n_reads = n_reads_vec[idx - start_idx]
        (cd_for_idx, cd_rv_idx,
         r1_cd_sfs, r2_cd_sfs,
         cd_r1_map, cd_r2_map,
         cd_r1_rev, cd_r2_rev) = chosen_pairs[idx]

        # Build the read(s) for this UMI:
        umi_map = {(label, L): umi_list[idx]}
        # Always build the "primary" once
        r1_cd_primary = build_read_optimized(
            r1_cd_sfs, read_len_for,
            amplicon=ins_list[idx],
            terminators=cdna_cfg['amplicon_terminate'],
            umi_map=umi_map, umi_pool_dict={},
            u_label_by_ord=cd_r1_map, revcomp_by_ord=cd_r1_rev,
            read_tag='f', builder=builder
        )
        r2_cd_primary = build_read_optimized(
            r2_cd_sfs, read_len_rev,
            amplicon=ins_list[idx],
            terminators=cdna_cfg['amplicon_terminate'],
            umi_map=umi_map, umi_pool_dict={},
            u_label_by_ord=cd_r2_map, revcomp_by_ord=cd_r2_rev,
            read_tag='r', builder=builder
        )

        # Optionally build a "secondary" insert for this UMI
        have_secondary = (sec_ins_list is not None and sec_prob > 0.0 and sec_ins_list[idx] != 'N')
        if have_secondary:
            r1_cd_secondary = build_read_optimized(
                r1_cd_sfs, read_len_for,
                amplicon=sec_ins_list[idx],
                terminators=cdna_cfg['amplicon_terminate'],
                umi_map=umi_map, umi_pool_dict={},
                u_label_by_ord=cd_r1_map, revcomp_by_ord=cd_r1_rev,
                read_tag='f', builder=builder
            )
            r2_cd_secondary = build_read_optimized(
                r2_cd_sfs, read_len_rev,
                amplicon=sec_ins_list[idx],
                terminators=cdna_cfg['amplicon_terminate'],
                umi_map=umi_map, umi_pool_dict={},
                u_label_by_ord=cd_r2_map, revcomp_by_ord=cd_r2_rev,
                read_tag='r', builder=builder
            )

        # Emit reads with per-read choice: primary ~ 2/3, secondary ~ 1/3 (if enabled)
        for _ in range(n_reads):
            use_secondary = have_secondary and (np.random.random() < sec_prob)
            if use_secondary:
                r1_cd_const = r1_cd_secondary
                r2_cd_const = r2_cd_secondary
            else:
                r1_cd_const = r1_cd_primary
                r2_cd_const = r2_cd_primary

            q1_cd = quals_r1[qual_idx] if qual_idx < len(quals_r1) else phred_line_vectorized(read_len_for)
            q2_cd = quals_r2[qual_idx] if qual_idx < len(quals_r2) else phred_line_vectorized(read_len_rev)
            qual_idx += 1
            if mean_phred(q1_cd) >= cdna_cfg['min_mean_qual'] and mean_phred(q2_cd) >= cdna_cfg['min_mean_qual']:
                results.append((idx, r1_cd_const, q1_cd, r2_cd_const, q2_cd))
    
    return results

# ───────────────────── Optimized graph-driven simulation ─────────────

def simulate_from_graph(npz_path: str,
                        umi0_inserts_path: str,
                        umi1_inserts_path: str,
                        cdna_cfg: dict,
                        uei_cfg: dict,
                        outdir: Path,
                        avg_reads_uei: float,
                        min_reads_uei: int,
                        avg_reads_cdna: float | None,
                        min_reads_cdna: int | None,
                        use_uei_weights: bool,
                        k_scale: float = 1.0,
                        cdna_secondary_insert: bool = False,
                        encode_digits: str = 'ACGT',
                        seed: int | None = None):
    """Optimized graph-based simulation with parallel processing.

    When cdna_secondary_insert=True, each cDNA UMI i gets a second insert with index:
    new_index = orig_index(i) + max_index(all UMIs) + 1, base-4 encoded with the
    same width and digits as the original. During cDNA read emission, each read
    independently uses secondary with prob=1/3 (primary with prob=2/3).
    """
    try:
        from scipy.sparse import load_npz
    except Exception as e:
        raise RuntimeError("Graph mode requires SciPy. Please install scipy>=1.8.") from e

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    def _outpath(cfg: dict) -> tuple[Path, Path]:
        r1 = cfg.get('source_for', ['R1.fastq'])[0]
        r2 = cfg.get('source_rev', ['R2.fastq'])[0]
        return outdir / r1, outdir / r2

    outdir.mkdir(parents=True, exist_ok=True)
    r1_cd_path, r2_cd_path = _outpath(cdna_cfg)
    r1_ue_path, r2_ue_path = _outpath(uei_cfg)
    
    # Use batched writers
    handles = {
        'R1_cdna': BatchedFASTQWriter(str(r1_cd_path)),
        'R2_cdna': BatchedFASTQWriter(str(r2_cd_path)),
        'R1_uei': BatchedFASTQWriter(str(r1_ue_path)),
        'R2_uei': BatchedFASTQWriter(str(r2_ue_path)),
    }

    cdna_len_for = cdna_cfg['read_length_for']
    cdna_len_rev = cdna_cfg['read_length_rev']
    uei_len_for = uei_cfg['read_length_for']
    uei_len_rev = uei_cfg['read_length_rev']

    cd_lens = _label_lengths(cdna_cfg)
    ue_lens = _label_lengths(uei_cfg)

    for lab in (0, 1):
        if lab in cd_lens and lab in ue_lens and cd_lens[lab] != ue_lens[lab]:
            raise ValueError(f"Label {lab} length differs between libraries")

    if 2 not in ue_lens:
        raise ValueError("UEI settings must define a U-block labeled 2 via -u2.")

    L0 = cd_lens.get(0, ue_lens.get(0))
    L1 = cd_lens.get(1, ue_lens.get(1))
    L2 = ue_lens[2]
    if L0 is None or L1 is None:
        raise ValueError("Both UMI labels 0 and 1 must be defined")

    ins0 = _read_inserts_list(umi0_inserts_path)
    ins1 = _read_inserts_list(umi1_inserts_path)
    n0, n1 = len(ins0), len(ins1)

    M = load_npz(npz_path).tocsr()
    if M.shape[0] != M.shape[1]:
        raise ValueError("UEI graph must be a square CSR matrix.")
    if M.shape[0] != n0 + n1:
        raise ValueError(f"Matrix size {M.shape} does not match inserts")

    # Validate bipartite structure
    coo = M.tocoo()
    intra0 = np.count_nonzero((coo.row < n0) & (coo.col < n0) & (coo.row != coo.col))
    intra1 = np.count_nonzero((coo.row >= n0) & (coo.col >= n0) & (coo.row != coo.col))
    if intra0 or intra1:
        raise ValueError(f"Matrix contains intra-partition edges")

    # Pre-generate UMI sequences
    umi0 = [_fast_random.get_base_string(L0) for _ in range(n0)]
    umi1 = [_fast_random.get_base_string(L1) for _ in range(n1)]

    if avg_reads_cdna is None:
        avg_reads_cdna = avg_reads_uei
    if min_reads_cdna is None:
        min_reads_cdna = min_reads_uei

    # Prepare secondary cDNA inserts if requested
    sec_prob = 0.0
    sec0 = None
    sec1 = None
    if cdna_secondary_insert:
        # compute max index across all non-N inserts (both partitions)
        def _safe_decode_list(xs: list[str], digits: str) -> list[int]:
            vals = []
            for s in xs:
                if s == 'N':
                    continue
                # allow width/padding; just decode the whole string
                if any(ch not in digits for ch in s):
                    continue
                vals.append(decode_index_base4(s, digits=digits))
            return vals

        decoded0 = _safe_decode_list(ins0, encode_digits)
        decoded1 = _safe_decode_list(ins1, encode_digits)
        max_idx = max(decoded0 + decoded1) if (decoded0 or decoded1) else -1
        offset = max_idx + 1 if max_idx >= 0 else 0

        def _make_secondary_list(primary: list[str]) -> list[str]:
            out = []
            for s in primary:
                if s == 'N' or any(ch not in encode_digits for ch in s):
                    out.append('N')
                else:
                    orig = decode_index_base4(s, digits=encode_digits)
                    new_idx = orig + offset
                    out.append(encode_index_base4(new_idx, width=len(s), digits=encode_digits))
            return out

        sec0 = _make_secondary_list(ins0)
        sec1 = _make_secondary_list(ins1)
        # Probability for "secondary" per read so its expected frequency is 1/2 of primary:
        # primary : secondary = 2 : 1  →  p_secondary = 1 / (2+1) = 1/3
        sec_prob = 1.0 / 3.0

    uid = 0

    # Process UMI0 and UMI1 in parallel batches
    chunk_size = min(100, max(1, n0 // 4))
    n_workers = min(4, max(1, (n0 + n1) // chunk_size))
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Derive per-batch seeds if global seed provided
        batch_seeds0 = None
        batch_seeds1 = None
        if seed is not None:
            ss0 = np.random.SeedSequence(seed + 1000)
            ss1 = np.random.SeedSequence(seed + 2000)
            batch_seeds0 = iter(ss0.spawn(max(1, (n0 + chunk_size - 1)//chunk_size)))
            batch_seeds1 = iter(ss1.spawn(max(1, (n1 + chunk_size - 1)//chunk_size)))

        # Process UMI0
        futures = []
        for i in range(0, n0, chunk_size):
            batch_args = (i, min(i + chunk_size, n0),
                          ins0, sec0, sec_prob,
                          umi0, 0, L0,
                          cdna_cfg, avg_reads_cdna, min_reads_cdna,
                          cdna_len_for, cdna_len_rev,
                          (next(batch_seeds0).generate_state(1)[0] if batch_seeds0 else None))
            futures.append(executor.submit(process_umi_batch, batch_args))
        
        # Collect UMI0 results
        for future in as_completed(futures):
            for idx, r1_cd, q1_cd, r2_cd, q2_cd in future.result():
                uid += 1
                handles['R1_cdna'].write_entry(f"C{uid:09d}/1", r1_cd, q1_cd)
                handles['R2_cdna'].write_entry(f"C{uid:09d}/2", r2_cd, q2_cd)
        
        # Process UMI1
        futures = []
        for i in range(0, n1, chunk_size):
            batch_args = (i, min(i + chunk_size, n1),
                          ins1, sec1, sec_prob,
                          umi1, 1, L1,
                          cdna_cfg, avg_reads_cdna, min_reads_cdna,
                          cdna_len_for, cdna_len_rev,
                          (next(batch_seeds1).generate_state(1)[0] if batch_seeds1 else None))
            futures.append(executor.submit(process_umi_batch, batch_args))
        
        # Collect UMI1 results
        for future in as_completed(futures):
            for idx, r1_cd, q1_cd, r2_cd, q2_cd in future.result():
                uid += 1
                handles['R1_cdna'].write_entry(f"C{uid:09d}/1", r1_cd, q1_cd)
                handles['R2_cdna'].write_entry(f"C{uid:09d}/2", r2_cd, q2_cd)

    # Process UEI edges (sequential; could be parallelized later)
    builder = ReadBuilder()
    coo = M.tocoo()
    for i, j, v in zip(coo.row, coo.col, coo.data):
        if not (i < j):
            continue
        if not (i < n0 and j >= n0):
            continue
        u0 = i
        u1 = j - n0

        # Number of distinct UEIs for this association
        K = max(1, int(np.round(k_scale * max(1.0, float(v)))))
        uei_seqs = [_fast_random.get_base_string(L2) for _ in range(K)]

        # Total reads for association, then split across K UEIs
        assoc_scale = float(v) if use_uei_weights else 1.0
        total_reads = int(np.random.poisson(max(0.0, avg_reads_uei) * K * assoc_scale))
        # Base allocation; allow zeros (we’ll enforce per-UEI minima next)
        if K == 1:
            alloc = np.array([total_reads], dtype=int)
        else:
            alloc = np.random.multinomial(total_reads, np.ones(K) / K).astype(int)
        # Enforce per-UEI minimum reads (may increase total slightly)
        alloc = np.maximum(alloc, min_reads_uei)

        for uei_seq, c in zip(uei_seqs, alloc):
            for _ in range(int(c)):
                uid += 1
                ue_for_idx, ue_rv_idx, r1_ue_sfs, r2_ue_sfs, ue_r1_map, ue_r2_map, ue_r1_rev, ue_r2_rev = _choose_seqform_pair_with_labels(uei_cfg, require_labels={0, 1, 2})

                umi_map = {
                    (0, L0): umi0[u0],
                    (1, L1): umi1[u1],
                    (2, L2): uei_seq,
                }

                r1_ue = build_read_optimized(r1_ue_sfs, uei_len_for,
                                             amplicon='',
                                             terminators=uei_cfg['amplicon_terminate'],
                                             umi_map=umi_map, umi_pool_dict={}, 
                                             u_label_by_ord=ue_r1_map, 
                                             read_tag='f', builder=builder)
                r2_ue = build_read_optimized(r2_ue_sfs, uei_len_rev,
                                             amplicon='',
                                             terminators=uei_cfg['amplicon_terminate'],
                                             umi_map=umi_map, umi_pool_dict={}, 
                                             u_label_by_ord=ue_r2_map, 
                                             read_tag='r', builder=builder)

                q1_ue = phred_line_vectorized(uei_len_for)
                q2_ue = phred_line_vectorized(uei_len_rev)
                
                if (mean_phred(q1_ue) >= uei_cfg['min_mean_qual'] and
                    mean_phred(q2_ue) >= uei_cfg['min_mean_qual']):
                    handles['R1_uei'].write_entry(f"U{uid:09d}/1", r1_ue, q1_ue)
                    handles['R2_uei'].write_entry(f"U{uid:09d}/2", r2_ue, q2_ue)

    for h in handles.values():
        h.close()

    shuffle_paired_fastqs(r1_cd_path, r2_cd_path,
                          seed=(None if seed is None else seed + 3001))
    shuffle_paired_fastqs(r1_ue_path, r2_ue_path,
                          seed=(None if seed is None else seed + 3002))

# ─────────────────────────────── CLI ───────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Optimized FASTQ simulator for vDNAmic lib stage."
    )

    # I/O
    ap.add_argument("-o", "--outdir", required=True, type=Path, help="Output directory")
    ap.add_argument("--cdna-settings", required=True, help="Path to cDNA lib.settings")
    ap.add_argument("--uei-settings",  required=True, help="Path to UEI lib.settings")

    # Build mode
    ap.add_argument("--build-from-posfile", help="CSV with columns id,label,x[,y[,z...]]")
    ap.add_argument("--rescale", type=float, default=2.0)
    ap.add_argument("--rescale2", type=float, default=0.5)
    ap.add_argument("--weight2", type=float, default=0.0)
    ap.add_argument("--mperPt", type=float, default=50.0)
    ap.add_argument("--neg-bin-p", type=float, default=0.8)
    ap.add_argument("--amp-dispersion", type=float, default=0.0)
    ap.add_argument("--dropout0", type=float, default=0.0)
    ap.add_argument("--dropout1", type=float, default=0.0)
    ap.add_argument("--encode-digits", default="ACGT")
    ap.add_argument("--encode-width", type=int, default=0)

    # Simulate mode
    ap.add_argument("--uei-graph-npz", help="CSR .npz file")
    ap.add_argument("--umi0-inserts", help="Path to umi0.txt")
    ap.add_argument("--umi1-inserts", help="Path to umi1.txt")

    # Coverage / multiplicity
    ap.add_argument("--avg-reads-uei", type=float, required=True)
    ap.add_argument("--min-reads-uei", type=int, default=1)
    ap.add_argument("--avg-reads-cdna", type=float)
    ap.add_argument("--min-reads-cdna", type=int)
    ap.add_argument("--use-uei-weights", action="store_true")
    ap.add_argument("--k-scale", type=float, default=1.0,
                    help="Scale factor for UEIs per association: K ≈ round(k_scale * v).")

    # NEW: secondary cDNA inserts
    ap.add_argument("--cdna-secondary-insert", action="store_true",
                    help="If set, each cDNA UMI gets a second insert whose index is "
                         "orig_index + max_index + 1 (base-4 encoded using the same "
                         "digits/width). During cDNA simulation, per-read choice is "
                         "~2/3 primary vs ~1/3 secondary (secondary at 50% of primary).")

    # Misc
    ap.add_argument("--umi-pool", type=int, default=16)
    ap.add_argument("--seed", type=int)

    args = ap.parse_args()

    # Parse settings
    cdna_cfg = parse_settings(args.cdna_settings)
    uei_cfg  = parse_settings(args.uei_settings)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Build if requested
    umi0_path = args.umi0_inserts
    umi1_path = args.umi1_inserts
    npz_path  = args.uei_graph_npz

    if args.build_from_posfile:
        umi0_path, umi1_path, npz_path, n0, n1 = build_inputs_from_positions(
            pos_csv=args.build_from_posfile,
            outdir=outdir,
            rescale=args.rescale,
            rescale2=args.rescale2,
            weight2=args.weight2,
            mperPt=args.mperPt,
            negbin_p=args.neg_bin_p,
            amp_dispersion=args.amp_dispersion,
            dropout0=args.dropout0,
            dropout1=args.dropout1,
            encode_width=args.encode_width,
            encode_digits=args.encode_digits,
            seed=args.seed,
        )

    # Require graph + inserts
    if not (umi0_path and umi1_path and npz_path):
        raise SystemExit("Need --uei-graph-npz, --umi0-inserts, --umi1-inserts (or use --build-from-posfile).")

    simulate_from_graph(
        npz_path=str(npz_path),
        umi0_inserts_path=str(umi0_path),
        umi1_inserts_path=str(umi1_path),
        cdna_cfg=cdna_cfg,
        uei_cfg=uei_cfg,
        outdir=outdir,
        avg_reads_uei=args.avg_reads_uei,
        min_reads_uei=args.min_reads_uei,
        avg_reads_cdna=args.avg_reads_cdna,
        min_reads_cdna=args.min_reads_cdna,
        use_uei_weights=bool(args.use_uei_weights),
        k_scale=float(args.k_scale),
        cdna_secondary_insert=bool(args.cdna_secondary_insert),
        encode_digits=str(args.encode_digits),
        seed=args.seed,
    )

    # Post-simulation: export a random (d-1) cross-section slab through the point cloud.
    # Only available when we were given the original positions via --build-from-posfile.
    if args.build_from_posfile:
        try:
            export_random_cross_section(
                pos_csv=str(args.build_from_posfile),
                outdir=outdir,
                umi0_inserts_path=str(umi0_path),
                umi1_inserts_path=str(umi1_path),
                rescale=float(args.rescale),
                thickness_frac=0.10,
                seed=args.seed,
                encode_digits=str(args.encode_digits),
                cdna_secondary_insert=bool(args.cdna_secondary_insert),
            )
        except Exception as e:
            print(f"[WARN] cross-section export failed: {e}")

    points_path, meta_path, png_path = export_random_cross_section(
        pos_csv=str(args.build_from_posfile),
        outdir=outdir,
        umi0_inserts_path=str(umi0_path),
        umi1_inserts_path=str(umi1_path),
        rescale=float(args.rescale),
        thickness_frac=0.10,
        seed=args.seed,
        encode_digits=str(args.encode_digits),
        cdna_secondary_insert=bool(args.cdna_secondary_insert),
    )

    # Build AnnData + write .h5ad
    df_cs = pd.read_csv(points_path)

    with open(meta_path) as fh:
        meta = json.load(fh)
    d = int(meta["d"])

    adata = build_cross_section_anndata(
        df_cs,
        d=d,
        encode_digits=str(args.encode_digits),
        knn_k=15,  # or make CLI arg
    )

    # Optional: store slice geometry in adata.uns for provenance
    adata.uns["cross_section_json"] = json.dumps(meta)

    h5ad_path = outdir / "cross_section.h5ad"
    adata.write_h5ad(h5ad_path)
    print(f"[INFO] wrote {h5ad_path}")


if __name__ == "__main__":
    main()