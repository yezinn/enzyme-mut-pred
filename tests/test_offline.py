"""Offline validation: everything that does not need the real model weights or DMS data."""
import pandas as pd, torch, sys
from transformers import EsmConfig, EsmForMaskedLM
import proteingym_benchmark as pgb

print("=" * 70)
print("TEST 1: metadata + token length (real ProteinGym reference)")
meta = pgb.load_assay_metadata("AMIE_PSEAE_Wrenbeck_2017")
print(f"  {meta.uniprot_id} | {meta.seq_len} aa | {meta.n_single_mutants} single mutants")
print(f"  phenotype={meta.raw_phenotype} directionality={meta.raw_directionality}")
assert meta.seq_len == 346 and len(meta.target_seq) == 346

print("\nTEST 2: official baseline lookup")
for dms in ["AMIE_PSEAE_Wrenbeck_2017", "BLAT_ECOLX_Stiffler_2015", "BLAT_ECOLX_Jacquier_2013"]:
    print(f"  {dms:28s} ESM2-650M = {pgb.official_baseline(dms)}")

print("\nTEST 3: mutant validation (synthetic, built from the real AMIE sequence)")
seq = meta.target_seq
good = pd.DataFrame({"mutant": [f"{seq[p-1]}{p}A" for p in (1, 50, 200, 346)],
                     "DMS_score": [0.1, -1.0, 0.5, -2.0],
                     "DMS_score_bin": [1, 0, 1, 0]})
v = pgb.validate_mutants(good, seq)
assert list(v["position"]) == [1, 50, 200, 346]
bad = pd.DataFrame({"mutant": ["X5A"], "DMS_score": [0.0]})
try:
    pgb.validate_mutants(bad, seq); print("  FAIL: mismatch not caught")
except ValueError as e:
    print(f"  mismatch correctly rejected: {str(e)[:60]}...")
oob = pd.DataFrame({"mutant": [f"{seq[0]}9999A"], "DMS_score": [0.0]})
try:
    pgb.validate_mutants(oob, seq); print("  FAIL: out-of-range not caught")
except ValueError as e:
    print(f"  out-of-range correctly rejected: {str(e)[:60]}...")

print("\nTEST 4: score direction check")
pgb.check_score_direction(good, meta)
flipped = good.copy(); flipped["DMS_score"] *= -1
try:
    pgb.check_score_direction(flipped, meta); print("  FAIL: inverted scores not caught")
except ValueError as e:
    print(f"  inverted scores correctly rejected: {str(e)[:60]}...")

print("\n" + "=" * 70)
print("TEST 5: batch-cached scoring == naive per-mutation scoring (random-init tiny ESM)")
# Build an ESM tokenizer locally (no network): standard ESM-2 alphabet.
import os
from transformers import EsmTokenizer
vocab = ["<cls>","<pad>","<eos>","<unk>","L","A","G","V","S","E","R","T","I","D","P","K","Q",
         "N","F","Y","M","H","W","C","X","B","U","Z","O",".","-","<null_1>","<mask>"]
os.makedirs("tmp_tok", exist_ok=True)
open("tmp_tok/vocab.txt","w").write("\n".join(vocab))
tok = EsmTokenizer(vocab_file="tmp_tok/vocab.txt")

torch.manual_seed(0)
cfg = EsmConfig(vocab_size=len(vocab), hidden_size=64, num_hidden_layers=2,
                num_attention_heads=4, intermediate_size=128, max_position_embeddings=512,
                pad_token_id=tok.pad_token_id, mask_token_id=tok.mask_token_id)
model = EsmForMaskedLM(cfg).eval()

test_seq = meta.target_seq[:120]
muts = pd.DataFrame({"mutant": [f"{test_seq[p-1]}{p}{m}"
                                for p, m in [(1,"A"),(7,"K"),(7,"W"),(7,"D"),(60,"G"),(119,"P"),(120,"C")]]})
muts["DMS_score"] = 0.0
muts = pgb.validate_mutants(muts, test_seq)

# reference implementation: one forward pass per mutation
@torch.no_grad()
def naive(seq, wt, pos, mut):
    t = tok(seq, return_tensors="pt")
    t["input_ids"][0, pos] = tok.mask_token_id
    lg = model(**t).logits
    lp = torch.log_softmax(lg[0, pos], dim=-1)
    return (lp[tok.convert_tokens_to_ids(mut)] - lp[tok.convert_tokens_to_ids(wt)]).item()

ref = [naive(test_seq, r.wt_aa, r.position, r.mut_aa) for r in muts.itertuples()]
fast = pgb.score_variants(muts, test_seq, model, tok, "cpu", batch_size=3)["esm2_llr"].tolist()

max_diff = max(abs(a-b) for a, b in zip(ref, fast))
print(f"  mutations: {len(ref)}, max |naive - batched| = {max_diff:.3e}")
for mt, a, b in zip(muts["mutant"], ref, fast):
    print(f"    {mt:>7s}  naive={a:+.6f}  batched={b:+.6f}")
assert max_diff < 1e-4, "batched scoring does not match naive scoring!"
print("  PASS: batched position-cached scoring is numerically identical to naive scoring")
print("=" * 70)
print("ALL OFFLINE TESTS PASSED")
