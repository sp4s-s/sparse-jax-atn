import argparse
import sys
import jax
from sparse_attention.data import create_dummy_inputs
from sparse_attention.masks import create_block_mask
from sparse_attention.kernel import sparse_attention
from sparse_attention.dense_attention import dense_attention

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--target", choices=["sparse", "dense"], required=True)
    parser.add_argument("--sparsity", default="combined")
    parser.add_argument("--no-pallas", action="store_true")
    args = parser.parse_args()
    
    try:
        q, k, v = create_dummy_inputs(args.b, args.n, 8, 32)
        if args.target == "sparse":
            mask = create_block_mask(args.n, 128, args.sparsity)
            out = sparse_attention(q, k, v, mask, use_pallas=not args.no_pallas)
        else:
            out = dense_attention(q, k, v, causal=True)
        
        out.block_until_ready()
        
        print("PASS")
        sys.exit(0)
    except Exception as e:
        print(f"FAIL: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
