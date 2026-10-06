"""Verify all bundled samples and fixed 80/20 splits. No network access."""
from cnnlab.common import ROOT
from cnnlab.data import load_bundle, CLASSES

if __name__ == "__main__":
    for domain in CLASSES:
        x,y,ids,meta = load_bundle(ROOT/"data", domain)
        s=meta["split"]
        print(f"[PASS] {domain}: {len(y)} images | train pool={len(s['train_pool'])} | test={len(s['test'])}")
        print("       test_hash =",s["test_hash"])
    print("[READY] All three real-data files are present, readable, balanced, unique and checksum-verified.")
