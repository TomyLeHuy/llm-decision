"""Check that the frozen files are unchanged (compares against confirmatory/SHA256SUMS). Exit code 1 if anything differs."""
import hashlib, os, sys
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(here); bad = 0
for line in open(os.path.join(here, "SHA256SUMS")):
    if not line.strip(): continue
    h, p = line.split(maxsplit=1); p = p.strip()
    got = hashlib.sha256(open(os.path.join(root, p), "rb").read()).hexdigest()
    print(("OK      " if got == h else "CHANGED ") + p); bad += got != h
sys.exit(1 if bad else 0)
