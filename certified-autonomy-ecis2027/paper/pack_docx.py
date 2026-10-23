import os, sys, zipfile
src, out = sys.argv[1], sys.argv[2]
files = []
for root, _, fs in os.walk(src):
    for f in fs: files.append(os.path.relpath(os.path.join(root, f), src))
files.sort(key=lambda p: (p != "[Content_Types].xml", p))
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in files: z.write(os.path.join(src, p), p)
print("packed", len(files), "parts ->", out)
