#!/usr/bin/env python3
"""
Dependency-free previewer for the ICP PCD dumps.

Reads the ASCII .pcd files written by `ICP_DUMP=1 ./bin/icp_serial`, projects
them into one orthographic 3D view, colours them by category, and writes a PNG
(no numpy / matplotlib / PIL -- just the standard library).

  target.pcd          gray    the target cloud (fixed)
  source_initial.pcd  red     the source before alignment (misaligned)
  source_aligned.pcd  green   the source after ICP (should overlay the target)

Usage: python3 tools/render_pcd.py [out.png]
"""
import math, struct, zlib, sys

W, H = 1100, 850
AZ, EL = math.radians(-55), math.radians(16)   # view direction


def read_pcd(path, stride=1):
    pts, started, i = [], False, 0
    try:
        f = open(path)
    except OSError:
        return pts
    for line in f:
        if not started:
            if line.startswith("DATA"):
                started = True
            continue
        s = line.split()
        if len(s) >= 3:
            if i % stride == 0:
                pts.append((float(s[0]), float(s[1]), float(s[2])))
            i += 1
    f.close()
    return pts


def project(p, c):
    x, y, z = p[0]-c[0], p[1]-c[1], p[2]-c[2]
    x1 = x*math.cos(AZ) - y*math.sin(AZ)
    y1 = x*math.sin(AZ) + y*math.cos(AZ)
    y2 = y1*math.cos(EL) - z*math.sin(EL)
    z2 = y1*math.sin(EL) + z*math.cos(EL)
    return x1, z2, y2            # screen-x, screen-y(up), depth(far = larger)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "preview.png"
    layers = [
        ("target.pcd",         1, (150, 150, 150), 1),
        ("source_initial.pcd", 1, (235,  50,  50), 1),
        ("source_aligned.pcd", 1, ( 40, 200,  60), 1),
    ]
    clouds = [(read_pcd(p, st), col, rad) for p, st, col, rad in layers]
    allpts = [q for c, _, _ in clouds for q in c]
    if not allpts:
        print("No PCD files found -- run `ICP_DUMP=1 ./bin/icp_serial` first.")
        return
    c = [sum(a[k] for a in allpts)/len(allpts) for k in range(3)]

    proj = []
    for cl, col, rad in clouds:
        pl = [project(q, c) for q in cl]
        proj.append((pl, [col]*len(cl), rad))

    xs = [s[0] for pl, _, _ in proj for s in pl]
    ys = [s[1] for pl, _, _ in proj for s in pl]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    pad = 40
    sc = min((W-2*pad)/(maxx-minx+1e-9), (H-2*pad)/(maxy-miny+1e-9))

    def to_px(s):
        px = int(pad + (s[0]-minx)*sc)
        py = int(H-1 - (pad + (s[1]-miny)*sc))   # flip y for image coords
        return px, py

    img = bytearray((18,)*3 * W*H)               # dark background

    def plot(px, py, col, rad):
        for dx in range(-rad, rad+1):
            for dy in range(-rad, rad+1):
                x, y = px+dx, py+dy
                if 0 <= x < W and 0 <= y < H:
                    o = (y*W + x)*3
                    img[o], img[o+1], img[o+2] = col

    for pl, cols, rad in proj:                   # painter's order: far points first
        order = sorted(range(len(pl)), key=lambda i: -pl[i][2])
        for i in order:
            px, py = to_px(pl[i])
            plot(px, py, cols[i], rad)

    # PNG encode (stdlib only)
    raw = bytearray()
    for y in range(H):
        raw.append(0)
        raw += img[y*W*3:(y+1)*W*3]

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data +
                struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))

    png = (b"\x89PNG\r\n\x1a\n" +
           chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)) +
           chunk(b"IDAT", zlib.compress(bytes(raw), 9)) +
           chunk(b"IEND", b""))
    open(out, "wb").write(png)
    print(f"wrote {out}  (" +
          ", ".join(f"{p}={len(cl)}" for (p, _, _, _), (cl, _, _) in zip(layers, clouds)) + ")")


if __name__ == "__main__":
    main()
