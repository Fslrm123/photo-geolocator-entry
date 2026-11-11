# photo-geolocator-01: a code-only photo geolocator (no model calls, no API key).
#
# The agent fetches the instance's 25 street-level photos from the env,
# decodes each JPEG with a small pure-stdlib baseline-DCT decoder (DC
# coefficients only: one average colour per 8x8 block), reads simple visual
# cues from those block colours (snow on the ground, red dirt, yellow centre
# lines, lush tropical green, overcast sky, sand), scores eleven world
# regions from the cues, and answers the softmax-weighted centre of the
# regions, blended toward the dataset's geographic centre. Every photo is
# always answered: an unanswered photo counts the maximum error, so even a
# rough guess beats leaving it blank. Fully deterministic: no clock, no
# randomness, requests in photo order.

import json
import math
import os
import urllib.request

DEFAULT = (57.0, 27.0)  # geographic centre of the photo pool's landmass

REGIONS = {
    "NA": (42.0, -96.0), "LATAM": (-12.0, -55.0), "EUR": (51.0, 10.0),
    "RUS": (58.0, 62.0), "EASIA": (35.0, 118.0), "SASIA": (24.0, 80.0),
    "SEASIA": (6.0, 108.0), "MENA": (27.0, 42.0), "AFRICA": (2.0, 24.0),
    "AUS": (-25.0, 134.0), "NZ": (-42.0, 173.0),
}
BASE = {"NA": 0.4, "LATAM": 0.4, "EUR": 0.4, "RUS": 0.4, "EASIA": 0.2,
        "SASIA": 0.2, "SEASIA": 0.2, "MENA": 0.2, "AFRICA": 0.2, "AUS": 0.2,
        "NZ": 0.0}


# ---------------------------------------------------------------- JPEG decode
# Baseline DCT only (SOF0). Reads the DC coefficient of every 8x8 block and
# skips the AC coefficients, so the result is one average colour per block.

class Bits:
    """Bit reader over JPEG entropy data, undoing 0xFF00 stuffing."""

    def __init__(self, data, start):
        self.d = data
        self.i = start
        self.acc = 0
        self.n = 0

    def bit(self):
        if self.n == 0:
            if self.i >= len(self.d):
                raise ValueError("entropy data ends early")
            b = self.d[self.i]
            self.i += 1
            if b == 0xFF:
                if self.i >= len(self.d):
                    raise ValueError("entropy data ends early")
                nb = self.d[self.i]
                if nb == 0x00:
                    self.i += 1
                else:
                    raise ValueError("unexpected marker in entropy data")
            self.acc = b
            self.n = 8
        self.n -= 1
        return (self.acc >> self.n) & 1

    def bits(self, n):
        v = 0
        for _ in range(n):
            v = (v << 1) | self.bit()
        return v

    def align(self):
        self.n = 0


class Huff:
    """Canonical Huffman table from a DHT segment's counts and symbols."""

    def __init__(self, counts, symbols):
        self.map = {}
        code = 0
        k = 0
        for length in range(1, 17):
            for _ in range(counts[length - 1]):
                self.map[(length, code)] = symbols[k]
                k += 1
                code += 1
            code <<= 1

    def decode(self, bits):
        code = 0
        length = 0
        while True:
            code = (code << 1) | bits.bit()
            length += 1
            sym = self.map.get((length, code))
            if sym is not None:
                return sym
            if length > 16:
                raise ValueError("bad Huffman code")


def _dc_value(bits, table):
    cat = table.decode(bits)
    if cat == 0:
        return 0
    v = bits.bits(cat)
    if v < (1 << (cat - 1)):
        v -= (1 << cat) - 1
    return v


def _skip_ac(bits, table):
    n = 0
    while n < 63:
        s = table.decode(bits)
        run = s >> 4
        size = s & 15
        if size == 0:
            if run == 15:
                n += 16
                continue
            return  # end of block
        n += run + 1
        bits.bits(size)


def decode_jpeg_dc(data):
    """Return (cols, rows, grid): grid is rows of (r, g, b) block averages."""
    if len(data) < 4 or data[0] != 0xFF or data[1] != 0xD8:
        raise ValueError("not a JPEG")
    qtables = {}
    dc_tables = {}
    ac_tables = {}
    restart_interval = 0
    width = height = 0
    comps = []  # each: dict(id, h, v, qt, dc, ac)
    i = 2
    scan_start = None
    while i < len(data) - 1:
        if data[i] != 0xFF:
            raise ValueError("bad marker at %d" % i)
        marker = data[i + 1]
        i += 2
        if marker == 0xD9:  # EOI
            break
        if marker == 0xDA:  # SOS: entropy data follows
            ncomp = data[i + 2]
            for k in range(ncomp):
                cid = data[i + 3 + k * 2]
                sel = data[i + 4 + k * 2]
                for c in comps:
                    if c["id"] == cid:
                        c["dc"] = sel >> 4
                        c["ac"] = sel & 15
            scan_start = i + 2 + 1 + ncomp * 2 + 3
            break
        if marker in (0x01,) or 0xD0 <= marker <= 0xD8:
            continue
        length = (data[i] << 8) | data[i + 1]
        seg = data[i + 2:i + length]
        if marker == 0xDB:  # DQT: only the DC value (index 0) is needed
            j = 0
            while j < len(seg):
                pq = seg[j] >> 4
                tq = seg[j] & 15
                if pq == 0:
                    qtables[tq] = seg[j + 1]
                    j += 65
                else:
                    qtables[tq] = (seg[j + 1] << 8) | seg[j + 2]
                    j += 129
        elif marker == 0xC4:  # DHT
            j = 0
            while j < len(seg):
                tc = seg[j] >> 4
                th = seg[j] & 15
                counts = list(seg[j + 1:j + 17])
                total = sum(counts)
                symbols = list(seg[j + 17:j + 17 + total])
                table = Huff(counts, symbols)
                if tc == 0:
                    dc_tables[th] = table
                else:
                    ac_tables[th] = table
                j += 17 + total
        elif marker == 0xDD:  # DRI: restart interval in MCUs
            restart_interval = (seg[0] << 8) | seg[1]
        elif marker == 0xC0:  # SOF0 baseline
            height = (seg[1] << 8) | seg[2]
            width = (seg[3] << 8) | seg[4]
            ncomp = seg[5]
            comps = []
            for k in range(ncomp):
                cid = seg[6 + k * 3]
                hv = seg[7 + k * 3]
                qt = seg[8 + k * 3]
                comps.append({"id": cid, "h": hv >> 4, "v": hv & 15,
                              "qt": qt, "dc": 0, "ac": 0})
        elif marker == 0xC2:
            raise ValueError("progressive JPEG not supported")
        i += length
    if scan_start is None or not comps:
        raise ValueError("no scan found")
    hmax = max(c["h"] for c in comps)
    vmax = max(c["v"] for c in comps)
    mcu_x = (width + 8 * hmax - 1) // (8 * hmax)
    mcu_y = (height + 8 * vmax - 1) // (8 * vmax)
    # DC value per component, keyed by (block_x, block_y)
    dc = []
    for c in comps:
        dc.append({})
    bits = Bits(data, scan_start)
    pred = [0] * len(comps)
    mcu_index = 0
    for my in range(mcu_y):
        for mx in range(mcu_x):
            for ci, c in enumerate(comps):
                for by in range(c["v"]):
                    for bx in range(c["h"]):
                        pred[ci] += _dc_value(bits, dc_tables[c["dc"]])
                        _skip_ac(bits, ac_tables[c["ac"]])
                        dc[ci][(mx * c["h"] + bx, my * c["v"] + by)] = (
                            pred[ci] * qtables[c["qt"]])
            mcu_index += 1
            # A restart marker, when the file uses them, starts at a byte
            # boundary after the MCU that ends its interval; without one the
            # bitstream runs on uninterrupted, so never align otherwise.
            if restart_interval and mcu_index % restart_interval == 0:
                bits.align()
                if (bits.i + 1 < len(data) and data[bits.i] == 0xFF
                        and 0xD0 <= data[bits.i + 1] <= 0xD7):
                    bits.i += 2
                    pred = [0] * len(comps)
    ycomp = comps[0]
    cols = mcu_x * ycomp["h"]
    rows = mcu_y * ycomp["v"]
    grid = []
    for by in range(rows):
        row = []
        for bx in range(cols):
            y = dc[0].get((bx, by), 0) / 8.0 + 128.0
            if len(comps) == 1:
                cb = cr = 128.0
            else:
                cb = cr = 0.0
                for ci in (1, 2):
                    c = comps[ci]
                    cx = bx * c["h"] // ycomp["h"]
                    cy = by * c["v"] // ycomp["v"]
                    val = dc[ci].get((cx, cy), 0) / 8.0 + 128.0
                    if ci == 1:
                        cb = val
                    else:
                        cr = val
            r = y + 1.402 * (cr - 128.0)
            g = y - 0.34414 * (cb - 128.0) - 0.71414 * (cr - 128.0)
            b = y + 1.772 * (cb - 128.0)
            row.append((max(0.0, min(255.0, r)),
                        max(0.0, min(255.0, g)),
                        max(0.0, min(255.0, b))))
        grid.append(row)
    return cols, rows, grid


# ------------------------------------------------------------------ features

def _frac(grid, rows, pred):
    n = 0
    hit = 0
    for y in rows:
        for (r, g, b) in grid[y]:
            n += 1
            if pred(r, g, b):
                hit += 1
    return hit / n if n else 0.0


def features(cols, rows, grid):
    h = rows
    sky = range(0, max(1, h // 7))
    mid = range(h // 4, (3 * h) // 5)
    gnd = range((3 * h) // 5, h)
    allrows = range(h)
    lum_sum = 0.0
    sat_sum = 0.0
    n = 0
    for y in range(h):
        for (r, g, b) in grid[y]:
            mx = max(r, g, b)
            mn = min(r, g, b)
            lum_sum += (r + g + b) / 3.0
            sat_sum += (mx - mn) / mx if mx > 0 else 0.0
            n += 1
    bright = lum_sum / n if n else 0.0
    sat = sat_sum / n if n else 0.0

    def lum_sat(r, g, b):
        mx = max(r, g, b)
        mn = min(r, g, b)
        return (r + g + b) / 3.0, (mx - mn) / mx if mx > 0 else 0.0

    return {
        "yel_g": _frac(grid, gnd, lambda r, g, b: r > 150 and g > 110
                       and b < 130 and r - b > 50),
        "red_any": _frac(grid, allrows, lambda r, g, b: r > 90 and r > g + 25
                         and g > b and r < 215),
        "green_m": _frac(grid, mid, lambda r, g, b: g > r + 12 and g > b + 8
                         and g > 60),
        "green_sky": _frac(grid, sky, lambda r, g, b: g > r + 15 and g > b + 10
                           and g > 70),
        "snow": _frac(grid, gnd, lambda r, g, b: lum_sat(r, g, b)[0] > 200
                      and lum_sat(r, g, b)[1] < 0.12),
        "snow_any": _frac(grid, allrows, lambda r, g, b: lum_sat(r, g, b)[0] > 205
                         and lum_sat(r, g, b)[1] < 0.10),
        "skyblue": _frac(grid, sky, lambda r, g, b: b > r + 20 and b > 100),
        "overcast": _frac(grid, sky, lambda r, g, b: abs(r - b) < 15
                         and (r + g + b) / 3.0 > 90),
        "sand": _frac(grid, gnd, lambda r, g, b: r > 130 and g > 100 and b < 140
                      and r - b > 25 and lum_sat(r, g, b)[1] < 0.35),
        "dry": _frac(grid, mid, lambda r, g, b: r > 120 and g > 100 and b < 130
                     and r - b > 30 and g > 70),
        "bright": bright,
        "sat": sat,
    }


# ------------------------------------------------------------------ guessing

def region_scores(f):
    s = dict(BASE)
    if f["snow"] > 0.22 or f["snow_any"] > 0.30:
        s["RUS"] += 3.0
        s["NA"] += 0.7
        s["EUR"] += 0.5
    if f["red_any"] > 0.06:
        s["AUS"] += 2.5
        s["AFRICA"] += 1.6
        s["LATAM"] += 1.0
        s["MENA"] += 0.5
        if f["dry"] > 0.05 and f["green_m"] < 0.08:
            s["AUS"] += 1.2
    if f["yel_g"] > 0.015:
        s["NA"] += 2.0
        s["LATAM"] += 2.0
        if f["green_m"] > 0.10 or f["bright"] > 135:
            s["LATAM"] += 1.0
        else:
            s["NA"] += 0.8
    if f["green_m"] > 0.14:
        s["SEASIA"] += 2.0
        s["LATAM"] += 1.2
        s["AFRICA"] += 1.2
        s["SASIA"] += 1.0
        if f["skyblue"] > 0.5 and f["bright"] > 120:
            s["SEASIA"] += 1.0
    if f["green_sky"] > 0.08:
        s["SEASIA"] += 1.5
        s["LATAM"] += 0.8
        s["AFRICA"] += 0.8
    if f["overcast"] > 0.5 and f["green_m"] > 0.05:
        s["EUR"] += 2.0
        s["EASIA"] += 1.0
        s["NA"] += 0.6
        s["RUS"] += 0.5
    if f["sand"] > 0.10 and f["green_m"] < 0.06:
        s["MENA"] += 2.5
        s["AUS"] += 1.0
        s["AFRICA"] += 0.5
    if f["skyblue"] > 0.7 and f["bright"] > 130 and f["green_m"] < 0.05:
        s["MENA"] += 1.0
        s["AUS"] += 1.0
    return s


def _unit(lat, lon):
    la = math.radians(lat)
    lo = math.radians(lon)
    return (math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo),
            math.sin(la))


def _from_unit(x, y, z):
    return (math.degrees(math.atan2(z, math.hypot(x, y))),
            math.degrees(math.atan2(y, x)))


def guess(f):
    """Softmax-weighted centre of the scored regions, blended 40% toward
    the dataset's geographic centre: the cues are weak, so a confident
    wrong region must not fling the answer to the far side of the world."""
    s = region_scores(f)
    total = 0.0
    x = y = z = 0.0
    for name, score in s.items():
        w = math.exp(score)
        total += w
        ux, uy, uz = _unit(*REGIONS[name])
        x += w * ux
        y += w * uy
        z += w * uz
    x /= total
    y /= total
    z /= total
    dx, dy, dz = _unit(*DEFAULT)
    x = 0.6 * x + 0.4 * dx
    y = 0.6 * y + 0.4 * dy
    z = 0.6 * z + 0.4 * dz
    return _from_unit(x, y, z)


# ---------------------------------------------------------------------- env

def http_get(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def http_post(url, payload, timeout=60):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def main():
    env = os.environ["MAKEFUN_ENV_URL"].rstrip("/")
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE", "")
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8", errors="replace") as fh:
            fh.read()
    listing = json.loads(http_get(env + "/photos").decode("utf-8"))
    answers = []
    for photo in listing["photos"]:
        pid = photo["id"]
        if photo.get("answered"):
            continue
        try:
            data = http_get(env + photo["url"])
            cols, rows, grid = decode_jpeg_dc(data)
            lat, lon = guess(features(cols, rows, grid))
        except Exception as err:
            print("photo %s: %s; using default guess" % (pid, err))
            lat, lon = DEFAULT
        lat = max(-90.0, min(90.0, lat))
        lon = max(-180.0, min(180.0, lon))
        answers.append({"id": pid, "lat": round(lat, 6), "lon": round(lon, 6)})
    if answers:
        print(http_post(env + "/answers", {"answers": answers}).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
