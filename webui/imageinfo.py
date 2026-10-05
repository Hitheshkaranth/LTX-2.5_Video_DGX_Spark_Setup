#!/usr/bin/env python3
"""Image sniffing for image-to-video (stdlib only, no Pillow needed).

The pipeline resizes and center-crops the conditioning image to the output size, so the output
should share the image's aspect ratio or the crop eats the subject. best_size() picks the preset
closest in aspect ratio at the requested quality tier.

CLI: imageinfo.py IMAGE [tier]  ->  prints "WIDTH HEIGHT" (used by i2v.sh)
"""
import struct
import sys

# (width, height) presets: all multiples of 64 (pass 1 renders at half size).
TIERS = {
    "fast": [(768, 512), (512, 768), (640, 640)],
    "hd": [(1024, 576), (576, 1024), (1280, 704), (704, 1280), (1024, 768), (768, 1024), (768, 768)],
    "max": [(1536, 1024), (1024, 1536), (1024, 1024)],
}
SIZES = {f"{w}x{h}" for tier in TIERS.values() for w, h in tier}


def _exif_orientation(app1: bytes) -> int:
    """Orientation tag (1-8) from an APP1 Exif payload, 1 if absent."""
    if app1[:6] != b"Exif\x00\x00":
        return 1
    t = app1[6:]
    end = {b"II": "<", b"MM": ">"}.get(t[:2])
    if not end or len(t) < 8:
        return 1
    ifd = struct.unpack(end + "I", t[4:8])[0]
    if ifd + 2 > len(t):
        return 1
    for k in range(struct.unpack(end + "H", t[ifd:ifd + 2])[0]):
        e = ifd + 2 + 12 * k
        if e + 12 <= len(t) and struct.unpack(end + "H", t[e:e + 2])[0] == 0x0112:
            return struct.unpack(end + "H", t[e + 8:e + 10])[0]
    return 1


def sniff(data: bytes):
    """Return (kind, width, height) for PNG/JPEG/WebP bytes, or None if unrecognised.

    For JPEG, width/height are as displayed: EXIF orientations 5-8 (90° rotations, typical for
    phone photos) swap them, matching how the pipeline's decoder rotates the image.
    """
    if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        w, h = struct.unpack(">II", data[16:24])
        return "png", w, h
    if data[:2] == b"\xff\xd8":
        i, orientation = 2, 1
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            seg = struct.unpack(">H", data[i + 2:i + 4])[0]
            if marker == 0xE1:
                orientation = _exif_orientation(data[i + 4:i + 2 + seg])
            # SOF0..SOF15 except DHT (C4), JPG (C8), DAC (CC) carry the frame size
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                return ("jpeg", h, w) if orientation >= 5 else ("jpeg", w, h)
            i += 2 + seg
        return None
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        chunk = data[12:16]
        if chunk == b"VP8X":
            w = 1 + int.from_bytes(data[24:27], "little")
            h = 1 + int.from_bytes(data[27:30], "little")
            return "webp", w, h
        if chunk == b"VP8 ":
            w, h = struct.unpack("<HH", data[26:30])
            return "webp", w & 0x3FFF, h & 0x3FFF
        if chunk == b"VP8L":
            b = int.from_bytes(data[21:25], "little")
            return "webp", (b & 0x3FFF) + 1, ((b >> 14) & 0x3FFF) + 1
    return None


def best_size(width: int, height: int, tier: str = "fast"):
    """Preset in `tier` whose aspect ratio is closest to width/height."""
    import math
    target = math.log(width / height)
    return min(TIERS[tier], key=lambda s: abs(math.log(s[0] / s[1]) - target))


if __name__ == "__main__":
    with open(sys.argv[1], "rb") as f:
        info = sniff(f.read(1 << 20))
    if not info:
        sys.exit(f"{sys.argv[1]}: not a PNG, JPEG or WebP image")
    w, h = best_size(info[1], info[2], sys.argv[2] if len(sys.argv) > 2 else "fast")
    print(w, h)
