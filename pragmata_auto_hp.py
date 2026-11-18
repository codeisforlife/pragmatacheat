"""
Pragmata - Auto HP Scanner & Lock v3 (single-player, PC)
--------------------------------------------------------
Alur v3 (lebih simpel):
  1. Ketik angka darah yang kelihatan di layar (mis. 1200).
     Script mencari angka itu sekaligus sebagai float, double, dan int.
  2. Balik ke game, kena damage / heal sampai angka darah BERUBAH.
  3. Balik ke CMD, ketik angka darah yang baru (mis. 1066).
  4. Ulangi 2-3 sampai kandidat sedikit, lalu tes / kunci.

Syarat: pip install pymem numpy  |  Python 64-bit  |  CMD sebagai Administrator.
Tips: pakai mode Windowed / Borderless supaya game tidak pause saat alt-tab.
Tombol saat lock: F1 = ON/OFF, F2 = kembali ke menu.
"""

import ctypes
import ctypes.wintypes as wt
import time

import numpy as np
import pymem
import pymem.exception
import pymem.process

PROCESS_NAME = "PRAGMATA.exe"      # sesuaikan dengan nama di Task Manager > Details
CHUNK = 16 * 1024 * 1024
UNKNOWN_MIN, UNKNOWN_MAX = 1, 100000
AUTO_LIST = 20

MEM_COMMIT, MEM_PRIVATE, MEM_MAPPED = 0x1000, 0x20000, 0x40000
PAGE_READWRITE, PAGE_EXECUTE_READWRITE, PAGE_GUARD = 0x04, 0x40, 0x100

# nama, dtype numpy, toleransi pencocokan nilai tepat
TYPES = [("float", np.float32, 0.5), ("double", np.float64, 0.5), ("int", np.int32, 0)]

k32 = ctypes.windll.kernel32
user32 = ctypes.windll.user32


class MBI(ctypes.Structure):
    _fields_ = [
        ("BaseAddress", ctypes.c_ulonglong), ("AllocationBase", ctypes.c_ulonglong),
        ("AllocationProtect", wt.DWORD), ("_pad1", wt.DWORD),
        ("RegionSize", ctypes.c_ulonglong), ("State", wt.DWORD),
        ("Protect", wt.DWORD), ("Type", wt.DWORD), ("_pad2", wt.DWORD),
    ]


k32.VirtualQueryEx.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.POINTER(MBI), ctypes.c_size_t]
k32.VirtualQueryEx.restype = ctypes.c_size_t
k32.ReadProcessMemory.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
k32.ReadProcessMemory.restype = wt.BOOL


def list_regions(handle):
    out, addr, mbi = [], 0, MBI()
    while addr < 0x7FFFFFFFFFFF:
        if not k32.VirtualQueryEx(handle, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi)):
            break
        writable = mbi.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE)
        if (mbi.State == MEM_COMMIT and mbi.Type in (MEM_PRIVATE, MEM_MAPPED)
                and writable and not (mbi.Protect & PAGE_GUARD)):
            for off in range(0, mbi.RegionSize, CHUNK):
                out.append((mbi.BaseAddress + off, min(CHUNK, mbi.RegionSize - off)))
        addr = mbi.BaseAddress + mbi.RegionSize
    return out


def read_raw(handle, base, size):
    buf = bytearray(size)
    cbuf = (ctypes.c_char * size).from_buffer(buf)
    n = ctypes.c_size_t()
    if not k32.ReadProcessMemory(handle, ctypes.c_void_p(base), cbuf, size, ctypes.byref(n)) or n.value != size:
        return None
    return bytes(buf)


class Progress:
    """Progress bar realtime di satu baris: [#####-----] 45.0%  1,234/2,740 MB  sisa ~12 dtk"""

    def __init__(self, total_bytes, label):
        self.total = max(total_bytes, 1)
        self.done = 0
        self.label = label
        self.start = time.time()
        self.last = 0

    def add(self, n, force=False):
        self.done += n
        now = time.time()
        if not force and now - self.last < 0.2:
            return
        self.last = now
        pct = min(self.done * 100 / self.total, 100)
        elapsed = now - self.start
        eta = elapsed / self.done * (self.total - self.done) if self.done else 0
        filled = int(pct / 5)
        bar = "#" * filled + "-" * (20 - filled)
        print(f"\r  {self.label} [{bar}] {pct:5.1f}%  "
              f"{self.done / 2**20:,.0f}/{self.total / 2**20:,.0f} MB  sisa ~{eta:3.0f} dtk   ",
              end="", flush=True)

    def finish(self):
        self.done = self.total
        self.add(0, force=True)
        print(f"\n  Selesai dalam {time.time() - self.start:.1f} detik.")


class TypeScanner:
    """Kandidat untuk satu tipe data."""

    def __init__(self, pm, name, dtype, tol):
        self.pm, self.name, self.dtype, self.tol = pm, name, dtype, tol
        self.size = np.dtype(dtype).itemsize
        self.cands = {}   # (base, size) -> (index, nilai lama)

    def view(self, raw):
        return np.frombuffer(raw, dtype=self.dtype, count=len(raw) // self.size)

    def match_exact(self, arr, exact):
        if self.tol:
            return np.abs(arr - exact) <= self.tol
        return arr == int(round(exact))

    def read(self, addr):
        return {"float": self.pm.read_float, "double": self.pm.read_double,
                "int": self.pm.read_int}[self.name](addr)

    def write(self, addr, v):
        if self.name == "float":
            self.pm.write_float(addr, float(v))
        elif self.name == "double":
            self.pm.write_double(addr, float(v))
        else:
            self.pm.write_int(addr, int(v))

    def total(self):
        return sum(len(i) for i, _ in self.cands.values())

    def first(self, base, size, raw, exact):
        arr = self.view(raw)
        with np.errstate(invalid="ignore", over="ignore"):
            if exact is None:
                mask = (arr >= UNKNOWN_MIN) & (arr <= UNKNOWN_MAX)
                if self.tol:
                    mask &= np.isfinite(arr)
            else:
                mask = self.match_exact(arr, exact)
        idx = np.nonzero(mask)[0].astype(np.uint32)
        if idx.size:
            self.cands[(base, size)] = (idx, arr[idx].copy())

    def next(self, key, raw, mode, exact):
        idx, old = self.cands[key]
        cur = self.view(raw)[idx]
        with np.errstate(invalid="ignore", over="ignore"):
            if mode == "e":
                m = self.match_exact(cur, exact)
            else:
                m = {"d": cur < old, "i": cur > old, "s": cur == old, "c": cur != old}[mode]
            if self.tol:
                m &= np.isfinite(cur)
        if m.any():
            self.cands[key] = (idx[m], cur[m].copy())
        else:
            del self.cands[key]

    def addresses(self):
        for (base, _), (idx, _) in self.cands.items():
            for i in idx:
                yield base + int(i) * self.size


class Scanner:
    def __init__(self, pm):
        self.pm = pm
        self.h = pm.process_handle
        self.subs = []

    def total(self):
        return sum(s.total() for s in self.subs)

    def summary(self):
        return "  ".join(f"{s.name}={s.total():,}" for s in self.subs)

    def first_scan(self, exact=None):
        # nilai tepat -> cari di semua tipe; unknown -> float saja (hemat RAM)
        types = TYPES if exact is not None else TYPES[:1]
        self.subs = [TypeScanner(self.pm, *t) for t in types]
        regions = list_regions(self.h)
        prog = Progress(sum(sz for _, sz in regions), f"Scan {len(regions)} blok")
        fail = 0
        for base, size in regions:
            raw = read_raw(self.h, base, size)
            if raw is None:
                fail += 1
            else:
                for s in self.subs:
                    s.first(base, size, raw, exact)
            prog.add(size)
        prog.finish()
        if fail:
            print(f"  ({fail} blok gagal dibaca)")
        return self.total()

    def next_scan(self, mode, exact=None):
        keys = set()
        for s in self.subs:
            keys.update(s.cands.keys())
        prog = Progress(sum(k[1] for k in keys), "Rescan")
        fail = 0
        for key in keys:
            raw = read_raw(self.h, *key)
            prog.add(key[1])
            for s in self.subs:
                if key not in s.cands:
                    continue
                if raw is None:
                    del s.cands[key]
                else:
                    s.next(key, raw, mode, exact)
            if raw is None:
                fail += 1
        prog.finish()
        if fail:
            print(f"  ({fail} blok gagal dibaca)")
        return self.total()

    def candidates(self, limit):
        res = []
        for s in self.subs:
            for a in s.addresses():
                res.append((s, a))
                if len(res) >= limit:
                    return res
        return res


# ------------------------------------------------------------------- aksi
def key_pressed(vk):
    return user32.GetAsyncKeyState(vk) & 0x1


def safe_read(s, a):
    try:
        return s.read(a)
    except pymem.exception.MemoryReadError:
        return None


def test_write(s, a):
    old = safe_read(s, a)
    if old is None:
        print("  Alamat tidak bisa dibaca.")
        return
    test = old + 7
    try:
        s.write(a, test)
    except pymem.exception.MemoryWriteError:
        print("  GAGAL menulis (memori diproteksi).")
        return
    time.sleep(0.5)
    now = safe_read(s, a)
    print(f"  Sebelum={old:g}  ditulis={test:g}  setelah 0.5 dtk={now}")
    if now is not None and abs(now - test) < 1e-3:
        print("  Nilai nempel. Cek di game: angka darah ikut naik 7?")
    else:
        print("  Nilai dikembalikan game -> kemungkinan cuma salinan, coba kandidat lain.")


def lock(targets):
    vals = [v for v in (safe_read(s, a) for s, a in targets) if v is not None]
    default = max(vals) if vals else 100
    t = input(f"  Kunci darah di nilai berapa? (Enter = {default:g}): ").strip()
    target = float(t) if t else default
    print(f"  Mengunci {len(targets)} alamat di {target:g}. F1 = ON/OFF, F2 = kembali.")
    key_pressed(0x70); key_pressed(0x71)
    enabled, fails, last = True, 0, 0
    while True:
        if key_pressed(0x70):
            enabled = not enabled
            print("  Lock:", "ON" if enabled else "OFF")
        if key_pressed(0x71):
            return
        if enabled:
            for s, a in targets:
                v = safe_read(s, a)
                if v is not None and v < target:
                    try:
                        s.write(a, target)
                    except pymem.exception.MemoryWriteError:
                        fails += 1
        if time.time() - last > 1:
            vs = ", ".join(str(safe_read(s, a)) for s, a in targets[:5])
            print(f"  nilai: {vs}{' ...' if len(targets) > 5 else ''} | gagal tulis: {fails}")
            last = time.time()
        time.sleep(0.03)


def show_and_pick(sc):
    cands = sc.candidates(AUTO_LIST)
    for n, (s, a) in enumerate(cands, 1):
        print(f"  [{n}] {s.name:6} 0x{a:X} = {safe_read(s, a)}")
    print("  nomor = kunci | t<nomor> = tes tulis (mis. t3) | a = kunci semua | Enter = lanjut")
    c = input("  Pilih: ").strip().lower()
    if c == "a":
        lock(cands)
    elif c.startswith("t") and c[1:].isdigit() and 1 <= int(c[1:]) <= len(cands):
        test_write(*cands[int(c[1:]) - 1])
    elif c.isdigit() and 1 <= int(c) <= len(cands):
        lock([cands[int(c) - 1]])


def parse_num(s):
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None


def session(pm):
    sc = Scanner(pm)
    while True:
        print("\n=== SCAN PERTAMA ===")
        s = input("  Ketik angka darah di layar (mis. 1200), atau 'u' kalau tidak ada angka: ").strip().lower()
        if s == "u":
            total = sc.first_scan()
        elif parse_num(s) is not None:
            total = sc.first_scan(exact=parse_num(s))
        else:
            continue
        print(f"  Kandidat: {total:,}  ({sc.summary()})")

        while total:
            if total <= AUTO_LIST:
                show_and_pick(sc)
            print("\n  >> Balik ke game, buat darah BERUBAH (kena damage / heal), lalu balik ke sini.")
            print("  Ketik angka darah baru, atau: d=turun i=naik s=tetap c=berubah l=lihat r=ulang")
            m = input("  Input: ").strip().lower()
            if m == "r":
                break
            if m == "l":
                show_and_pick(sc)
                continue
            num = parse_num(m)
            if num is not None:
                total = sc.next_scan("e", num)
            elif m in ("d", "i", "s", "c"):
                total = sc.next_scan(m)
            else:
                continue
            print(f"  Kandidat: {total:,}  ({sc.summary()})")
        else:
            print("  Kandidat habis -> mulai ulang. Pastikan angka yang diketik sama persis dengan di layar.")


def main():
    while True:
        print(f"Menunggu {PROCESS_NAME} ...")
        while pymem.process.process_from_name(PROCESS_NAME) is None:
            time.sleep(1)
        input("Game terdeteksi! Masuk ke gameplay dulu, lalu tekan Enter.")
        try:
            pm = pymem.Pymem(PROCESS_NAME)
            print(f"Terhubung (PID {pm.process_id}).")
            session(pm)
        except (pymem.exception.ProcessError, pymem.exception.CouldNotOpenProcess) as e:
            print("Koneksi ke game putus:", e)
        except KeyboardInterrupt:
            print("\nKeluar.")
            return


if __name__ == "__main__":
    main()
