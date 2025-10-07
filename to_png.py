import argparse, os, sys
from pathlib import Path
from PIL import Image, ImageOps

# รองรับ HEIC/HEIF/AVIF หากติดตั้ง pillow-heif
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

def is_image(path: Path) -> bool:
    # พยายามเปิดจริง ๆ เพื่อกันไฟล์ที่นามสกุลปลอม
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False

def convert_one(in_path: Path, out_path: Path, overwrite=False, apply_exif=False):
    if (not overwrite) and out_path.exists():
        return "skip_exists"

    try:
        with Image.open(in_path) as im:
            # เก็บข้อมูลที่สำคัญ (ไม่กระทบขนาดพิกเซล)
            icc = im.info.get("icc_profile")
            exif_bytes = None
            try:
                exif = im.getexif()
                if exif and len(exif) > 0:
                    exif_bytes = exif.tobytes()
            except Exception:
                exif_bytes = None

            if apply_exif:
                # หมายเหตุ: ถ้ามีการหมุนแก้ EXIF orientation อาจทำให้กว้าง/สูงสลับกัน
                im = ImageOps.exif_transpose(im)

            # ไม่เปลี่ยนขนาดพิกเซล: ไม่เรียก resize ใด ๆ ทั้งสิ้น
            # โหมดสี: PNG รองรับ 1, L, LA, P, RGB, RGBA เป็นต้น
            mode = im.mode
            if mode == "CMYK":
                # แปลงเป็น RGB เพื่อความเข้ากันได้ (สีอาจต่างเล็กน้อย ขึ้นกับโปรไฟล์)
                im = im.convert("RGB")
            elif mode not in ("1", "L", "LA", "P", "RGB", "RGBA"):
                # โหมดอื่น ๆ แปลงให้เหมาะสม โดยคงความโปร่งใสถ้ามี
                im = im.convert("RGBA" if ("transparency" in im.info or mode.endswith("A")) else "RGB")

            out_path.parent.mkdir(parents=True, exist_ok=True)

            save_kwargs = dict(format="PNG", optimize=True, compress_level=6)
            if icc:
                save_kwargs["icc_profile"] = icc
            if exif_bytes:
                # PNG รองรับ eXIf chunk ได้บน Pillow รุ่นใหม่ ๆ
                save_kwargs["exif"] = exif_bytes
            if "dpi" in im.info:
                save_kwargs["dpi"] = im.info["dpi"]

            im.save(out_path, **save_kwargs)
            return "ok"
    except Exception as e:
        return f"error: {e}"

def main():
    ap = argparse.ArgumentParser(description="Convert all images to PNG without changing pixel size or visual quality.")
    ap.add_argument("src", help="โฟลเดอร์ต้นทาง")
    ap.add_argument("dst", help="โฟลเดอร์ปลายทาง (ไฟล์จะเป็น .png)")
    ap.add_argument("--no-recursive", action="store_true", help="ไม่เดินโฟลเดอร์ย่อย")
    ap.add_argument("--overwrite", action="store_true", help="เขียนทับถ้ามีไฟล์อยู่แล้ว")
    ap.add_argument("--apply-exif", action="store_true",
                    help="ปรับทิศทางตาม EXIF (อาจทำให้กว้าง/สูงสลับกันได้)")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        print(f"ไม่พบโฟลเดอร์: {src}", file=sys.stderr)
        sys.exit(1)

    files = []
    if args.no_recursive:
        files = [p for p in src.iterdir() if p.is_file()]
    else:
        files = [p for p in src.rglob("*") if p.is_file()]

    total = ok = skipped = err = 0
    for p in files:
        total += 1
        if not is_image(p):
            skipped += 1
            continue

        rel = p.relative_to(src) if p.is_relative_to(src) else p.name
        out_path = dst / Path(rel).with_suffix(".png")
        res = convert_one(p, out_path, overwrite=args.overwrite, apply_exif=args.apply_exif)
        if res == "ok":
            ok += 1
        elif res == "skip_exists":
            skipped += 1
        elif res.startswith("error"):
            err += 1
            print(f"[ERROR] {p} -> {res}", file=sys.stderr)
        else:
            skipped += 1

    print(f"done. total={total}, converted={ok}, skipped={skipped}, errors={err}")

if __name__ == "__main__":
    main()

    #วิธีการใช้งาน python to_png.py "C:\Users\flukt\AI\Test_Project\Dataset\drug_generic.or_notpng" "C:\Users\flukt\AI\Test_Project\Dataset\drug_generic" ในcmd