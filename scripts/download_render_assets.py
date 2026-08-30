from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from urllib.request import Request
from urllib.request import urlopen


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
ASSET_DIR = WORKSPACE_DIR / "assets" / "polyhaven"
AMBIENTCG_DIR = WORKSPACE_DIR / "assets" / "ambientcg"
BOOK_MODEL_DIR = WORKSPACE_DIR / "assets" / "book_models"
USER_AGENT = "synthetic-blender-asset-downloader/1.0"

ASSETS = {
    "brown_photostudio_05": {
        "type": "hdri",
        "source_url": "https://polyhaven.com/a/brown_photostudio_05",
        "files": {
            "brown_photostudio_05_2k.hdr": "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/brown_photostudio_05_2k.hdr",
        },
    },
    "wooden_lounge": {
        "type": "hdri",
        "source_url": "https://polyhaven.com/a/wooden_lounge",
        "recommended_for": "cozy indoor home study / library with wooden furniture, couch, lamp, and warm lighting",
        "files": {
            "wooden_lounge_2k.hdr": "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/wooden_lounge_2k.hdr",
        },
    },
    "lythwood_room": {
        "type": "hdri",
        "source_url": "https://polyhaven.com/a/lythwood_room",
        "recommended_for": "realistic residential room interior with windows, neutral walls, and soft natural light",
        "files": {
            "lythwood_room_2k.hdr": "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/lythwood_room_2k.hdr",
        },
    },
    "wood_floor_worn": {
        "type": "texture",
        "source_url": "https://polyhaven.com/a/wood_floor_worn",
        "recommended_for": "worn floor variation with natural scratches, knots, seams, and color variation",
        "files": {
            "wood_floor_worn_ao_4k.jpg": "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/4k/wood_floor_worn/wood_floor_worn_ao_4k.jpg",
            "wood_floor_worn_diff_4k.jpg": "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/4k/wood_floor_worn/wood_floor_worn_diff_4k.jpg",
            "wood_floor_worn_disp_4k.png": "https://dl.polyhaven.org/file/ph-assets/Textures/png/4k/wood_floor_worn/wood_floor_worn_disp_4k.png",
            "wood_floor_worn_nor_gl_4k.jpg": "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/4k/wood_floor_worn/wood_floor_worn_nor_gl_4k.jpg",
            "wood_floor_worn_rough_4k.jpg": "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/4k/wood_floor_worn/wood_floor_worn_rough_4k.jpg",
        },
    },
    "wood_table": {
        "type": "texture",
        "source_url": "https://polyhaven.com/a/wood_table",
        "recommended_for": "wooden block faces; 4k maps make the block less uniform in close-ups",
        "files": {
            "wood_table_ao_4k.jpg": "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/4k/wood_table/wood_table_ao_4k.jpg",
            "wood_table_diff_4k.jpg": "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/4k/wood_table/wood_table_diff_4k.jpg",
            "wood_table_disp_4k.png": "https://dl.polyhaven.org/file/ph-assets/Textures/png/4k/wood_table/wood_table_disp_4k.png",
            "wood_table_nor_gl_4k.jpg": "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/4k/wood_table/wood_table_nor_gl_4k.jpg",
            "wood_table_rough_4k.jpg": "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/4k/wood_table/wood_table_rough_4k.jpg",
        },
    },
    "stained_pine": {
        "type": "texture",
        "source_url": "https://polyhaven.com/a/stained_pine",
        "recommended_for": "warmer varnished pine option for wood blocks with less striped close-up appearance",
        "files": {
            "stained_pine_ao_4k.jpg": "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/4k/stained_pine/stained_pine_ao_4k.jpg",
            "stained_pine_diff_4k.jpg": "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/4k/stained_pine/stained_pine_diff_4k.jpg",
            "stained_pine_disp_4k.png": "https://dl.polyhaven.org/file/ph-assets/Textures/png/4k/stained_pine/stained_pine_disp_4k.png",
            "stained_pine_nor_gl_4k.jpg": "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/4k/stained_pine/stained_pine_nor_gl_4k.jpg",
            "stained_pine_rough_4k.jpg": "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/4k/stained_pine/stained_pine_rough_4k.jpg",
        },
    },
    "concrete_floor_damaged_01": {
        "type": "texture",
        "source_url": "https://polyhaven.com/a/concrete_floor_damaged_01",
        "recommended_for": "weathered outdoor courtyard/patio ground for car_ramp_climb",
        "files": {
            "concrete_floor_damaged_01_ao_4k.jpg": "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/4k/concrete_floor_damaged_01/concrete_floor_damaged_01_ao_4k.jpg",
            "concrete_floor_damaged_01_diff_4k.jpg": "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/4k/concrete_floor_damaged_01/concrete_floor_damaged_01_diff_4k.jpg",
            "concrete_floor_damaged_01_disp_4k.png": "https://dl.polyhaven.org/file/ph-assets/Textures/png/4k/concrete_floor_damaged_01/concrete_floor_damaged_01_disp_4k.png",
            "concrete_floor_damaged_01_nor_gl_4k.jpg": "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/4k/concrete_floor_damaged_01/concrete_floor_damaged_01_nor_gl_4k.jpg",
            "concrete_floor_damaged_01_rough_4k.jpg": "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/4k/concrete_floor_damaged_01/concrete_floor_damaged_01_rough_4k.jpg",
        },
    },
    "red_brick": {
        "type": "texture",
        "source_url": "https://polyhaven.com/a/red_brick",
        "recommended_for": "brick wall backdrop for the car_ramp_climb scene",
        "files": {
            "red_brick_ao_4k.jpg": "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/4k/red_brick/red_brick_ao_4k.jpg",
            "red_brick_diff_4k.jpg": "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/4k/red_brick/red_brick_diff_4k.jpg",
            "red_brick_disp_4k.png": "https://dl.polyhaven.org/file/ph-assets/Textures/png/4k/red_brick/red_brick_disp_4k.png",
            "red_brick_nor_gl_4k.jpg": "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/4k/red_brick/red_brick_nor_gl_4k.jpg",
            "red_brick_rough_4k.jpg": "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/4k/red_brick/red_brick_rough_4k.jpg",
        },
    },
    "book_cover_01": {
        "type": "texture",
        "source_url": "https://polyhaven.com/a/decorative_book_set_01",
        "recommended_for": "book cover - classic poetry (18th Century gold frame)",
        "files": {
            "book_softcover_01_cover12_4k.jpg": "https://dl.polyhaven.org/file/ph-assets/Models/jpg/4k/decorative_book_set_01/book_softcover_01_cover12_4k.jpg",
        },
    },
    "book_cover_02": {
        "type": "texture",
        "source_url": "https://polyhaven.com/a/decorative_book_set_01",
        "recommended_for": "book cover - classic mystery thriller",
        "files": {
            "book_softcover_01_cover14_4k.jpg": "https://dl.polyhaven.org/file/ph-assets/Models/jpg/4k/decorative_book_set_01/book_softcover_01_cover14_4k.jpg",
        },
    },
    "book_cover_03": {
        "type": "texture",
        "source_url": "https://polyhaven.com/a/decorative_book_set_01",
        "recommended_for": "book cover - classic gothic novel",
        "files": {
            "book_softcover_01_cover24_4k.jpg": "https://dl.polyhaven.org/file/ph-assets/Models/jpg/4k/decorative_book_set_01/book_softcover_01_cover24_4k.jpg",
        },
    },
    "book_cover_04": {
        "type": "texture",
        "source_url": "https://polyhaven.com/a/decorative_book_set_01",
        "recommended_for": "book cover - classic adventure novel",
        "files": {
            "book_softcover_01_cover28_4k.jpg": "https://dl.polyhaven.org/file/ph-assets/Models/jpg/4k/decorative_book_set_01/book_softcover_01_cover28_4k.jpg",
        },
    },
    "book_cover_05": {
        "type": "texture",
        "source_url": "https://polyhaven.com/a/decorative_book_set_01",
        "recommended_for": "book cover - classic literature brown leather",
        "files": {
            "book_softcover_01_cover26_4k.jpg": "https://dl.polyhaven.org/file/ph-assets/Models/jpg/4k/decorative_book_set_01/book_softcover_01_cover26_4k.jpg",
        },
    },
    "book_cover_plain": {
        "type": "texture",
        "source_url": "https://polyhaven.com/a/book_pattern",
        "recommended_for": "book cover - plain olive woven cotton fabric without text",
        "files": {
            "book_pattern_col1_4k.jpg": "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/4k/book_pattern/book_pattern_col1_4k.jpg",
        },
    },
}
AMBIENTCG_ZIP_ASSETS = {
    "Asphalt031": {
        "type": "texture",
        "source": "ambientCG",
        "license": "CC0",
        "source_url": "https://ambientcg.com/view?id=Asphalt031",
        "zip_url": "https://ambientcg.com/get?file=Asphalt031_4K-JPG.zip",
        "recommended_for": "grey asphalt/road ramp surface for car_ramp_climb (dark variant reuses this "
        "same texture, darkened via a Hue/Saturation node rather than a second download)",
        "files": {
            "Asphalt031_4K-JPG_Color.jpg": "Asphalt031_4K-JPG_Color.jpg",
            "Asphalt031_4K-JPG_Displacement.jpg": "Asphalt031_4K-JPG_Displacement.jpg",
            "Asphalt031_4K-JPG_NormalGL.jpg": "Asphalt031_4K-JPG_NormalGL.jpg",
            "Asphalt031_4K-JPG_Roughness.jpg": "Asphalt031_4K-JPG_Roughness.jpg",
        },
    },
    "Leather039": {
        "type": "texture",
        "source": "ambientCG",
        "license": "CC0",
        "source_url": "https://ambientcg.com/view?id=Leather039",
        "zip_url": "https://ambientcg.com/get?file=Leather039_4K-JPG.zip",
        "recommended_for": "high-quality realistic leather grain for basketball / sports ball surface",
        "files": {
            "Leather039_4K-JPG_Color.jpg": "Leather039_4K-JPG_Color.jpg",
            "Leather039_4K-JPG_Displacement.jpg": "Leather039_4K-JPG_Displacement.jpg",
            "Leather039_4K-JPG_NormalGL.jpg": "Leather039_4K-JPG_NormalGL.jpg",
            "Leather039_4K-JPG_Roughness.jpg": "Leather039_4K-JPG_Roughness.jpg",
        },
    },
    "Rubber002": {
        "type": "texture",
        "source": "ambientCG",
        "license": "CC0",
        "source_url": "https://ambientcg.com/view?id=Rubber002",
        "zip_url": "https://ambientcg.com/get?file=Rubber002_4K-JPG.zip",
        "recommended_for": "seamless rubber micro-surface for ball roughness, normal, and height detail",
        "files": {
            "Rubber002_4K-JPG_Color.jpg": "Rubber002_4K-JPG_Color.jpg",
            "Rubber002_4K-JPG_Displacement.jpg": "Rubber002_4K-JPG_Displacement.jpg",
            "Rubber002_4K-JPG_NormalGL.jpg": "Rubber002_4K-JPG_NormalGL.jpg",
            "Rubber002_4K-JPG_Roughness.jpg": "Rubber002_4K-JPG_Roughness.jpg",
        },
    },
    "Marble012": {
        "type": "texture",
        "source": "ambientCG",
        "license": "CC0",
        "source_url": "https://ambientcg.com/view?id=Marble012",
        "zip_url": "https://ambientcg.com/get?file=Marble012_4K-JPG.zip",
        "recommended_for": "white marble with grey veins for glass marble surface color and detail",
        "files": {
            "Marble012_4K-JPG_Color.jpg": "Marble012_4K-JPG_Color.jpg",
            "Marble012_4K-JPG_Displacement.jpg": "Marble012_4K-JPG_Displacement.jpg",
            "Marble012_4K-JPG_NormalGL.jpg": "Marble012_4K-JPG_NormalGL.jpg",
            "Marble012_4K-JPG_Roughness.jpg": "Marble012_4K-JPG_Roughness.jpg",
        },
    },
    "Metal032": {
        "type": "texture",
        "source": "ambientCG",
        "license": "CC0",
        "source_url": "https://ambientcg.com/view?id=Metal032",
        "zip_url": "https://ambientcg.com/get?file=Metal032_4K-JPG.zip",
        "recommended_for": "smooth grey steel surface for small steel ball rendering",
        "files": {
            "Metal032_4K-JPG_Color.jpg": "Metal032_4K-JPG_Color.jpg",
            "Metal032_4K-JPG_Displacement.jpg": "Metal032_4K-JPG_Displacement.jpg",
            "Metal032_4K-JPG_Metalness.jpg": "Metal032_4K-JPG_Metalness.jpg",
            "Metal032_4K-JPG_NormalGL.jpg": "Metal032_4K-JPG_NormalGL.jpg",
            "Metal032_4K-JPG_Roughness.jpg": "Metal032_4K-JPG_Roughness.jpg",
        },
    },
}

BOOK_ZIP_MODELS = {
    # 外部下载源已移除（Sketchfab 需要登录，free3d 网络不可达）
    # 书模型现在通过 Blender 脚本程序化生成
    # 生成脚本: generate_book_model.py
    # 输出文件: assets/book_models/generated_book/book.glb
}


def download(url: str, path: Path) -> dict[str, object]:
    if path.exists() and path.stat().st_size > 0:
        return {"path": str(path.relative_to(WORKSPACE_DIR)), "size": path.stat().st_size, "cached": True}
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5()
    total = 0
    request = Request(url, headers={"User-Agent": USER_AGENT})
    partial_path = path.with_suffix(path.suffix + ".part")
    try:
        with urlopen(request, timeout=120) as response, partial_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                total += len(chunk)
        partial_path.replace(path)
    except Exception as exc:
        if partial_path.exists():
            partial_path.unlink(missing_ok=True)
        return {
            "path": str(path.relative_to(WORKSPACE_DIR)),
            "size": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "url": url,
        }
    return {
        "path": str(path.relative_to(WORKSPACE_DIR)),
        "size": total,
        "md5": digest.hexdigest(),
        "cached": False,
    }


def file_metadata(path: Path, *, cached: bool) -> dict[str, object]:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return {
        "path": str(path.relative_to(WORKSPACE_DIR)),
        "size": path.stat().st_size,
        "md5": digest.hexdigest(),
        "cached": cached,
    }


def download_ambientcg_zip_asset(asset_name: str, asset: dict[str, object]) -> dict[str, object]:
    asset_dir = AMBIENTCG_DIR / asset_name
    asset_dir.mkdir(parents=True, exist_ok=True)
    files = asset["files"]
    assert isinstance(files, dict)
    cached = all((asset_dir / filename).exists() and (asset_dir / filename).stat().st_size > 0 for filename in files)
    if not cached:
        zip_path = asset_dir / f"{asset_name}_download.zip.part"
        print(f"Downloading {asset_name}/{zip_path.name}")
        result = download(str(asset["zip_url"]), zip_path)
        if "error" in result:
            print(f"[SKIP] Failed to download {asset_name}: {result['error']}")
            return {
                "type": asset["type"],
                "source": asset["source"],
                "license": asset["license"],
                "source_url": asset["source_url"],
                "recommended_for": asset.get("recommended_for"),
                "error": result["error"],
                "files": {},
            }
        try:
            with zipfile.ZipFile(zip_path) as archive:
                for filename, member_name in files.items():
                    target = asset_dir / filename
                    print(f"Extracting {asset_name}/{filename}")
                    with archive.open(str(member_name)) as source, target.open("wb") as handle:
                        while True:
                            chunk = source.read(1024 * 1024)
                            if not chunk:
                                break
                            handle.write(chunk)
        except (zipfile.BadZipFile, KeyError, OSError) as exc:
            print(f"[SKIP] Failed to extract {asset_name}: {exc}")
            return {
                "type": asset["type"],
                "source": asset["source"],
                "license": asset["license"],
                "source_url": asset["source_url"],
                "recommended_for": asset.get("recommended_for"),
                "error": f"extract: {exc}",
                "files": {},
            }
        zip_path.unlink(missing_ok=True)

    return {
        "type": asset["type"],
        "source": asset["source"],
        "license": asset["license"],
        "source_url": asset["source_url"],
        "recommended_for": asset.get("recommended_for"),
        "files": {
            filename: file_metadata(asset_dir / filename, cached=cached)
            for filename in files
        },
    }


def download_book_model(asset_name: str, asset: dict[str, object]) -> dict[str, object]:
    asset_dir = BOOK_MODEL_DIR / asset_name
    asset_dir.mkdir(parents=True, exist_ok=True)

    main_file = asset.get("expected_main_file")
    expected = asset_dir / main_file if main_file else None

    if expected is not None and expected.exists() and expected.stat().st_size > 0:
        found_files = {
            p.name: file_metadata(p, cached=True)
            for p in sorted(asset_dir.iterdir())
            if p.is_file() and p.suffix.lower() in {".obj", ".fbx", ".glb", ".gltf", ".mtl", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tga"}
        }
        return {
            "type": asset["type"],
            "source": asset["source"],
            "license": asset["license"],
            "source_url": asset["source_url"],
            "recommended_for": asset.get("recommended_for"),
            "main_file": str(expected.relative_to(WORKSPACE_DIR)),
            "approx_size_m": asset.get("approx_size_m"),
            "files": found_files,
        }

    zip_path = asset_dir / f"{asset_name}_download.zip"
    print(f"Downloading {asset_name} -> {zip_path}")
    result = download(str(asset["zip_url"]), zip_path)
    if "error" in result:
        print(f"[SKIP] Failed to download {asset_name}: {result['error']}")
        return {
            "type": asset["type"],
            "source": asset["source"],
            "license": asset["license"],
            "source_url": asset["source_url"],
            "recommended_for": asset.get("recommended_for"),
            "approx_size_m": asset.get("approx_size_m"),
            "error": result["error"],
            "files": {},
        }

    extracted_files: dict[str, dict[str, object]] = {}
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.namelist():
                if not member or member.endswith("/"):
                    continue
                target = asset_dir / Path(member).name
                print(f"  extracting {member} -> {target.name}")
                with archive.open(member) as source, target.open("wb") as handle:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                extracted_files[target.name] = file_metadata(target, cached=False)
    except (zipfile.BadZipFile, OSError) as exc:
        print(f"[SKIP] Failed to extract {asset_name}: {exc}")
        return {
            "type": asset["type"],
            "source": asset["source"],
            "license": asset["license"],
            "source_url": asset["source_url"],
            "recommended_for": asset.get("recommended_for"),
            "approx_size_m": asset.get("approx_size_m"),
            "error": f"extract: {exc}",
            "files": {},
        }
    finally:
        if zip_path.exists():
            zip_path.unlink(missing_ok=True)

    main_path: str | None = None
    for suffix in (".obj", ".fbx", ".glb", ".gltf"):
        for name, meta in extracted_files.items():
            if name.lower().endswith(suffix):
                main_path = meta["path"]
                break
        if main_path:
            break
    if main_path is None and extracted_files:
        main_path = next(iter(extracted_files.values()))["path"]

    return {
        "type": asset["type"],
        "source": asset["source"],
        "license": asset["license"],
        "source_url": asset["source_url"],
        "recommended_for": asset.get("recommended_for"),
        "approx_size_m": asset.get("approx_size_m"),
        "main_file": main_path,
        "files": extracted_files,
    }


def main() -> None:
    manifest: dict[str, object] = {
        "source": "Poly Haven",
        "license": "CC0",
        "assets": {},
    }
    for asset_name, asset in ASSETS.items():
        asset_dir = ASSET_DIR / asset_name
        files = {}
        for filename, url in asset["files"].items():
            print(f"Downloading {asset_name}/{filename}")
            files[filename] = download(url, asset_dir / filename)
        manifest["assets"][asset_name] = {
            "type": asset["type"],
            "source_url": asset["source_url"],
            "recommended_for": asset.get("recommended_for"),
            "files": files,
        }
    manifest_path = ASSET_DIR / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {manifest_path}")

    ambient_manifest: dict[str, object] = {
        "source": "ambientCG",
        "license": "CC0",
        "assets": {},
    }
    for asset_name, asset in AMBIENTCG_ZIP_ASSETS.items():
        ambient_manifest["assets"][asset_name] = download_ambientcg_zip_asset(asset_name, asset)
    ambient_manifest_path = AMBIENTCG_DIR / "manifest.json"
    ambient_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    ambient_manifest_path.write_text(json.dumps(ambient_manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {ambient_manifest_path}")

    # 生成书模型（使用 Blender 脚本）
    book_manifest: dict[str, object] = {
        "source": "generated",
        "license": "procedurally generated",
        "assets": {},
    }
    BOOK_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 调用 Blender 生成书模型
    import subprocess
    import shutil
    
    blender = shutil.which("blender")
    if blender:
        generate_script = WORKSPACE_DIR / "scripts" / "generate_book_model.py"
        if generate_script.exists():
            print("Generating book model with Blender...")
            result = subprocess.run(
                [blender, "--background", "--python", str(generate_script)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                book_model_path = BOOK_MODEL_DIR / "generated_book" / "book.glb"
                if book_model_path.exists():
                    book_manifest["assets"]["generated_book"] = {
                        "type": "model",
                        "source": "generated",
                        "license": "procedurally generated",
                        "source_url": "scripts/generate_book_model.py",
                        "recommended_for": "高质量程序化书模型，包含封面、书脊、书页细节",
                        "main_file": str(book_model_path.relative_to(WORKSPACE_DIR)),
                        "approx_size_m": [0.22, 0.15, 0.025],
                    }
                    print(f"Generated: {book_model_path}")
            else:
                print(f"[SKIP] Failed to generate book model: {result.stderr}")
        else:
            print(f"[SKIP] Generate script not found: {generate_script}")
    else:
        print("[SKIP] Blender not found, cannot generate book model")
    
    book_manifest_path = BOOK_MODEL_DIR / "manifest.json"
    book_manifest_path.write_text(json.dumps(book_manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {book_manifest_path}")


if __name__ == "__main__":
    main()
