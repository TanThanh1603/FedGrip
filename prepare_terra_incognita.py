"""Build the DomainBed Terra Incognita subset from the official LILA archives.

Selection follows facebookresearch/DomainBed/domainbed/scripts/download.py.
Only selected images are extracted; no full CCT-20 image tree is created.
This prepares data only, not training partitions or model configuration.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tarfile

from PIL import Image

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data/_downloads/terra_incognita"
OUTPUT = ROOT / "data/terra_incognita"
LOCATIONS = ("38", "46", "100", "43")
CATEGORIES = ("bird", "bobcat", "cat", "coyote", "dog", "empty", "opossum",
              "rabbit", "raccoon", "squirrel")
BASE_URL = "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/"
IMAGE_ARCHIVE = "eccv_18_all_images_sm.tar.gz"
ANNOTATION_ARCHIVE = "eccv_18_annotations.tar.gz"
IMAGE_MD5 = "8143c17aa2a12872b66f284ff211531f"


def selection():
    data = defaultdict(list)
    with tarfile.open(CACHE / ANNOTATION_ARCHIVE) as archive:
        for member in archive:
            if member.isfile() and member.name.endswith(".json"):
                with archive.extractfile(member) as stream:
                    for key, values in json.load(stream).items():
                        data[key].extend(values)
    categories = {item["id"]: item["name"] for item in data["categories"]}
    labels = defaultdict(set)
    for annotation in data["annotations"]:
        category = categories[annotation["category_id"]]
        if category in CATEGORIES:
            labels[annotation["image_id"]].add(category)
    selected = defaultdict(set)
    for image in data["images"]:
        location = str(image["location"])
        if location not in LOCATIONS:
            continue
        name = image["file_name"]
        if Path(name).name != name:
            raise ValueError(f"Unexpected image filename: {name}")
        for category in labels[image["id"]]:
            selected[name].add(f"location_{location}/{category}/{name}")
    return dict(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect-selection", action="store_true")
    args = parser.parse_args()
    selected = selection()
    counts = Counter(path.split("/")[0] for paths in selected.values() for path in paths)
    print("Selected images:", sum(counts.values()), dict(counts), flush=True)
    if args.inspect_selection:
        return
    checksum = hashlib.md5()
    with (CACHE / IMAGE_ARCHIVE).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    actual = checksum.hexdigest()
    if actual != IMAGE_MD5:
        raise ValueError("Image archive is incomplete or differs from official checksum")
    raw = OUTPUT / "raw"
    found = set()
    with tarfile.open(CACHE / IMAGE_ARCHIVE, "r|gz") as archive:
        for member in archive:
            name = Path(member.name).name
            if not member.isfile() or name not in selected:
                continue
            with archive.extractfile(member) as source:
                payload = source.read()
            for relative in selected[name]:
                destination = raw / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    if destination.stat().st_size != member.size:
                        raise ValueError(f"Existing image size mismatch: {destination}")
                    continue
                temporary = destination.with_suffix(".partial")
                temporary.write_bytes(payload)
                temporary.replace(destination)
            found.add(name)
            if len(found) % 5000 == 0:
                print(f"Extracted {len(found)}/{len(selected)} selected images", flush=True)
    if found != set(selected):
        raise ValueError(f"Missing {len(set(selected) - found)} selected images")
    expected = {path for paths in selected.values() for path in paths}
    actual = {str(path.relative_to(raw)) for path in raw.rglob("*") if path.is_file()}
    if actual != expected:
        raise ValueError("Unexpected or missing files in dataset output")
    for relative in sorted(expected):
        with Image.open(raw / relative) as image:
            image.load()
    manifest = dict(
        dataset="terra_incognita", protocol="DomainBed subset",
        reference="https://github.com/facebookresearch/DomainBed/blob/main/domainbed/scripts/download.py",
        sources=[BASE_URL + IMAGE_ARCHIVE, BASE_URL + ANNOTATION_ARCHIVE],
        image_archive_md5=IMAGE_MD5, domains=dict(counts), classes=list(CATEGORIES),
        total_images=sum(counts.values()), unique_source_filenames=len(selected),
        images=sorted(expected),
    )
    (OUTPUT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified all {len(expected)} images: {raw}", flush=True)


if __name__ == "__main__":
    main()
