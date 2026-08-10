#!/usr/bin/env python3
"""Pre-pull SWE-bench instance evaluation Docker images.

When `run_evaluation` is called with a `namespace` (the project default is
`swebench`), it checks for an instance image locally and pulls it from the
registry if missing.  This script pulls all of those images *before* the
evaluator starts, so the evaluation phase can run without waiting on the
network for each instance.

Two image sources are supported:

* `docker` (default): official Docker Hub `swebench/sweb.eval.*` images.
  These are subject to Docker Hub's unauthenticated pull rate limit
  (100 pulls per 6 hours per IP at time of writing).

* `ghcr`: Epoch AI's mirror on the GitHub Container Registry
  (`ghcr.io/epoch-research/swe-bench.eval.*`).  Public GHCR images are not
  subject to Docker Hub's rate limits and are typically smaller due to better
  layer sharing.  Images are pulled from GHCR and re-tagged locally to the
  `swebench/sweb.eval.*` names expected by the SWE-bench harness.

The target image name calculation mirrors `swebench.harness.test_spec.TestSpec`:

    {namespace}/sweb.eval.{arch}.{instance_id}:{tag}

with all `__` sequences replaced by `_1776_` (the same transform SWE-bench
applies to make the name valid as a remote image reference).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


GHCR_REGISTRY = "ghcr.io"
GHCR_OWNER = "epoch-research"
GHCR_NAME_PREFIX = "swe-bench.eval"


def swebench_image_key(
    instance_id: str, namespace: str, arch: str, tag: str
) -> str:
    """Compute the local target image name for an instance (what SWE-bench expects)."""
    local_key = f"sweb.eval.{arch}.{instance_id.lower()}:{tag}"
    if namespace:
        # SWE-bench replaces every `__` in the image reference, but the
        # namespace itself never contains one in practice.
        return f"{namespace}/{local_key}".replace("__", "_1776_")
    return local_key


def ghcr_image_key(instance_id: str, arch: str, tag: str) -> str:
    """Compute the source image name on GHCR for an instance."""
    return f"{GHCR_REGISTRY}/{GHCR_OWNER}/{GHCR_NAME_PREFIX}.{arch}.{instance_id.lower()}:{tag}"


def load_instance_ids(subset_path: Path) -> list[str]:
    """Load instance ids from a subset file."""
    data = json.loads(subset_path.read_text(encoding="utf-8"))
    ids = data.get("instance_ids")
    if not ids:
        raise SystemExit(f"No 'instance_ids' found in {subset_path}")
    return list(ids)


def existing_images() -> set[str]:
    """Return the set of locally-present `repository:tag` images."""
    result = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def pull_image(
    target_image: str,
    source_image: str,
    timeout: int | None = None,
) -> tuple[str, bool, str]:
    """Pull one image (from source_image) and re-tag it to target_image if needed.

    Returns (target_image, success, combined_output).
    """
    output_parts: list[str] = []

    # Pull the source image.
    pull_result = subprocess.run(
        ["docker", "pull", source_image],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output_parts.append(pull_result.stdout)
    output_parts.append(pull_result.stderr)
    if pull_result.returncode != 0:
        return target_image, False, "".join(output_parts).strip()

    # If the source and target names differ, re-tag to what SWE-bench expects.
    if source_image != target_image:
        tag_result = subprocess.run(
            ["docker", "tag", source_image, target_image],
            capture_output=True,
            text=True,
        )
        output_parts.append(tag_result.stdout)
        output_parts.append(tag_result.stderr)
        if tag_result.returncode != 0:
            return target_image, False, "".join(output_parts).strip()

        # Remove the temporary source tag; the image remains because it now has
        # the target tag.  Failures here are non-fatal (e.g. tag not found).
        rmi_result = subprocess.run(
            ["docker", "rmi", source_image],
            capture_output=True,
            text=True,
        )
        output_parts.append(rmi_result.stdout)
        output_parts.append(rmi_result.stderr)

    return target_image, True, "".join(output_parts).strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-pull SWE-bench evaluation Docker images.",
    )
    parser.add_argument(
        "--subset",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "evals"
            / "swebench"
            / "subsets"
            / "verified_500.json"
        ),
        help="Path to the subset JSON file containing instance_ids",
    )
    parser.add_argument(
        "--source",
        choices=("docker", "ghcr"),
        default="docker",
        help=(
            "Image source: 'docker' for Docker Hub swebench namespace "
            "(default), 'ghcr' for Epoch AI's GitHub Container Registry mirror"
        ),
    )
    parser.add_argument(
        "--namespace",
        default="swebench",
        help='Target image namespace (use "none" for no namespace)',
    )
    parser.add_argument(
        "--arch",
        default="x86_64",
        help="Architecture tag used in image names (default: x86_64)",
    )
    parser.add_argument(
        "--tag",
        default="latest",
        help="Image tag to pull (default: latest)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of concurrent docker pull operations (default: 4)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds for each docker pull",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-pull images that are already present locally",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the images that would be pulled without pulling",
    )
    args = parser.parse_args()

    if not shutil.which("docker"):
        raise SystemExit("docker CLI not found in PATH")

    if not args.subset.exists():
        raise SystemExit(f"Subset file not found: {args.subset}")

    instance_ids = load_instance_ids(args.subset)
    target_namespace = "" if args.namespace.lower() == "none" else args.namespace

    target_images = [
        swebench_image_key(iid, target_namespace, args.arch, args.tag)
        for iid in instance_ids
    ]

    if args.source == "ghcr":
        source_images = [
            ghcr_image_key(iid, args.arch, args.tag) for iid in instance_ids
        ]
    else:
        source_images = target_images

    if args.dry_run:
        print(f"Would pull {len(instance_ids)} images:")
        for src, tgt in zip(source_images, target_images):
            if src == tgt:
                print(f"  {tgt}")
            else:
                print(f"  {src} -> {tgt}")
        return 0

    local_set = set() if args.force else existing_images()
    to_pull = [
        (src, tgt)
        for src, tgt in zip(source_images, target_images)
        if tgt not in local_set
    ]

    already_present = len(instance_ids) - len(to_pull)
    if already_present:
        print(f"{already_present} of {len(instance_ids)} images already present locally")
    if not to_pull:
        print("No images need to be pulled.")
        return 0

    print(
        f"Pulling {len(to_pull)} images from {args.source} "
        f"with {args.max_workers} workers..."
    )
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(pull_image, tgt, src, args.timeout): tgt
            for src, tgt in to_pull
        }
        for future in as_completed(futures):
            img, ok, output = future.result()
            if ok:
                print(f"[ok] {img}")
            else:
                print(f"[fail] {img}")
                if output:
                    print(output)
                failed.append(img)

    if failed:
        print(f"Failed to pull {len(failed)} images.")
        return 1
    print("All images pulled successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
