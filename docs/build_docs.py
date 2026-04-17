import argparse
import logging
import os
import re
import shutil
import subprocess
import sys

import yaml
from sphinx.application import Sphinx


docs_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(docs_dir, "source")
conf_dir = source_dir
build_dir = os.path.join(docs_dir, "_build")
linkcheck_dir = os.path.join(build_dir, "linkcheck")
html_dir = os.path.join(build_dir, "html")
doctree_dir = os.path.join(build_dir, "doctrees")
versions_file = os.path.join(docs_dir, "versions.yaml")


logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build WecOptTool Sphinx docs in debug or production mode."
    )
    parser.add_argument(
        "-b",
        "--build",
        choices=["debug", "production"],
        default="debug",
        help="Build mode: debug builds current branch, production builds all configured versions via sphinx-multiversion.",
    )
    parser.add_argument(
        "--skip-notebook-execution",
        action="store_true",
        help="Include tutorial/example pages but do not execute notebooks during the build.",
    )
    return parser.parse_args()


def _run_command(args: list[str], cwd: str | None = None) -> None:
    result = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        details = stderr or stdout or "command failed"
        raise RuntimeError(f"{' '.join(args)} failed: {details}")


def _remove_if_exists(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)


def _load_versions() -> dict[str, str]:
    with open(versions_file, "r", encoding="utf-8") as v_file:
        versions = yaml.safe_load(v_file)

    if not isinstance(versions, dict) or not versions:
        raise ValueError("versions.yaml must contain a non-empty mapping of version name to git ref")
    if "latest" not in versions:
        raise ValueError("versions.yaml must include a 'latest' entry")

    for name, ref in versions.items():
        if not isinstance(name, str) or not isinstance(ref, str):
            raise ValueError("All entries in versions.yaml must map string names to string git refs")
        if not name.strip() or not ref.strip():
            raise ValueError("Version names and git refs in versions.yaml cannot be empty")

    return versions


def _build_with_sphinx_multiversion(versions: dict[str, str]) -> None:
    refs = sorted({ref for ref in versions.values()})
    whitelist = "^({})$".format("|".join(re.escape(ref) for ref in refs))

    logger.info("Running sphinx-multiversion for refs: %s", ", ".join(refs))
    _run_command(
        [
            sys.executable,
            "-m",
            "sphinx_multiversion",
            source_dir,
            html_dir,
            "-D",
            f"smv_branch_whitelist={whitelist}",
            "-D",
            f"smv_tag_whitelist={whitelist}",
        ],
        cwd=docs_dir,
    )


def linkcheck() -> None:
    app = Sphinx(
        source_dir,
        conf_dir,
        linkcheck_dir,
        doctree_dir,
        "linkcheck",
        warningiserror=False,
    )
    app.build()


def html() -> None:
    app = Sphinx(
        source_dir,
        conf_dir,
        html_dir,
        doctree_dir,
        "html",
        warningiserror=True,
    )
    app.build()


def build_doc(version: str, build: str) -> None:
    os.environ["current_version"] = version
    if build != "debug":
        raise ValueError("build_doc should only be used for debug mode")

    logger.info("Running Sphinx linkcheck")
    linkcheck()
    logger.info("Building Sphinx HTML")
    html()


def move_pages_debug() -> None:
    pages_dir = os.path.join(docs_dir, "pages")
    logger.info("Publishing HTML pages to %s", pages_dir)

    _remove_if_exists(pages_dir)
    os.makedirs(os.path.dirname(pages_dir), exist_ok=True)
    shutil.copytree(html_dir, pages_dir)

    shutil.rmtree(build_dir, ignore_errors=False)
    logger.info("Publish complete")


def move_pages_multiversion(versions: dict[str, str]) -> None:
    pages_root = os.path.join(docs_dir, "pages")
    logger.info("Publishing sphinx-multiversion output to %s", pages_root)

    _remove_if_exists(pages_root)
    os.makedirs(pages_root, exist_ok=True)

    for name, ref in versions.items():
        source = os.path.join(html_dir, ref)
        if not os.path.exists(source):
            raise RuntimeError(f"Expected sphinx-multiversion output not found for ref '{ref}' at {source}")

        target = pages_root if name == "latest" else os.path.join(pages_root, name)
        if name != "latest":
            os.makedirs(os.path.dirname(target), exist_ok=True)
        _remove_if_exists(target)
        shutil.copytree(source, target)

    shutil.rmtree(build_dir, ignore_errors=False)
    logger.info("Publish complete")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    build = args.build

    if args.skip_notebook_execution:
        os.environ["WOT_DOCS_SKIP_NOTEBOOK_EXECUTION"] = "1"
        logger.info("Skipping notebook execution for this build")
    else:
        os.environ.pop("WOT_DOCS_SKIP_NOTEBOOK_EXECUTION", None)

    if build == "debug":
        logger.info("Building docs for current branch in debug mode")
        build_doc("latest", build)
        move_pages_debug()
        return

    versions = _load_versions()
    logger.info("Starting production build via sphinx-multiversion")
    _build_with_sphinx_multiversion(versions)
    move_pages_multiversion(versions)


if __name__ == "__main__":
    main()
