import importlib.metadata
import sys
import tomllib

from packaging.version import InvalidVersion, Version


def get_min_versions_from_pyproject() -> dict[str, str | None]:
    with open("pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)

    try:
        dependencies = pyproject["project"]["dependencies"]
    except KeyError as err:
        raise RuntimeError("No dependencies found in pyproject.toml") from err

    min_versions = {}
    for dep in dependencies:
        if ">=" in dep:
            package, version = dep.split(">=")
            min_versions[package.strip()] = version.strip()
        else:
            package = dep.split()[0].strip()
            min_versions[package] = None

    return min_versions


def check_dependencies() -> None:
    min_versions = get_min_versions_from_pyproject()

    if sys.platform == "win32":
        cmd = "./install.bat"
    else:
        cmd = "./install.sh"

    for package, min_version in min_versions.items():
        try:
            installed_version = importlib.metadata.version(package)
        except Exception:
            raise RuntimeError(
                f"{package} is not installed. Please run this command to install: {cmd}"
            ) from None

        if min_version:
            try:
                if Version(installed_version) < Version(min_version):
                    raise RuntimeError(
                        f"{package} version {installed_version} is lower than the required version {min_version}. Please run this command to update dependencies: {cmd}"
                    )
            except InvalidVersion:
                raise RuntimeError(
                    f"Invalid version format for {package}: {installed_version}"
                ) from None

        # print(f"{package}: {installed_version} (OK)")
