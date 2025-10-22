import pkg_resources


def get_installed_packages() -> list[tuple[str, str]]:
    packages = [(dist.project_name, dist.version) for dist in pkg_resources.working_set]
    return packages


def get_installed_packages_str():
    packages_with_version = get_installed_packages()
    return [f"{package}=={version}" for package, version in packages_with_version]
