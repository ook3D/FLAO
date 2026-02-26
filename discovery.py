"""
This finds FiveM resources and their Lua scripts.

Handles the standard FiveM resource structure:
  resources/[category]/resource_name/
  resources/resource_name/

Each resource is identified by fxmanifest.lua or __resource.lua
"""

from pathlib import Path
from typing import Dict, List


def discover_resources(root_path: Path) -> Dict[str, List[Path]]:
    """
    Discover all FiveM resources and their script files.

    Returns dict mapping resource name -> list of script file paths
    """
    resources = {}
    root_path = Path(root_path)

    # Check if this IS a resource directly (has manifest)
    if _is_fivem_resource(root_path):
        scripts = find_fivem_scripts(root_path)
        if scripts:
            resources[root_path.name] = scripts
        return resources

    # Look for resources with fxmanifest.lua
    for manifest in root_path.rglob('fxmanifest.lua'):
        resource_dir = manifest.parent
        resource_name = _get_resource_name(resource_dir, root_path)
        if resource_name not in resources:
            scripts = find_fivem_scripts(resource_dir)
            if scripts:
                resources[resource_name] = scripts

    # Also check for __resource.lua (older format)
    for manifest in root_path.rglob('__resource.lua'):
        resource_dir = manifest.parent
        resource_name = _get_resource_name(resource_dir, root_path)
        if resource_name not in resources:
            scripts = find_fivem_scripts(resource_dir)
            if scripts:
                resources[resource_name] = scripts

    return resources


def _is_fivem_resource(path: Path) -> bool:
    """Check if a directory is a FiveM resource."""
    return (path / 'fxmanifest.lua').exists() or (path / '__resource.lua').exists()


def _get_resource_name(resource_dir: Path, root_path: Path) -> str:
    """Get a unique resource name including parent category if exists."""
    try:
        rel = resource_dir.relative_to(root_path)
        # If in a category folder (e.g., [qb]/qb-core), include it
        parts = rel.parts
        if len(parts) > 1:
            return '/'.join(parts)
        return resource_dir.name
    except ValueError:
        return resource_dir.name


def find_fivem_scripts(resource_dir: Path) -> List[Path]:
    """Find all Lua scripts in a FiveM resource."""
    scripts = []
    seen = set()

    # Common FiveM script directory patterns
    script_patterns = [
        '*.lua',           # Root lua files
        'client/*.lua',    # Client scripts
        'server/*.lua',    # Server scripts
        'shared/*.lua',    # Shared scripts
        'config/*.lua',    # Config files
        'locales/*.lua',   # Locale files
        'lib/*.lua',       # Library files
        'modules/*.lua',   # Module files
    ]

    # First check common patterns
    for pattern in script_patterns:
        for lua_file in resource_dir.glob(pattern):
            if lua_file not in seen:
                scripts.append(lua_file)
                seen.add(lua_file)

    # Then do recursive search for any remaining .lua files
    for lua_file in resource_dir.rglob('*.lua'):
        if lua_file not in seen:
            # Skip node_modules and hidden directories
            if 'node_modules' in lua_file.parts:
                continue
            if any(part.startswith('.') and part not in ('.', '..') for part in lua_file.parts):
                continue
            scripts.append(lua_file)
            seen.add(lua_file)

    return sorted(scripts)


def find_scripts(scripts_dir: Path) -> List[Path]:
    """Find all Lua script files in a directory (legacy compatibility)."""
    scripts = []

    # .lua files
    for ext in ("*.lua",):
        scripts.extend(scripts_dir.glob(ext))

    # also check subdirectories
    for ext in ("**/*.lua",):
        for f in scripts_dir.glob(ext):
            if f not in scripts:
                scripts.append(f)

    return sorted(scripts)


def get_resource_info(resource_path: Path) -> dict:
    """
    Try to extract resource info from manifest file.
    Returns dict with name, version, author if found.
    """
    info = {"name": resource_path.name}

    # Try fxmanifest.lua first
    manifest = resource_path / "fxmanifest.lua"
    if not manifest.exists():
        manifest = resource_path / "__resource.lua"

    if manifest.exists():
        try:
            content = manifest.read_text(encoding='utf-8', errors='ignore')
            for line in content.splitlines():
                line = line.strip()
                # Parse common manifest fields
                if line.startswith("name"):
                    # name 'Resource Name' or name "Resource Name"
                    val = _extract_string_value(line)
                    if val:
                        info["name"] = val
                elif line.startswith("version"):
                    val = _extract_string_value(line)
                    if val:
                        info["version"] = val
                elif line.startswith("author"):
                    val = _extract_string_value(line)
                    if val:
                        info["author"] = val
                elif line.startswith("description"):
                    val = _extract_string_value(line)
                    if val:
                        info["description"] = val
        except (OSError, IOError):
            pass

    return info


def _extract_string_value(line: str) -> str:
    """Extract a string value from a manifest line like: name 'value' or name \"value\""""
    # Remove the key
    for quote in ["'", '"']:
        if quote in line:
            start = line.index(quote) + 1
            end = line.rindex(quote)
            if end > start:
                return line[start:end]
    return ""


def discover_direct(path: Path) -> Dict[str, List[Path]]:
    """
    Discover scripts directly without resource structure.

    - If path is a .lua file, return just that file
    - If path is a directory, find all scripts in it (recursively)

    Returns dict mapping resource name -> list of script file paths
    """
    path = Path(path)
    resources = {}

    # Single file
    if path.is_file():
        if path.suffix == '.lua':
            resources["(direct)"] = [path]
        return resources

    # Directory - find all Lua scripts
    if path.is_dir():
        scripts = []
        seen = set()

        # Root level
        for lua_file in path.glob("*.lua"):
            scripts.append(lua_file)
            seen.add(lua_file)

        # Recursive
        for lua_file in path.rglob("*.lua"):
            if lua_file not in seen:
                # Skip common non-script directories
                if 'node_modules' in lua_file.parts:
                    continue
                if any(part.startswith('.') and part not in ('.', '..') for part in lua_file.parts):
                    continue
                scripts.append(lua_file)

        if scripts:
            resources["(direct)"] = sorted(scripts)

    return resources


# Legacy alias for backwards compatibility
discover_mods = discover_resources
