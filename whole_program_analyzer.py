"""
Whole-program analyzer for cross-file dead code detection in FiveM resources.

Originally based on ALAO (Anomaly Lua Auto Optimizer) by Abraham (Priler).
Refactored for FiveM/GTA 5 Lua optimization.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional, Any
from collections import defaultdict
import sys
import io

from luaparser import ast
from luaparser.astnodes import (
    Node, Chunk, Block,
    Function, LocalFunction, Method,
    Assign, LocalAssign,
    Call, Invoke,
    Index, Name, String,
    Return, Break,
)


@dataclass
class SymbolDefinition:
    """Tracks where a symbol is defined."""
    name: str
    file_path: Path
    line: int
    symbol_type: str  # 'global_function', 'module_function', 'local_function', 'global_var', 'local_var'
    scope: str  # 'global', 'module', 'local'
    is_callback: bool = False
    is_class_method: bool = False


@dataclass
class SymbolUsage:
    """Tracks where a symbol is used."""
    name: str
    file_path: Path
    line: int
    usage_type: str  # 'call', 'read', 'callback_register'


@dataclass
class CrossFileAnalysis:
    """Results of whole-program analysis."""
    definitions: Dict[str, List[SymbolDefinition]] = field(default_factory=lambda: defaultdict(list))
    usages: Dict[str, List[SymbolUsage]] = field(default_factory=lambda: defaultdict(list))
    registered_callbacks: Set[str] = field(default_factory=set)
    exported_symbols: Set[str] = field(default_factory=set)  # symbols that might be used externally
    
    def is_symbol_used(self, name: str) -> bool:
        """Check if a symbol is used anywhere."""
        return name in self.usages or name in self.registered_callbacks or name in self.exported_symbols
    
    def get_unused_globals(self) -> List[SymbolDefinition]:
        """Get global symbols that appear unused."""
        unused = []
        for name, defs in self.definitions.items():
            for d in defs:
                if d.scope == 'global' and not self.is_symbol_used(name):
                    unused.append(d)
        return unused


# Known callback/event names that FiveM can call
KNOWN_CALLBACKS = frozenset({
    # FiveM Client Events
    'onClientResourceStart', 'onClientResourceStop',
    'onClientMapStart', 'onClientMapStop',
    'onClientGameTypeStart', 'onClientGameTypeStop',
    'gameEventTriggered', 'populationPedCreating',

    # FiveM Server Events
    'onResourceStart', 'onResourceStop', 'onResourceStarting',
    'playerConnecting', 'playerDropped',

    # BaseEvents (common FiveM resource)
    'baseevents:onPlayerDied', 'baseevents:onPlayerKilled',
    'baseevents:onPlayerWasted', 'baseevents:enteringVehicle',
    'baseevents:enteredVehicle', 'baseevents:enteringAborted',
    'baseevents:leftVehicle',

    # Common thread/loop naming patterns
    'onTick', 'OnTick', 'tick', 'Tick',
    'mainLoop', 'MainLoop', 'gameLoop',

    # NUI callbacks (often registered dynamically)
    'RegisterNUICallback',

    # Common export patterns
    '__export', 'exports',
})

# Patterns that indicate a function is exported/public in FiveM
EXPORT_PATTERNS = {
    '_G',           # _G.func = ...
    'rawset',       # rawset(_G, "name", func)
    'exports',      # exports('name', func) - FiveM exports
    'RegisterNetEvent',      # Event handlers
    'RegisterServerEvent',   # Server event handlers
    'AddEventHandler',       # Event handlers
}


class WholeProgramAnalyzer:
    """Performs whole-program analysis across multiple script files."""
    
    def __init__(self):
        self.analysis = CrossFileAnalysis()
        self.files_analyzed: Set[Path] = set()
        self.parse_errors: List[Tuple[Path, str]] = []
    
    def analyze_directory(self, directory: Path, recursive: bool = True) -> CrossFileAnalysis:
        """Analyze all .script files in a directory."""
        pattern = '**/*.script' if recursive else '*.script'
        script_files = list(directory.glob(pattern))
        
        # Pass 1: Collect all definitions
        for script_path in script_files:
            self._collect_definitions(script_path)
        
        # Pass 2: Collect all usages
        for script_path in script_files:
            self._collect_usages(script_path)
        
        return self.analysis
    
    def analyze_files(self, files: List[Path]) -> CrossFileAnalysis:
        """Analyze a specific list of files."""
        # Pass 1: Collect all definitions
        for script_path in files:
            self._collect_definitions(script_path)
        
        # Pass 2: Collect all usages
        for script_path in files:
            self._collect_usages(script_path)
        
        return self.analysis
    
    def _parse_file(self, file_path: Path) -> Optional[Chunk]:
        """Parse a Lua file, returning None on error."""
        try:
            source = file_path.read_text(encoding='utf-8', errors='ignore')
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                tree = ast.parse(source)
            finally:
                sys.stderr = old_stderr
            return tree
        except Exception as e:
            self.parse_errors.append((file_path, str(e)))
            return None
    
    def _collect_definitions(self, file_path: Path):
        """Pass 1: Collect all symbol definitions from a file."""
        tree = self._parse_file(file_path)
        if not tree:
            return
        
        self.files_analyzed.add(file_path)
        self._visit_for_definitions(tree, file_path)
    
    def _collect_usages(self, file_path: Path):
        """Pass 2: Collect all symbol usages from a file."""
        tree = self._parse_file(file_path)
        if not tree:
            return
        
        source = file_path.read_text(encoding='utf-8', errors='ignore')
        self._visit_for_usages(tree, file_path, source)
    
    def _get_line(self, node: Node) -> int:
        """Extract line number from node."""
        ft = getattr(node, 'first_token', None)
        if ft:
            s = str(ft)
            if ',' in s:
                parts = s.rsplit(',', 1)
                if len(parts) == 2:
                    line_col = parts[1].rstrip(']')
                    if ':' in line_col:
                        try:
                            return int(line_col.split(':')[0])
                        except ValueError:
                            pass
        return 0
    
    def _node_to_string(self, node: Node) -> str:
        """Convert AST node to string representation."""
        if isinstance(node, Name):
            return node.id
        elif isinstance(node, Index):
            value = self._node_to_string(node.value)
            idx = self._node_to_string(node.idx)
            idx_token = getattr(node.idx, 'first_token', None)
            if idx_token is not None and str(idx_token) != 'None':
                return f"{value}[{idx}]"
            else:
                return f"{value}.{idx}"
        elif isinstance(node, String):
            s = node.s
            if isinstance(s, bytes):
                s = s.decode('utf-8', errors='replace')
            return s
        return ""
    
    def _visit_for_definitions(self, node: Node, file_path: Path, in_local_scope: bool = False):
        """Visit AST to collect definitions."""
        if node is None:
            return
        
        # Global function: function name() ... end
        if isinstance(node, Function):
            if isinstance(node.name, Name):
                name = node.name.id
                line = self._get_line(node)
                is_callback = name in KNOWN_CALLBACKS
                
                self.analysis.definitions[name].append(SymbolDefinition(
                    name=name,
                    file_path=file_path,
                    line=line,
                    symbol_type='global_function',
                    scope='global',
                    is_callback=is_callback,
                ))
                
                if is_callback:
                    self.analysis.registered_callbacks.add(name)
            
            # Module function: function module.name() ... end
            elif isinstance(node.name, Index):
                full_name = self._node_to_string(node.name)
                line = self._get_line(node)
                
                self.analysis.definitions[full_name].append(SymbolDefinition(
                    name=full_name,
                    file_path=file_path,
                    line=line,
                    symbol_type='module_function',
                    scope='module',
                ))
                # module functions are potentially exported
                self.analysis.exported_symbols.add(full_name)
        
        # Local function: local function name() ... end
        elif isinstance(node, LocalFunction):
            if isinstance(node.name, Name):
                name = node.name.id
                line = self._get_line(node)
                is_callback = name in KNOWN_CALLBACKS
                
                self.analysis.definitions[f"local:{file_path.stem}:{name}"].append(SymbolDefinition(
                    name=name,
                    file_path=file_path,
                    line=line,
                    symbol_type='local_function',
                    scope='local',
                    is_callback=is_callback,
                ))
        
        # Method definition: function class:method() ... end
        elif isinstance(node, Method):
            source = self._node_to_string(node.source)
            method = node.name.id if isinstance(node.name, Name) else ""
            full_name = f"{source}:{method}"
            line = self._get_line(node)
            is_class_method = method in KNOWN_CALLBACKS
            
            self.analysis.definitions[full_name].append(SymbolDefinition(
                name=full_name,
                file_path=file_path,
                line=line,
                symbol_type='method',
                scope='module',
                is_class_method=is_class_method,
            ))
            
            if is_class_method:
                self.analysis.exported_symbols.add(full_name)
        
        # Global assignment: name = value
        elif isinstance(node, Assign):
            for target in node.targets:
                if isinstance(target, Name):
                    name = target.id
                    line = self._get_line(node)
                    
                    # Check if assigning a function
                    if node.values and len(node.values) == 1:
                        val = node.values[0]
                        if isinstance(val, Function):
                            is_callback = name in KNOWN_CALLBACKS
                            self.analysis.definitions[name].append(SymbolDefinition(
                                name=name,
                                file_path=file_path,
                                line=line,
                                symbol_type='global_function',
                                scope='global',
                                is_callback=is_callback,
                            ))
                            if is_callback:
                                self.analysis.registered_callbacks.add(name)
                        else:
                            self.analysis.definitions[name].append(SymbolDefinition(
                                name=name,
                                file_path=file_path,
                                line=line,
                                symbol_type='global_var',
                                scope='global',
                            ))
                
                # Module assignment: module.name = value
                elif isinstance(target, Index):
                    full_name = self._node_to_string(target)
                    line = self._get_line(node)
                    
                    self.analysis.definitions[full_name].append(SymbolDefinition(
                        name=full_name,
                        file_path=file_path,
                        line=line,
                        symbol_type='module_var',
                        scope='module',
                    ))
                    self.analysis.exported_symbols.add(full_name)
        
        # Recurse into children
        for child in ast.walk(node):
            if child is not node:
                self._visit_for_definitions(child, file_path, in_local_scope)
    
    def _visit_for_usages(self, node: Node, file_path: Path, source: str):
        """Visit AST to collect usages."""
        if node is None:
            return
        
        # Function call: name() or module.name()
        if isinstance(node, Call):
            func_name = self._node_to_string(node.func)
            line = self._get_line(node)
            
            if func_name:
                self.analysis.usages[func_name].append(SymbolUsage(
                    name=func_name,
                    file_path=file_path,
                    line=line,
                    usage_type='call',
                ))
            
            # Check for AddEventHandler("eventName", func) - FiveM event registration
            if func_name == 'AddEventHandler' and len(node.args) >= 2:
                event_name = self._node_to_string(node.args[0])
                callback_func = self._node_to_string(node.args[1])

                if event_name:
                    self.analysis.registered_callbacks.add(event_name)
                if callback_func:
                    self.analysis.registered_callbacks.add(callback_func)
                    self.analysis.usages[callback_func].append(SymbolUsage(
                        name=callback_func,
                        file_path=file_path,
                        line=line,
                        usage_type='callback_register',
                    ))

            # Check for exports("name", func) - FiveM exports
            if func_name == 'exports' and len(node.args) >= 2:
                export_name = self._node_to_string(node.args[0])
                export_func = self._node_to_string(node.args[1])

                if export_name:
                    self.analysis.exported_symbols.add(export_name)
                if export_func:
                    self.analysis.exported_symbols.add(export_func)
        
        # Method call: obj:method()
        elif isinstance(node, Invoke):
            source_name = self._node_to_string(node.source)
            method_name = node.func.id if isinstance(node.func, Name) else ""
            full_name = f"{source_name}:{method_name}"
            line = self._get_line(node)
            
            # Track usage of the object
            if source_name:
                self.analysis.usages[source_name].append(SymbolUsage(
                    name=source_name,
                    file_path=file_path,
                    line=line,
                    usage_type='read',
                ))
        
        # Variable read: name
        elif isinstance(node, Name):
            name = node.id
            line = self._get_line(node)
            
            self.analysis.usages[name].append(SymbolUsage(
                name=name,
                file_path=file_path,
                line=line,
                usage_type='read',
            ))
        
        # Index read: module.name or module["name"]
        elif isinstance(node, Index):
            full_name = self._node_to_string(node)
            line = self._get_line(node)
            
            if full_name:
                self.analysis.usages[full_name].append(SymbolUsage(
                    name=full_name,
                    file_path=file_path,
                    line=line,
                    usage_type='read',
                ))


def analyze_resources_directory(resources_path: Path) -> CrossFileAnalysis:
    """Convenience function to analyze entire FiveM resources directory."""
    analyzer = WholeProgramAnalyzer()

    # Find all Lua scripts in resources with fxmanifest.lua or __resource.lua
    all_scripts = []

    for manifest in resources_path.rglob('fxmanifest.lua'):
        resource_dir = manifest.parent
        all_scripts.extend(resource_dir.rglob('*.lua'))

    for manifest in resources_path.rglob('__resource.lua'):
        resource_dir = manifest.parent
        # Avoid duplicates
        for lua_file in resource_dir.rglob('*.lua'):
            if lua_file not in all_scripts:
                all_scripts.append(lua_file)

    return analyzer.analyze_files(all_scripts)


# Legacy alias for backwards compatibility
analyze_mods_directory = analyze_resources_directory
