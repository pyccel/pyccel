# Pyccel Plugin System

This document provides an overview of the Pyccel plugin system and a guide on how to create new plugins.

## Overview

The Pyccel plugin system allows for extending Pyccel's functionality through plugins. Plugins can modify the behaviour of existing classes by patching their methods or adding new methods. This is particularly useful for adding support for new language features, optimisations, or code generation targets without modifying the core codebase.

The plugin system is built around the following key components:

1. `Plugin` - Abstract base class that all plugins must implement
2. `Plugins` - Singleton manager that handles loading, registering, and un-registering plugins
3. `PatchInfo` - Class that stores information about a single method patch
4. `PatchRegistry` - Class that manages all patches applied to a single target class

## Plugin Base Class

All plugins must extend the `Plugin` abstract base class and implement its required methods:

```python
class Plugin(ABC):
    def __init__(self):
        self._patch_registries = []
        
    @abstractmethod
    def register(self, instances):
        """Register instances with this plugin."""
        pass
        
    @abstractmethod
    def refresh(self):
        """Refresh all registered targets."""
        pass
        
    @abstractmethod
    def unregister(self, instances):
        """Un-register instances from this plugin."""
        pass
        
    @abstractmethod
    def set_options(self, options):
        """Set options for this plugin."""
        pass
        
    @property
    def name(self):
        """Return the plugin name, defaults to class name."""
        return self.__class__.__name__
```

### Required Methods

1. `register(instances)` - Register instances with the plugin. This method should apply patches to the provided instances.
2. `refresh()` - Refresh all registered targets. This method should reapply patches to all registered instances.
3. `unregister(instances)` - Un-register instances from the plugin. This method should remove all patches applied to the provided instances.
4. `set_options(options)` - Set options for the plugin. This method should update the plugin's behaviour based on the provided options.

## Patch System

The patch system is the core mechanism that allows plugins to modify the behaviour of existing classes. It consists of two main components:

### `PatchInfo`

The `PatchInfo` class stores information about a single method patch:

```python
@dataclass
class PatchInfo:
    original_method: Optional[Callable]
    patched_method: Callable
    method_name: str
```

- `original_method` - The original method being patched (None for new methods)
- `patched_method` - The new method that will replace the original
- `method_name` - The name of the method being patched

### `PatchRegistry`

The `PatchRegistry` class manages all patches applied to a single target class:

```python
@dataclass
class PatchRegistry:
    target: Any
    patches: Dict[str, List[PatchInfo]] = field(default_factory=dict)
    
    def register_patch(self, method_name, patch_info):
        """Register a patch for a method."""
        if method_name not in self.patches:
            self.patches[method_name] = []
        self.patches[method_name].append(patch_info)
```

- `target` - The target class or object to which patches will be applied
- `patches` - A dictionary mapping method names to lists of `PatchInfo` objects

## Plugin Manager

The `Plugins` class is a singleton that manages all plugins in Pyccel:

```python
class Plugins(metaclass=Singleton):
    def __init__(self, plugins_dir=None):
        self._plugins = []
        self._options = {}
        self.load_plugins(plugins_dir)
        
    def load_plugins(self, plugins_dir=None):
        """Discover and load all plugins from the plugins directory."""
        # ...
        
    def register(self, instances, plugins=()):
        """Register instances with plugins."""
        # ...
        
    def unregister(self, instances, plugins=()):
        """Un-register instances from plugins."""
        # ...
        
    def set_options(self, options, refresh=False):
        """Set options for all plugins."""
        # ...
```

## Creating a New Plugin

Here's a step-by-step guide to creating a new plugin for Pyccel:

### Step 1: Create a Plugin Directory

Create a new directory for your plugin in the `pyccel/plugins` directory. The name of the directory should reflect the purpose of your plugin.

```text
pyccel/plugins/MyPlugin/
```

### Step 2: Create the Plugin Module

Create a `plugin.py` file in your plugin directory. This file will contain your plugin implementation.

```text
pyccel/plugins/MyPlugin/plugin.py
```

### Step 3: Implement the Plugin Class

In your `plugin.py` file, implement a class that extends the `Plugin` base class and implements all required methods:

### Step 4: Implement Patching Logic

The core functionality of your plugin will be in the `register` method, which applies patches to the target class. Here's an example of how to patch a method:

```python
def register(self, registry):
    """Apply patches to the target."""
    target = registry.target
    
    # Example: Patch the 'parse' method
    original_method = getattr(target, 'parse', None)
    
    def parse(self, *args, **kwargs):
        # Do something before the original method
        print("Before parsing")
        
        # Call the original method
        result = original_method(self, *args, **kwargs)
        
        # Do something after the original method
        print("After parsing")
        
        return result
    
    # Apply the patch
    from types import MethodType
    setattr(target, 'parse', MethodType(parse, target))
    
    # Register the patch
    patch_info = PatchInfo(
        original_method=original_method,
        patched_method=parse,
        method_name='parse'
    )
    registry.register_patch('parse', patch_info)
```

### Step 5: Implement Un-patching Logic

The `unregister` method should remove all patches applied to the target class:

```python
def unregister(self, registry):
    """Remove patches from the target."""
    target = registry.target
    
    # Restore original methods
    for method_name, patch_infos in registry.patches.items():
        for patch_info in patch_infos:
            if patch_info.is_new_method:
                # Remove new methods
                if hasattr(target, method_name):
                    delattr(target, method_name)
            else:
                # Restore original methods
                original = patch_info.original_method
                if original:
                    setattr(target, method_name, original)
```

## Example: OpenMP Plugin

The OpenMP plugin which ships with Pyccel extends the plugin system to support OpenMP directives in source code.
