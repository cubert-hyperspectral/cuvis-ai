import ast
import inspect
import sys


def get_members(module):
    def predicate(x):
        return inspect.ismodule or inspect.function

    for x, y in inspect.getmembers(module, predicate):
        try:
            md = y.__module__
            if md != "__main__":
                yield (x, y.__name__, y.__module__)
        except:
            if inspect.ismodule(y):
                yield (x, y.__name__)


def get_imports(c):
    source_file = inspect.getsourcefile(c)
    with open(source_file) as f:
        lines = f.readlines()
    imports = [l for l in lines if l.strip().startswith("import") or l.strip().startswith("from")]
    return imports


def get_source_file(c):
    source_file = inspect.getsourcefile(c)
    with open(source_file) as f:
        return f.read()


def get_referenced(c):
    referenced = set()
    source_code = inspect.getsource(c)
    tree = ast.parse(source_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            # Check for class instantiations
            referenced.add(node.func.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            # Check for enum instanciations
            referenced.add(node.value.id)
    return referenced


def get_src(c, imports=True, visited=set()):
    if c in visited:
        return []
    visited.add(c)

    src = []
    try:
        source_code = inspect.getsource(c)
    except TypeError as e:
        print(e)
        return []

    if imports:
        import_lines = get_imports(c)
        src += import_lines
        src.append("\n")

    ref_classes = get_referenced(c)
    for ref_class_name in ref_classes:
        try:
            ref_class = sys.modules[c.__module__].__dict__[ref_class_name]
            if inspect.isclass(ref_class) and ref_class.__module__ not in [
                "builtins",
                "collections",
            ]:
                src += get_src(ref_class, imports=False, visited=visited)
                src.append("\n")
        except KeyError as e:
            print(f"Cant find class {e}")
            pass

    src += source_code

    return src
