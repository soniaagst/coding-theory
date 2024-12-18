from cx_Freeze import setup, Executable

base = None    

executables = [Executable("project.py", base=base)]

packages = ["idna"]
options = {
    'build_exe': {    
        'packages':packages,
    },
}

setup(
    name = "<syndrome decoding>",
    options = options,
    version = "1.2",
    description = 'decoding code in GF(q) field',
    executables = executables
)
