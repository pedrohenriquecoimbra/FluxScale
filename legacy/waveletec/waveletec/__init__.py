"""waveletec is now fluxscale: this package only keeps old imports working."""
import sys
import warnings

import fluxscale

warnings.warn("waveletec is now fluxscale: use `import fluxscale`.", FutureWarning, stacklevel=2)

# One module under two names, so that nothing gets loaded twice. main is left
# out because `python -m waveletec.main` has to find it under its own name.
for name, module in list(sys.modules.items()):
    if (name == "fluxscale" or name.startswith("fluxscale.")) and name != "fluxscale.main":
        sys.modules["waveletec" + name[len("fluxscale"):]] = module
