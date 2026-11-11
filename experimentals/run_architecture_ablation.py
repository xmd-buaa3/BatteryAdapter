"""CAF, Stage A, ATD, and shape-expert ablations for the five backbones.

--arm no_caf requires actual per-cycle capacity; an all-masked capacity field
cannot substitute for this input.
"""
from experimentals.common import main as run
from experimentals.protocol import ARCHITECTURE


def main(argv=None):
    return run(argv, arms=('full', *ARCHITECTURE), variants='configs/architecture/variants.json')


if __name__ == '__main__':
    main()
