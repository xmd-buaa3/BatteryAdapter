"""Curve, aging-condition, CE/EE+SOC, SOH, and anchor ablations for Toto-1/Sundial.

--arm selects a separately retrained variant. The SOH input channel and
observed-start anchor are controlled independently.
"""
from experimentals.common import main as run
from experimentals.protocol import INFORMATION


def main(argv=None):
    return run(argv, models=('toto1', 'sundial'), arms=('full', *INFORMATION),
               variants='configs/information/variants.json')


if __name__ == '__main__':
    main()
