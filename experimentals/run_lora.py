"""Native TSFM LoRA training and prediction without BA.

The native.lora section of --config records rank/alpha/dropout; --adapter loads trained LoRA.
--allow-source-protocol explicitly selects the source input and training recipe,
without claiming paper-protocol reproduction.
MOMENT also trains the native forecast head, and the result records that behavior.
"""
import sys
from experimentals.native_forecast import main as run


def main(argv=None):
    return run(sys.argv[1:] if argv is None else argv, mode='lora')


if __name__ == '__main__':
    main()
