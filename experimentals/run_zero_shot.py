"""Native TSFM zero-shot forecasting without BA or LoRA training.

--model selects a backbone; --model-path specifies a verified weights directory;
--manifest specifies an official fold.
--allow-source-protocol explicitly opts into the source implementation's input recipe;
it does not claim that the paper's identical-input protocol has been verified.
MOMENT reconstruction followed by linear extrapolation is not offered as zero-shot forecasting.
"""
import sys
from experimentals.native_forecast import main as run


def main(argv=None):
    return run(sys.argv[1:] if argv is None else argv, mode='zero-shot')


if __name__ == '__main__':
    main()
