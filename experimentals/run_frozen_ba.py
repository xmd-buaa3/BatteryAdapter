"""Freeze the five pretrained backbones and train only BA; support train/search/evaluate/summarize/matrix."""
from experimentals.common import main as run


def main(argv=None):
    return run(argv, full_finetune=False)


if __name__ == '__main__':
    main()
