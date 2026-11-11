"""Fully fine-tune each of the five pretrained backbones together with BA; validation selection saves both backbone and BA weights."""
from experimentals.common import main as run


def main(argv=None):
    return run(argv, full_finetune=True)


if __name__ == '__main__':
    main()
