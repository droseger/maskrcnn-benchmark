# Author: Petr Vytovtov <p.vytovtov@partner.samsung.com>
from enum import Enum


class DatasetMode(Enum):
    TRAIN = 1
    # Mode to calculate losses on validation set while training:
    VALID = 2
    # Mode to calculate AP on validation set after training:
    VAL_AP = 3
    TEST = 4
