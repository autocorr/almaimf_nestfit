#!/usr/bin/env python3

from pathlib import Path
from dataclasses import dataclass

import numpy as np


ROOT_PATH = Path('/lustre/aoc/users/bsvoboda/temp/alma_imf_nnhp')
DATA_PATH = ROOT_PATH / 'data'
PLOT_PATH = ROOT_PATH / 'plots'
RUN_PATH  = ROOT_PATH / 'run'


@dataclass
class Target:
    name: str
    filen: str
    vsys: str

    @property
    def path(self):
        return DATA_PATH / self.filen


ALL_TARGETS = {
    t.name: t for t in (
        Target('W43-MM1',  'W43-MM1_B3_spw0_12M_n2hp-core6_fixed.fits', 97.5),
        Target('W51-IRS2', 'cutout_for_brian_2021_March_19_fixed.fits',  0.0),
    )
}


