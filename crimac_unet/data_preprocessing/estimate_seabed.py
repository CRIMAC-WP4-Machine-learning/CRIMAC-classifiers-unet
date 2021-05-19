""""
Copyright 2021 the Norwegian Computing Center

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 3 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
"""


import os

import paths
from data.echogram import Echogram


def save_all_seabeds():
    """
    Loop through all echograms and generate seabed-estimates
    :return:
    """
    path_to_echograms = paths.path_to_echograms()
    echogram_names = os.listdir(path_to_echograms)
    echograms = [Echogram(path_to_echograms + e) for e in echogram_names]
    for e in echograms:
        e.get_seabed(save_to_file=True, ignore_saved=True)


if __name__ == '__main__':
    save_all_seabeds()