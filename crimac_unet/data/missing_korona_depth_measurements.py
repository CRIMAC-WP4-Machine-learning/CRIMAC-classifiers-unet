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


# Discrepancies in "Korona label" depths measurement files (automatic LSSS labeling)

# List of missing depths measurement files
depths_missing = [
    '2009107-D20090509-T133323',
    '2009107-D20090511-T064417',
    '2009107-D20090512-T193310',
    '2009107-D20090518-T185324',
    '2010205-D20100421-T022743',
    '2010205-D20100421-T183034',
    '2010205-D20100424-T053033',
    '2010205-D20100424-T105017',
    '2010205-D20100503-T071455',
    '2010205-D20100505-T132336',
    '2010205-D20100506-T071245',
    '2010205-D20100507-T075742',
    '2010205-D20100507-T155257',
    '2010205-D20100509-T193347',
    '2017843-D20170502-T064950',
    '2017843-D20170502-T073220'
]

# List of depth measurement files with shape that deviates from corresponding echogram shape
depths_shape_discrepancies = [
    '2008205-D20080513-T215110',
    '2008205-D20080514-T224500',
    '2008205-D20080515-T060032',
    '2010205-D20100422-T073220',
    '2010205-D20100423-T121059',
    '2010205-D20100426-T153105',
    '2010205-D20100429-T080945',
    '2010205-D20100429-T170312',
    '2010205-D20100508-T191340',
    '2018823-D20180503-T075932',
    '2018823-D20180506-T130501',
    '2018823-D20180513-T035302'
]

# Combined list of the two above - these echograms will be excluded in model evaluation
depth_excluded_echograms = depths_missing + depths_shape_discrepancies
