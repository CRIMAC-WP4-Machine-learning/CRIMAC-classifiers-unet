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

### Begin generated header
import sys
sys.path.append('c:/Users/nilsolav/.ApplicationData/lsss/config/packagesConfig/include')
import lsss
import json
### End generated header

erasedRegionsasSandEel = lsss.get('/lsss/regions/deletion')
with open('data2008.json', 'w') as f:
    json.dump(erasedRegionsasSandEel, f, sort_keys=True)

