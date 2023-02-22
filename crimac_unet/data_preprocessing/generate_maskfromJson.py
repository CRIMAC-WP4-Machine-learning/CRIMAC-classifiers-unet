
### Begin generated header
import sys
sys.path.append('c:/Users/nilsolav/.ApplicationData/lsss/config/packagesConfig/include')
import lsss
import json
### End generated header

erasedRegionsasSandEel = lsss.get('/lsss/regions/deletion')
with open('data2008.json', 'w') as f:
    json.dump(erasedRegionsasSandEel, f, sort_keys=True)

