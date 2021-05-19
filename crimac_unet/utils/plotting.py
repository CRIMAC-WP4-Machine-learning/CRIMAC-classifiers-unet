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

def set_localhost(val= None):
    import platform
    if platform.system() is not 'Windows':
        try:
            import os
            
            if val is None:
                import socket

                socket_name = socket.gethostname()
                f = open(os.getenv("HOME")+'/'+socket_name)
                val = f.readline().strip('\n')
                f.close()
                
                string = 'localhost:' + str(val) + '.0'
                string = string.replace('.0.0','.0')
                string = string.replace('localhost:localhost:', 'localhost:')
                print(string)
                os.environ['DISPLAY'] = string
        except:
            print('Could not set DISPLAY')
                    

def setup_matplotlib(local_host_no = None):
    set_localhost(local_host_no)
    import platform
    import matplotlib.pyplot as plt
    if platform.system() is not 'Windows':
        if plt.get_backend() != u'TkAgg' or plt.get_backend() != 'Qt5Agg':
            print('setup_matplotlib: switching backend from', plt.get_backend(), 'to', 'TkAgg')
            plt.switch_backend('TkAgg')
    return plt
