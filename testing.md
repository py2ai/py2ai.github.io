# PyShine Course

<button id="alert-button">Alert & Insert</button><script src="brython.js"> </script><script src="brython_stdlib.js" ></script>
<script type="text/python" id="script0">from browser import document,console,alert def show(e): console.log('Hello',e); alert('Hello world!'); import sys print("Here==>",sys.executable) a=2 import numpy as np print(a,np.__version__) document['alert-button'].bind('click',show)</script>
