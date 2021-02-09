<div id="html" markdown="0">


<head>
<meta charset="utf-8">
<title>PyShine Tutotial</title>
<link rel="stylesheet" src="..assets/brython.css">
<link rel="stylesheet" href="console.css">

<script type="text/javascript" src="../src/brython_builtins.js"></script>
<script type="text/javascript" src="../src/version_info.js"></script>
<script type="text/javascript" src="../src/py2js.js"></script>
<script type="text/javascript" src="../src/loaders.js"></script>
<script type="text/javascript" src="../src/py_object.js"></script>
<script type="text/javascript" src="../src/py_type.js"></script>
<script type="text/javascript" src="../src/py_utils.js"></script>
<script type="text/javascript" src="../src/py_sort.js"></script>
<script type="text/javascript" src="../src/py_builtin_functions.js"></script>
<script type="text/javascript" src="../src/py_exceptions.js"></script>
<script type="text/javascript" src="../src/py_range_slice.js"></script>
<script type="text/javascript" src="../src/py_bytes.js"></script>
<script type="text/javascript" src="../src/py_set.js"></script>
<script type="text/javascript" src="../src/js_objects.js"></script>
<script type="text/javascript" src="../src/stdlib_paths.js"></script>
<script type="text/javascript" src="../src/py_import.js"></script>

<script type="text/javascript" src="../src/unicode_data.js"></script>
<script type="text/javascript" src="../src/py_string.js"></script>
<script type="text/javascript" src="../src/py_int.js"></script>
<script type="text/javascript" src="../src/py_long_int.js"></script>
<script type="text/javascript" src="../src/py_float.js"></script>
<script type="text/javascript" src="../src/py_complex.js"></script>
<script type="text/javascript" src="../src/py_dict.js"></script>
<script type="text/javascript" src="../src/py_list.js"></script>
<script type="text/javascript" src="../src/py_generator.js"></script>
<script type="text/javascript" src="../src/py_dom.js"></script>

<script type="text/javascript" src="../src/builtin_modules.js"></script>
<script type="text/javascript" src="../src/async.js"></script>

<script type="text/javascript" src="../src/brython_stdlib.js"></script>

<script src="../assets/header.brython.js"></script>

<script src="https://pagecdn.io/lib/ace/1.4.12/ace.js" type="text/javascript" charset="utf-8"></script>
<script src="ace/ext-language_tools.js" type="text/javascript" charset="utf-8"></script>
</head>



<body onload="brython({debug:2})">
<script type="text/python3" id="tests_editor">
from browser import document as doc, window
from browser import html
import header
import editor
# Create a lambda around editor.run() so that the event object is not passed to it
doc['run'].bind('click',lambda *args: editor.run())
</script>
<div id="main_container"></div>


<table id="container">
  <tr>
    <td>Python Code <span id="version"></span></td>
    <td></td>
    <td>
        <button id="run"> ▶ Run</button>
    </td>
  </tr>

  <tr>
    <td id="left">
      <div id="editor" style="width:100%;"></div>
    </td>
    <td id="separator"></td>
    <td id="right">
      <textarea id="console" autocomplete="off"></textarea>
    </td>
  </tr>

  <tr>
    <td>
      Python learning series by PyShine
    </td>
    <td></td>
    <td>
      <a >Output</a>
    </td>
  </tr>
</table>

</body>




</div> 