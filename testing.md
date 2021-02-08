<div id="html" markdown="0">
    

<head>
<title>ACE in Action</title>
<meta charset="utf-8">
<script type="text/javascript"
    src="https://cdn.jsdelivr.net/npm/brython@3.9.0/brython.min.js">
</script>

<style type="text/css" media="screen">
    #editor { 
        position: absolute;
        top: 400px;
        left: 200px;
        width: 640px;
        height:480px;
    }
</style>
</head>






<body onload="brython()">

<div id="editor">function foo(items) {
    var x = "All this is syntax highlighted";
    return x;
}</div>
    
<script src="https://cdnjs.cloudflare.com/ajax/libs/ace/1.4.5/ace.js" type="text/javascript" charset="utf-8"></script>
<script>
    var editor = ace.edit("editor");
    editor.setTheme("ace/theme/monokai");

    editor.session.setMode("ace/mode/javascript");
</script>
<br><br><br><br><br><br><br><br><br><br><br><br><br><br>
<script type="text/python">
from browser import document

document <= "Hello !"</script>
</body>


</div> 






