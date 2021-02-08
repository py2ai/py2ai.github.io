<div id="html" markdown="0">
    

<head>
<title>ACE in Action</title>

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
<body>

<div id="editor">function foo(items) {
    var x = "All this is syntax highlighted";
    return x;
}</div>
    
<script src="https://pyshine.com/ace.js" type="text/javascript" charset="utf-8"></script>
<script>
    var editor = ace.edit("editor");
    editor.setTheme("ace/theme/monokai");

    editor.session.setMode("ace/mode/javascript");
</script>
</body>


</div> 






