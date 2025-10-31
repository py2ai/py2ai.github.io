    
      from browser import document, alert

      def echo(event):
          alert(document["zone"].value)

      document["mybutton"].bind("click", echo)
