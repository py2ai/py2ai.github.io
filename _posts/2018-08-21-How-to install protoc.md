---
layout: post
title: How to install protobuf on Mac OS
author: Hussain A.
categories: [tutorial]
featured-img: first_kaggle_cover
summary: A quick tutorial to install protoc for tensorflow

published: true
disqus: true 
---




Hi everybody, today i show you a way to install Google Protocol Buffers on MAC OS. You might already have tried:

`brew install protobuf` and hit by an error like this:

`Error: The brew link step did not complete successfully 
The formula built, but is not symlinked into /usr/local
Could not symlink include/google
/usr/local/include is not writable. `

Again you may have tried `brew link protobuf` which could probably lead to this: `Error: Could not symlink include/google/usr/local/include is not writable.` And may be more... But if you still can't get the luck. Then follow the steps below to install Google Protocol Buffers. The choice of version totally depends on you, but we will proceed with the version 3.6.1.

1) Go to [protobuf releases] and download the pre built binaries of your choice. For example, the latest 3.6.1 versions are [32 bit] and [64 bit].

2) Extract the binaries in a folder named `protoc-3`. Copy and paste this folder to the Library folder.

3) Open Terminal and execute this command:
`touch ~/.bash_profile`

4) Now execute this: 
`open ~/.bash_profile`



Note: Enter password of your Mac OS account in the popup window.

5) Add these two lines to the file and save it.

`PATH="/Library/protoc-3/bin:${PATH}"`
`PATH="/Library/protoc-3/include:${PATH}"`

6) Close the terminal to refresh the settings and reopen it using `command` + `Space` key.

7) Now check the update PATH by using this command.

`echo $PATH`

8) If nothing happend wrong, you will see this added to your PATH:

`/Library/protoc-3/include:/Library/protoc-3/bin:`

9) Now check the installed version and execute:

`protoc --version`

10) You will see this `libprotoc 3.6.1`, Congratulations! you have now protobuff ready!.



            
            


<!--
{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

-->


[protobuf releases]:https://github.com/google/protobuf/releases
[32 bit]:https://github.com/google/protobuf/releases/download/v3.6.1/protoc-3.6.1-osx-x86_32.zip
[64 bit]:https://github.com/google/protobuf/releases/download/v3.6.1/protoc-3.6.1-osx-x86_64.zip
[tutorial]:https://github.com/py2ai/py2ai.github.io/blob/master/blog/index.html
