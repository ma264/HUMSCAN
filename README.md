# HUMSCAN
=== Strawberry Perl (64-bit) 5.32.1.1-64bit README ===

What is Strawberry Perl?
------------------------

* 'Perl' is a programming language suitable for writing simple scripts as well
  as complex applications. See http://perldoc.perl.org/perlintro.html

* 'Strawberry Perl' is a perl environment for Microsoft Windows containing all
  you need to run and develop perl applications. It is designed to be as close
  as possible to perl environment on UNIX systems. See http://strawberryperl.com/

* If you are completely new to perl consider visiting http://learn.perl.org/

Installation instructions: (.ZIP distribution only, not .MSI installer)
-----------------------------------------------------------------------

* If installing this version from a .zip file, you MUST extract it to a 
  directory that does not have spaces in it - e.g. c:\myperl\
  and then run some commands and manually set some environment variables:

  c:\myperl\relocation.pl.bat         ... this is REQUIRED!
  c:\myperl\update_env.pl.bat         ... this is OPTIONAL

  You can specify " --nosystem" after update_env.pl.bat to install Strawberry 
  Perl's environment variables for the current user only.

* If having a fixed installation path does not suit you, try "Strawberry Perl
  Portable Edition" from http://strawberryperl.com/releases.html

How to use Strawberry Perl?
---------------------------

* In the command prompt window you can:

  1. run any perl script by launching
  
     c:\> perl c:\path\to\script.pl

  2. install additional perl modules (libraries) from http://www.cpan.org/ by

     c:\> cpan Module::Name
  
  3. run other tools included in Strawberry Perl like: perldoc, gcc, gmake ...

* You'll need a text editor to create perl scripts.  One is NOT included with 
  Strawberry Perl. A few options are Padre (which can be installed by running 
  "cpan Padre" from the command prompt) and Notepad++ (which is downloadable at
  http://notepad-plus-plus.org/ ) which both include syntax highlighting
  for perl scripts. You can even use Notepad, if you wish.
