## PAMPRO tests

PAMPRO uses [nose](https://nose.readthedocs.org) to test its internal functionality. The tests are executed after every change to the code, and no commits are made to the repository while the code isn't passing the tests. The tests are quite verbosely commented, and hopefully self-explanatory.

### Running tests
Most of the tests work by reading in manually created test files. These are supplied with the PAMPRO download, but to save space and time, they are packaged in a zipped archive. In the /pampro/tests/ folder, simply unzip _data.zip as a folder called /_data/.

Then, navigate to the /pampro/tests/ folder, and at the commandline enter:
```
nosetests *.py
```

If all is well, a message will appear to the effect:
```
Ran 22 tests in 0.325s

OK
```
If not, the software package has been compromised in some way. If you have encountered this error immediately upon downloading the code directly from the central repository, please contact me. If you have changed the code accidentally, you can restore any file with the command:
```
git checkout <file>
```
Finally, if you are changing the code deliberately, you are on your own.
