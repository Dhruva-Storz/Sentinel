<h1>MSc AI Group Project: <br/> A data mining system for Twitter-based event summarisation and sentiment analysis<\h1>

<h2>How to run webapp</h2>

Running Locally:

The virtual environment and ML models are stored in vol/bitbucket, which are accessible to members of Imperial College DOC.

In order to run, you require a linux machine with a GPU. The webapp defaults to port 5000, if you are running 
on a virtual machine, you will need to tunnel to this port to access the webapp.

Otherwise, you simply have to go to the 'sentinel' folder, and run

chmod +x run.sh

followed by 

./run.sh

run.sh connects to the virtual environment and then runs the webapp. As long as your computer can read from 
/vol/bitbucket, this should work without any issues.



