# Blockchain
Implementation of two simple proof-of-work blockchains for Quant Club, IIT Kharagpur.

For blockchain1.py file, the code is self explainatory. Simply, change the difficulty and observe how many hashes it needs to calculate as the target changes!

For blockchain2.py, download Postman to accept your API request from https://www.postman.com/downloads/ or you directly use the web version instead from https://web.postman.co/
and create a new workspace.

In the GET HTTP request box enter the following URLs after running this python file on your system:

* http://127.0.0.1:5000/mine_block to mine a new block.
* http://127.0.0.1:5000/get_chain to get the entire chain.
* http://127.0.0.1:5000/is_valid to check if the whole chain is valid or not.
