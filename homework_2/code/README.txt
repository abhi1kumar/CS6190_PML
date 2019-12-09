First use X-11 port forwarding if you are using ssh to login to the system. Without this python cannot import pyplot and therfore graphs cannot be plotted. In other words, you should use
ssh -X username@lab1-1.eng.utah.edu
After this login, open a terminal and migrate to the required directory

To execute all assignments in one go
chmod +x main.sh
./main.sh

Execution time of the entire code ~ 3 minutes


To run the experiments for Question 1 of the coding assignment
python3 q1.py

To run the experiments for Question 2 of the coding assignment
python3 q2.py

To run the experiments for Question 3 of the coding assignment
python3 q3.py