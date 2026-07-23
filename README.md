# **CoChem-SCAN: Massive Parallel Torsional Screening**

## **Overview**

**CoChem-SCAN** is the exploratory module designed for high-throughput, brute-force mapping of torsional barriers and complex multidimensional Potential Energy Surfaces (PES).

When analyzing floppy molecules, a single 1D scan is insufficient. CoChem-SCAN orchestrates 2D and 3D relaxed surface scans using ORCA's OpenMPI backbone. Because running thousands of parallel DFT calculations can easily crash a Linux node, SCAN acts as a rigorous governor and system watchdog.

## **Scientific & Technical Trade-offs**

* **RAM Disk (/dev/shm) I/O Routing:** Quantum chemistry writes massive amounts of temporary integral files (.tmp). To prevent destroying your SSD (TBW limits) and to achieve maximum speed, SCAN routes all temporary ORCA files to the Linux RAM disk (/dev/shm). **Trade-off:** If your molecule is too large, the integrals will exceed your RAM, crashing the OS. SCAN mathematically calculates the integral size prior to launch and refuses to run if it violates the 10% safety margin.  
* **Zombie Process Assassin:** If a user hits Ctrl+C or Jupyter crashes during a 32-core OpenMPI scan, ORCA's C++ child processes often detach and spin endlessly (Zombie processes), locking up the CPU. SCAN wraps all calls in a psutil tree-traversal block. If the parent dies, it hunts down and sends a SIGKILL to every orphaned Fortran worker.

## **Installation**

git clone \[https://github.com/CoChem/CoChem-SCAN.git\](https://github.com/CoChem/CoChem-SCAN.git)  
cd CoChem-SCAN

## **How to Run**

1. **Define the Scan Grid:**  
   Edit scan\_parameters.json to define your dihedrals (e.g., ![][image1] at ![][image2] increments).  
2. **Execute the Sieve & Scan:**  
   python cochem\_scan\_orchestrator.py \--config scan\_parameters.json

## **Advanced Features**

* **Bayesian Spectral Priors:** Instead of binary Match/Mismatch filtering, SCAN weights potential candidate conformers based on the probability that their predicted intensity matches the experimental peak shape, cutting through baseline noise automatically.  
* **Acoustic Mode Purging:** Automatically filters out the 6 lowest rotational/translational modes before mapping, ensuring torsional frequencies aren't artificially mixed.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHUAAAAaCAYAAACJphMzAAAD0UlEQVR4Xu2YPWhUQRDHE1QI+IUf55Hcx15eggcKfnCoCMFWJWgRrGK6ICpYRdBWCfYWEcXOQhLENphC9NAmnWlECFrExiKEYJFCJXf+55xJJpNcvI+3R5T9wXI7/5l9O2/3vb3d19YWCAQCgUAgdhKJxC7nXHmzksvlXth2rQL936YcstnsBMxt1t8K0PclHouZKIoOW79QKBR2IHaKYru7u89ZPwHfIMpPiqP4DfzzKMMob1H6rb8ucIEFSiadTqeMq51vaNno3qE+UZ5TPZPJnOLBcjbOJ+hzCRNwjep4uM9TDrBnbRy0O5xfnmPfodzQMfAvks7+HMefVv4xPdGwRxHWIXbd8MSVrU5AH+CbeWh9vqDVAX2WtAZ7vFqOPkBfV3lcniqt8vDrONbLtOppG2VO2aO2He7xntZQn9F+2J0Y84LWagYNI07io/UJ7F93M77g/opak7dVaz5BXw/sfaM+Z3Oghx3atNYoBpN2U9vUVsekUqk06bivy2TTkqz9aH+hq6vroNZqRpLHRa5bn2BvzifykCGfZ1qXQdCab5DD/d7e3j1io/+SzYHHZkBrFo4pai2fz+9m/TXZtPSi/hnVdozBEdRf6fi6UIlutz4imUzu5M6bGlDHyyf+R5LWp0HMCMXZSYXWyXl0ar1VoN83nNcJ0Wh55JzOonyHfQW/H/D7xLSlmKLW1CZ1SeuxwBf+ZXUBCfZxTLOTOkbX6OnpOWR9GvR3l+KqTWrD/zMNIvmglLAcZoxPdsY/tM55DmnbVZ/UpsZ1Q/jCL60uwDfJMZPWh8SPWq1ZttqkCrw80jjMiyaTav+6OE7/F5NdVCH+JhXJdPBA9VmfIB3LsknHHtiPWI996aABomtXm1T61boG9zEr+f6t2CWyFhzvP2SnK5OKsTlu4qSPfcou6hhvy68kaXUBm5MD3PFJ6+M3Kt6E2lY3RI7PqAIN3Ga5xg3ub4L602dH9WZWHjjZ7ECPVluuTqrjB5Dr73UMvSSsVzZKsYELfnPVjzLy4eGWdRC+JpXgftfkBfsi6VrzCeew5o12fGZHGVFaiY5bYrNWaWvsBR0jD6kcaWLDJiigw2PsW/n/sNQ7qa7G3S+R+/PxwZ4Hp6zmE77/chRFe5U2zTmsnBR4HMbFJigG+mNlr/v44HjjqLWGwYX6JeHNCgb2jG2raWBSa9r9CohdzvFnNXqaqW0rPxPS0YXHYphs+mX7k41lfZDqGJchlK8bxCyifKG6+uy58plwS1DvpAb+AcKk/kfgSLM/y19OUEpUtzGBQCAQCAQCgUAg4IffbMCQT4OwKLcAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB0AAAAZCAYAAADNAiUZAAABHElEQVR4Xu2UPQrCQBCFE6wEwSpVQjbJJTyDhWDvHWwsBA9hJWIhiIWN6A0srLWwEisLWwsFOwV/3kiUYdgNKtiEfDAw82Z23qbIWlZG6nFd1/N9vyf1F0EQlJRSB8Qd0ZD9MAwVztfRqyEukGw588RxnAKWVTG0pGXIh3KGiKKoiP7Rihchb9M8n8HZBa/RX/H6DRplRBcHgiRT9M6e5+WlRpehnC6Pr6yI/pzXWkymWBbJryKgjdAbxKWNuiP6G15rMZlCn+hMYdgUOhlfEHvsGTPdTILp9kPT70kw3emWp9J0rVv+b9OWbjm0PuIk9a9IMA0NpjP2y/yGyZRQ7CFg2k1qH0GvCJnpgs/Fz+AVaY5q5FM5k5GRkQ4eHBltOF87bREAAAAASUVORK5CYII=>