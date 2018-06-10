0. hardware setup, refer to https://www.modmypi.com/blog/hc-sr04-ultrasonic-range-sensor-on-the-raspberry-pi
1. ssh into donkey car, execute "pip install RPi.GPIO"
2. copy hcsr04.py into ~/donkeycar/donkeycar/parts
3. copy manage_sonar.py into ~/d2
4. cd ~/d2
5. execute "python manage_sonar.py drive --model models/mypilot"
6. play as usual, the car will stop if there's obstacle ahead