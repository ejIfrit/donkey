import RPi.GPIO as GPIO   #run 'pip install RPi.GPIO' if not already installed
import time

class hcsr04(object):

	def __init__(self):
		self.running = True
		self.time = time.time()
		self.deltatime = 0.5     #set minimum time gap between firing sonar
		GPIO.setmode(GPIO.BCM)
		self.TRIG = 23
		self.ECHO = 24
		GPIO.setup(self.TRIG,GPIO.OUT)
		GPIO.setup(self.ECHO,GPIO.IN)
		self.pulse_start = time.time()
		self.pulse_end = time.time()
		self.dist = 0.01

		GPIO.output(self.TRIG, False)


	def fire_pulse(self):
		GPIO.output(self.TRIG, True)	#fire sonar pulse
		time.sleep(0.00001)
		GPIO.output(self.TRIG, False)

		i = 0
		while GPIO.input(self.ECHO)==0:	#wait for echo
			i = i+1
			if i>1000000:    #break if echo take too long to come in
				break

		self.pulse_start = time.time()  #echo comes in, record pulse start time

		j = 0
		while GPIO.input(self.ECHO)==1: #wait for echo pulse end
			j = j+1
			if j>900:    #break if echo take too long to end
				break

		self.pulse_end = time.time()   #echo ended, record pulse end time

		self.pulse_duration = self.pulse_end - self.pulse_start
		self.dist = self.pulse_duration * 17150
		self.dist = round(self.dist, 2)
		print("Distance:",self.dist,"cm")
		#time.sleep(0.1)         #threaded use


	#def update(self):         #threaded use
	#	while(self.running):
	#		self.fire_pulse()

	def run(self):
		self.fire_pulse()
		return self.dist

	#def run_threaded(self):         #threaded use
	#	return self.dist


	def shutdown(self):
		self.running = False
		GPIO.cleanup()
