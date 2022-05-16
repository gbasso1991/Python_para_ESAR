#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 14:55:22 2022

@author: giuliano
"""

import time
import datetime as dt
import matplotlib.pyplot as plt
#from getTemp import getTemp
import serial


def getTemp(serialObj):
    if serialObj.isOpen():
        serialObj.write(b't1\r')
        recentPacket = serialObj.readline()
        #print(recentPacket)
        recentPacketString = recentPacket.decode('utf-8','ignore').rstrip('\n*')

        #print('2 -',recentPacketString)
        recentPacketString.strip()
        #print(recentPacketString, 'len', len(recentPacketString))
        #ime.sleep(0.1)
        return recentPacketString 

    else:
        print('puerto cerrado')





# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = []
ys = []


s = serial.Serial(port='/dev/ttyUSB0',baudrate=9800,bytesize=8,stopbits=1)

# Initialize communication with sensor
getTemp(s)

# Sample temperature every second for 15 seconds
for t in range(0, 10):

    # Read temperature (Celsius) from TMP102
    temp_c = getTemp(s)
    print(temp_c)
    # Add x and y to lists
    xs.append(dt.datetime.now().strftime('%H:%M:%S'))
    ys.append(temp_c)

    # Wait 1 second before sampling temperature again
    time.sleep(1.1)

# Draw plot
ax.plot(xs, ys)

# Format plot
plt.xticks(rotation=45, ha='right')
plt.subplots_adjust(bottom=0.30)
plt.title('Temperatura vs. Tiempo')
plt.ylabel('Temperatura (°C)')

# Draw the graph
plt.show()