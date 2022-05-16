#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 14:55:48 2022

@author: giuliano
"""

def getTemp(serialObj):
    if serialObj.isOpen():
        serialObj.write(b't1\r')
        recentPacket = serialObj.readline()
        recentPacketString = recentPacket.decode('utf-8','ignore').rstrip('\n*')
        print(recentPacketString)
        return recentPacketString 
        #time.sleep(0.1)
    else:
        print('puerto cerrado')