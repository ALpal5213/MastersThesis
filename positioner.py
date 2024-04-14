# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 15:24:24 2024

@author: jkris
"""

import serial
#https://www.ets-lindgren.com/sites/etsauthor/ProductsManuals/Positioners/2005(1).pdf

#ser = serial.Serial('/dev/ttyUSB0')
# help(serial)
ser = serial.Serial('/dev/ttyS0', 9600)


data = "SK020.0\n"

ser.write(data.encode('ascii'))

ser.close()