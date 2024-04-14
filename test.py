import datetime
import sys

try:
    print(x)
except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    logFile =  open("./test.log", "a")
    logFile.write("ERROR: Initialization Block\n")
    output = str(datetime.datetime.now()) + f" - Error on line {exc_tb.tb_lineno}: \"" + str(e) + "\"\n"
    logFile.write(output)
    logFile.close()