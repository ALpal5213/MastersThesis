#!/bin/sh

SCRIPT_NAME="doa_bf.py"

stop() {
  printf "Stopping $SCRIPT_NAME: "
  pkill -f $SCRIPT_NAME
  rm -f /var/lock/$SCRIPT_NAME
  echo "OK"
}

start() {
  printf "Starting $SCRIPT_NAME: "
  cd /home/analog/Desktop/code
  nohup python3 $SCRIPT_NAME > startup_error.log &
  touch /var/lock/$SCRIPT_NAME
  echo "OK"
}

restart() {
  stop
  start
}

case "$1" in
  start)
  restart
  ;;
  stop)
  stop
  ;;
  restart|reload)
  restart
  ;;
  *)
  echo "Usage: $0 {start|stop|restart}"
  exit 1
esac