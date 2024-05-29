#!/bin/bash
source .env

echo "Creating directory in $USER_NAME@$BOARD_IP at $FILE_DIR"
sshpass -p $USER_PASSWORD ssh $USER_NAME@$BOARD_IP "mkdir -p $FILE_DIR"

echo "Making /logs directory"
sshpass -p $USER_PASSWORD ssh $USER_NAME@$BOARD_IP "mkdir -p $FILE_DIR/logs"

echo "Sending files"
sshpass -p $USER_PASSWORD scp ./doa_bf.py $USER_NAME@$BOARD_IP:$FILE_DIR/doa_bf.py
sshpass -p $USER_PASSWORD scp ./music.py $USER_NAME@$BOARD_IP:$FILE_DIR/music.py
sshpass -p $USER_PASSWORD scp ./mvdr.py $USER_NAME@$BOARD_IP:$FILE_DIR/mvdr.py
sshpass -p $USER_PASSWORD scp ./run.sh $USER_NAME@$BOARD_IP:$FILE_DIR/run.sh
sshpass -p $USER_PASSWORD scp ./config.ini $USER_NAME@$BOARD_IP:$FILE_DIR/config.ini
sshpass -p $USER_PASSWORD scp ./calibration.py $USER_NAME@$BOARD_IP:$FILE_DIR/calibration.py

sshpass -p $USER_PASSWORD ssh $USER_NAME@$BOARD_IP "cd $FILE_DIR; chmod +x ./run.sh"
sshpass -p $ROOT_PASSWORD ssh root@$BOARD_IP "cd $FILE_DIR; chmod +x ./run.sh"

echo "Writing startup commands to /etc/rc.local"

RC_LOCAL="#!/bin/sh -e
#
# rc.local
#
# This script is executed at the end of each multiuser runlevel.
# Make sure that the script will \"exit 0\" on success or any other
# value on error.
#
# In order to enable or disable this script just change the execution
# bits.
#
# By default this script does nothing.

# Print the IP address
_IP=\$(hostname -I) || true
if [ \"\$_IP\" ]; then
  printf \"My IP address is %s\" \"\$_IP\"
fi

{
  cd $FILE_DIR
  sudo -u $USER_NAME ./run.sh start
  exit 0
} || {
  echo \"Start script failed\"
  exit 1
}"

sshpass -p $ROOT_PASSWORD ssh root@$BOARD_IP "cd /etc; echo -e '$RC_LOCAL' > rc.local"

echo "Done"