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

sshpass -p $USER_PASSWORD ssh $USER_NAME@$BOARD_IP "cd $FILE_DIR; chmod +x ./run.sh"

echo "Writing startup commands to /etc/rc.local"

RC_LOCAL="{
  sudo -u $USER_NAME $FILE_DIR/run.sh start
  exit 0
} || {
  echo \"Start script failed\"
  exit 1
}"

sshpass -p $ROOT_PASSWORD ssh root@$BOARD_IP "cd /etc; echo -e '$RC_LOCAL' > rc.local"

echo "Done"