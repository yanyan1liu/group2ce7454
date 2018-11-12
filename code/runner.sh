#!/bin/bash

echo "run for setting8"
#cat settings.py
python starter.py

echo "run for setting7"
sed -i '116s/setting8/setting7/g' settings.py
#cat settings.py
python starter.py

echo "run for setting6"
sed -i '116s/setting7/setting6/g' settings.py
#cat settings.py
python starter.py

echo "run for setting5"
sed -i '116s/setting6/setting5/g' settings.py
#cat settings.py
python starter.py

echo "run for setting4"
sed -i '116s/setting5/setting4/g' settings.py
#cat settings.py
python starter.py

echo "run for setting3"
sed -i '116s/setting4/setting3/g' settings.py
#cat settings.py
python starter.py

echo "run for setting2"
sed -i '116s/setting3/setting2/g' settings.py
#cat settings.py
python starter.py

echo "run for setting1"
sed -i '116s/setting2/setting1/g' settings.py
#cat settings.py
python starter.py

echo "back setting8"
sed -i '116s/setting1/setting8/g' settings.py
#cat settings.py
