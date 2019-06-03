#！/bin/sh

#nohup python -u clean_data.py > ../log/detect_modified_data.log 2>&1 &

#nohup python -u get_real_value.py > ../log/get_abnormal_real_value.log 2>&1 &

nohup python -u get_real_value.py > ../log/get_all_real_value.log 2>&1 &
