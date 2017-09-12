import csv
import classify

rowdata_path = 'updown_from_19970101_to_20170831.txt'
csv_data_path = '1month_1month_19970101_20170831.csv'

SEQUNCE_LENGTH = 30
MAX_MIN_VALUE_RANGE = 30

f_row_data = open(rowdata_path, 'r')
f_csv_data = open(csv_data_path, 'w', encoding='utf-8', newline='')
wr = csv.writer(f_csv_data)

jongmok_count = 0
while True:
    jongmok_code = f_row_data.readline()
    if jongmok_code is '':
        print('finished')
        break
    size = f_row_data.readline()
    data = f_row_data.readline().split()

    jongmok_count += 1
    
    for i in range(int(size)):
        if i + SEQUNCE_LENGTH + MAX_MIN_VALUE_RANGE > int(size):
            break

        write_data = []

        for k in range(SEQUNCE_LENGTH):
            write_data.append(classify.classify_input_18(float(data[i+k])))

        max_value = -99
        min_value = 99
        pre_value = 1
        for j in range(MAX_MIN_VALUE_RANGE):
            if data[i + SEQUNCE_LENGTH + j] is "0.0":
                continue
            
            ratio = 1 + (float(data[i + SEQUNCE_LENGTH + j]) / 100)
            pre_value = pre_value * ratio
            if max_value < pre_value:
                max_value = pre_value
            if min_value > pre_value:
                min_value = pre_value
            
        max_data = int((max_value - 1) * 100)
        write_data.append(classify.classify_output_18(float(max_data)))

        min_data = int((min_value - 1) * 100)
        write_data.append(classify.classify_output_18(float(min_data)))

        wr.writerow(write_data)

        print(jongmok_count, classify.classify_output_18(max_data), classify.classify_output_18(min_data))

        
        

