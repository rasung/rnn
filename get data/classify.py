def classify_input_18(data):
    if data <= -15:
        return 1
    elif data > -15 and data <= -10:
        return 2
    elif data > -10 and data <= -7:
        return 3
    elif data > -7 and data <= -5:
        return 4
    elif data > -5 and data <= -4:
        return 5
    elif data > -4 and data <= -3:
        return 6
    elif data > -3 and data <= -2:
        return 7      
    elif data > -2 and data <= -1:
        return 8
    elif data > -1 and data <= 0:
        return 9
    elif data > 0 and data <= 1:
        return 10
    elif data > 1 and data <= 2:
        return 11
    elif data > 2 and data <= 3:
        return 12
    elif data > 3 and data <= 4:
        return 13
    elif data > 4 and data <= 5:
        return 14
    elif data > 5 and data <= 7:
        return 15
    elif data > 7 and data <= 10:
        return 16
    elif data > 10 and data <= 15:
        return 17
    elif data > 15:
        return 18

        

def classify_output_18(data):
    if data <= -50:
        return 1
    elif data > -50 and data <= -30:
        return 2
    elif data > -30 and data <= -20:
        return 2
    elif data > -20 and data <= -15:
        return 3
    elif data > -15 and data <= -12:
        return 4
    elif data > -12 and data <= -9:
        return 5
    elif data > -9 and data <= -6:
        return 6
    elif data > -6 and data <= -3:
        return 7      
    elif data > -3 and data <= 0:
        return 8
    elif data > 0 and data <= 3:
        return 9
    elif data > 3 and data <= 6:
        return 10
    elif data > 6 and data <= 9:
        return 11
    elif data > 9 and data <= 12:
        return 12
    elif data > 12 and data <= 15:
        return 13
    elif data > 15 and data <= 20:
        return 14
    elif data > 20 and data <= 30:
        return 15
    elif data > 30 and data <= 50:
        return 15
    elif data > 50:
        return 16
