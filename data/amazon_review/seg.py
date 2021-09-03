import os
def gen_new_data(file_in,file_new1,file_new2):
    f2 = open(file_new1, 'a')
    f3= open(file_new2, 'a')
    for file in os.listdir(file_in):
        filepath = file_in + file
        print(filepath)
        with open(filepath, 'r') as f1:
            temp = []
            for line in open(filepath):
                temp.append(line.strip('\n'))
            temp = ''.join(temp)
            print(temp)
            a = temp.split(' ||| ')
            f2.writelines(''.join(a[0]) + '\n')
            f3.writelines(''.join(a[1]) + '\n')
    f1.close()
    f2.close()
    f3.close()

gen_new_data("./jp/music/parl/","./jp/parl/jp.txt","./jp/parl/en.txt" )