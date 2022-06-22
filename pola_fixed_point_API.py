import cv2
import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
#--------------choose your fix_pointer_dir-------------
fix_point_dir = "ocr_fix_point_dir"
#fix_point_dir = "smart_fix_point_dir"
WEIGHT_DIR = "OCR_WEIGHT"
#WEIGHT_DIR = "SMART_WEIGHT"

#----create-dir--------
path = fix_point_dir
if not os.path.isdir(path):
    os.makedirs(path)

#-----------parser_how_many_node_u_need_concate-----------#
def read_json_export_csv(layer_json_file):
    try:
        node_list  = []
        merge_list = []
        node_list_attribute  = {}
        merge_list_attribute = []
        merge_flag = 0
        merge_file_count = 0
        with open(layer_json_file) as f:
            data = json.load(f)
        for name in data['config']['layers']:
            #---next_conv or dense or lstm-----
            if(merge_flag == 1 and (name['class_name']=="Dense" or name['class_name']=="Conv2D" or name['class_name']=="LSTM")):
                node_list = []
                node_list_attribute = {}

            node_list.append(name['class_name'])

            #-----take attribute-------
            if(name['class_name']=="Conv2D" and name['config']['use_bias']==True):
                node_list_attribute["use_bias"] = 1
            else:
                node_list_attribute["use_bias"] = 0
            if(name['class_name']=="BatchNormalization"):
                node_list_attribute["use_batch"] = 1
            else:
                node_list_attribute["use_batch"] = 0


            if(name['class_name']=="Dense" or name['class_name']=="Conv2D" or name['class_name']=="LSTM"):
                merge_flag = 1
                merge_list.append(node_list)
                merge_list_attribute.append(node_list_attribute)
        #print("There is your Merge_node_List_Please_Cheack and total number is {}".format(len(merge_list)))
        #for node in range(len(merge_list)):
        #    print("{}{}".format(merge_list[node] , merge_list_attribute[node]))
        merge_file_count = len(merge_list)
        return merge_list , merge_list_attribute , merge_file_count
    except:
        print("open file error please check your path , your json file location is {}".format(layer_json_file))

def find_the_most_bit_present(np_array,bit_number):
    for i in range(bit_number):
        sel = np.absolute(np_array) < 2**i
        if(sel.all()==True):
            #print("the program detect the best input shift_number is {} !!!!".format(abs(i-(bit_number-1))))
            return abs(i-(bit_number-1))

    print("Not all element == True keep going !!!")
    return bit_number-1
    
#-----------read every layer answer and analyze and export the excel to parser-----------#
def read_input_feature_map_export_csv(layer_input_feature_map,bit_number):
    try:
        picture_numpy = np.array(read_picture_from_jpg(input_feature_map = layer_input_feature_map , gray_type=True)).astype(np.float)
        best_input_feature_map = find_the_most_bit_present(np_array=picture_numpy, bit_number=bit_number)
        if(best_input_feature_map == None):
            print("FALL check your layer input feature map")
            return None
        else:
            return best_input_feature_map
    except:
        print("check your input and your bit_number is correct or not there is your bit_number {} !!!!".format(bit_number))
        return None

def read_every_layer_txt_weight(layer_weight_dir,file_length,merge_list,merge_list_attribute,bit_number):
    file_count_sel = 0
    this_node_compare = ''
    pre_node_compare = ''
    total_layer_weight_list = []
    for file_number in range(file_length):

        to_do_list = {}

        #----------check this node is major node or not---------
        if('Conv2D' in merge_list[file_number]):
            this_node_compare = 'Conv2D'
        elif('Dense' in merge_list[file_number]):
            this_node_compare = 'Dense'
        elif('LSTM' in merge_list[file_number]):
            this_node_compare = 'LSTM'
        else:
            print("error no major node in this merge node please check!!!!!")
            return None

        #----------if this node is not previous then you need change file count sel to zero--------
        if(this_node_compare!=pre_node_compare):
            file_count_sel = 0
        else:
            file_count_sel = file_count_sel + 1
        pre_node_compare = this_node_compare

        #-----------check your node and search your file weight and parse --------------
        if('Conv2D' in merge_list[file_number] or 'BatchNormalization' in merge_list[file_number]):
            if(merge_list_attribute[file_number]['use_bias']==0 and merge_list_attribute[file_number]['use_batch']==0):
                weight = read_parameter(file_serial="conv2d", file_name="weight", file_number=file_count_sel)
                weight_bit = find_the_most_bit_present(weight,bit_number=bit_number)
                to_do_list = {"quant_batch":weight_bit,"quant_batch_bias_alpha":weight_bit,"quant_batch_bias_beta":weight_bit,"quant_finish":0,"quant_obuf":weight_bit,"quant_word_size":0,"pooling_quant_finish":0}
            elif(merge_list_attribute[file_number]['use_bias']==0 and merge_list_attribute[file_number]['use_batch']==1):
                weight   = read_parameter(file_serial="conv2d", file_name="weight", file_number=file_count_sel)
                gamma    = read_parameter(file_serial="conv2d",file_name="gamma",file_number=file_count_sel)
                variance = read_parameter(file_serial="conv2d",file_name="variance",file_number=file_count_sel)
                mean     = read_parameter(file_serial="conv2d",file_name="mean",file_number=file_count_sel)
                beta     = read_parameter(file_serial="conv2d",file_name="beta",file_number=file_count_sel)
                weight_bit   = find_the_most_bit_present(weight,bit_number=bit_number)
                alpha_bit    = find_the_most_bit_present(gamma/(variance**0.5),bit_number=bit_number)
                beta_bit     = find_the_most_bit_present(beta-(gamma*mean/(variance**0.5)),bit_number=bit_number)
                to_do_list = {"quant_batch":weight_bit,"quant_batch_bias_alpha":alpha_bit,"quant_batch_bias_beta":beta_bit,"quant_finish":0,"quant_obuf":weight_bit,"quant_word_size":0,"pooling_quant_finish":0}
            elif(merge_list_attribute[file_number]['use_bias']==1 and merge_list_attribute[file_number]['use_batch']==0):
                weight   = read_parameter(file_serial="conv2d", file_name="weight", file_number=file_count_sel)
                bias     = read_parameter(file_serial="conv2d", file_name="bias"  , file_number=file_count_sel)
                weight_bit  =   find_the_most_bit_present(weight,bit_number=bit_number)
                bias_bit    =   find_the_most_bit_present(bias,bit_number=bit_number)
                to_do_list = {"quant_batch":weight_bit,"quant_batch_bias_alpha":weight_bit,"quant_batch_bias_beta":bias_bit,"quant_finish":0,"quant_obuf":weight_bit,"quant_word_size":0,"pooling_quant_finish":0}
            elif(merge_list_attribute[file_number]['use_bias']==1 and merge_list_attribute[file_number]['use_batch']==1):
                weight   = read_parameter(file_serial="conv2d", file_name="weight", file_number=file_count_sel)
                bias     = read_parameter(file_serial="conv2d", file_name="bias"  , file_number=file_count_sel)
                gamma    = read_parameter(file_serial="conv2d",file_name="gamma",file_number=file_count_sel)
                variance = read_parameter(file_serial="conv2d",file_name="variance",file_number=file_count_sel)
                mean     = read_parameter(file_serial="conv2d",file_name="mean",file_number=file_count_sel)
                beta     = read_parameter(file_serial="conv2d",file_name="beta",file_number=file_count_sel)
                weight_bit   = find_the_most_bit_present(weight,bit_number=bit_number)
                bias_bit     = find_the_most_bit_present(bias,bit_number=bit_number)
                alpha_bit    = find_the_most_bit_present(gamma/(variance**0.5),bit_number=bit_number)
                beta_bit     = find_the_most_bit_present(beta-(gamma*mean/(variance**0.5)),bit_number=bit_number)
                to_do_list = {"quant_batch":weight_bit,"quant_batch_bias_alpha":alpha_bit,"quant_batch_bias_beta":beta_bit,"quant_finish":0,"quant_obuf":weight_bit,"quant_word_size":0,"pooling_quant_finish":0}
            else:
                print("ERROR!!!!!!!!!!!!!! please check read_every_layer_export_txt_weight func ")

        if('Dense' in merge_list[file_number]):
            if(merge_list_attribute[file_number]['use_bias']==1):
                weight  = read_parameter(file_serial="conv2d", file_name="weight", file_number=file_count_sel)
                bias    = read_parameter(file_serial="conv2d", file_name="bias"  , file_number=file_count_sel)
                weight_bit   = find_the_most_bit_present(weight,bit_number=bit_number)
                bias_bit    =   find_the_most_bit_present(bias,bit_number=bit_number)
                to_do_list = {"quant_batch":weight_bit,"quant_batch_bias_alpha":weight_bit,"quant_batch_bias_beta":bias_bit,"quant_finish":0,"quant_obuf":weight_bit,"quant_word_size":0,"pooling_quant_finish":0}
            else:
                weight  = read_parameter(file_serial="conv2d", file_name="weight", file_number=file_count_sel)
                weight_bit   = find_the_most_bit_present(weight,bit_number=bit_number)
                to_do_list = {"quant_batch":weight_bit,"quant_batch_bias_alpha":weight_bit,"quant_batch_bias_beta":weight_bit,"quant_finish":0,"quant_obuf":weight_bit,"quant_word_size":0,"pooling_quant_finish":0}

        if('LSTM' in merge_list[file_number]):
            if(merge_list_attribute[file_number]['use_bias']==1):
                weight  = read_parameter(file_serial="conv2d", file_name="weight", file_number=file_count_sel)
                bias    = read_parameter(file_serial="conv2d", file_name="bias"  , file_number=file_count_sel)
                weight_bit   = find_the_most_bit_present(weight,bit_number=bit_number)
                bias_bit    =   find_the_most_bit_present(bias,bit_number=bit_number)
                to_do_list = {"quant_batch":weight_bit,"quant_batch_bias_alpha":weight_bit,"quant_batch_bias_beta":bias_bit,"quant_finish":0,"quant_obuf":weight_bit,"quant_word_size":0,"pooling_quant_finish":0}
            else:
                weight  = read_parameter(file_serial="conv2d", file_name="weight", file_number=file_count_sel)
                weight_bit   = find_the_most_bit_present(weight,bit_number=bit_number)
                to_do_list = {"quant_batch":weight_bit,"quant_batch_bias_alpha":weight_bit,"quant_batch_bias_beta":weight_bit,"quant_finish":0,"quant_obuf":weight_bit,"quant_word_size":0,"pooling_quant_finish":0}
        #---------todo something---------check the weight-----
        #print(to_do_list)
        total_layer_weight_list.append(to_do_list)
        
        # --------------------- excel format by c++ parser reading --------------------------
        #merge_node_vector[quant_list_index]->quant_batch                = target_layer_quant_vector[quant_list_index][0];//11
        #merge_node_vector[quant_list_index]->quant_batch_bias_alpha     = target_layer_quant_vector[quant_list_index][1];//11
        #merge_node_vector[quant_list_index]->quant_batch_bias_beta      = target_layer_quant_vector[quant_list_index][2];//11
        #merge_node_vector[quant_list_index]->quant_finish               = target_layer_quant_vector[quant_list_index][3];//0
        #merge_node_vector[quant_list_index]->quant_obuf                 = target_layer_quant_vector[quant_list_index][4];//0 11
        #merge_node_vector[quant_list_index]->quant_word_size            = target_layer_quant_vector[quant_list_index][5];//0
        #merge_node_vector[quant_list_index]->pooling_quant_finish       = target_layer_quant_vector[quant_list_index][6];//0
    return total_layer_weight_list


def read_every_layer_csv_answer(layer_answer_dir,file_length,merge_list,bit_number):
    file_count_sel = 0
    this_node_compare = ''
    pre_node_compare = ''
    choose_dir_path = ''
    total_layer_answer_list = []
    for file_number in range(file_length):

        reader = []

        #----------check this node is major node or not---------
        if('Conv2D' in merge_list[file_number]):
            this_node_compare = 'Conv2D'
        elif('Dense' in merge_list[file_number]):
            this_node_compare = 'Dense'
        elif('LSTM' in merge_list[file_number]):
            this_node_compare = 'LSTM'
        else:
            print("error no major node in this merge node please check!!!!!")
            return None
        #----------if this node is not previous then you need change file count sel to zero--------
        if(this_node_compare!=pre_node_compare):
            file_count_sel = 0
        else:
            file_count_sel = file_count_sel + 1
        pre_node_compare = this_node_compare

        if('Conv2D' in merge_list[file_number]):
            choose_dir_path = layer_answer_dir + "/" + "conv2d" + "_" + str(file_count_sel)
        elif('Dense' in merge_list[file_number]):
            choose_dir_path = layer_answer_dir + "/" + "dense" + "_" + str(file_count_sel)
        elif('LSTM' in merge_list[file_number]):
            choose_dir_path = layer_answer_dir + "/" + "lstm" + "_" + str(file_count_sel)
        else:
            print("Sorry we cannot support this merge node please check !!!!!")

        #---------------------dfs all the file and compare the data ----------------
        for filename in os.listdir(choose_dir_path):
            if(not os.path.isdir(filename)):
                with open(os.path.join(choose_dir_path,filename),'r') as f:
                    reader_tmp = list(csv.reader(f,delimiter=','))
                reader.append(reader_tmp)
        reader = np.array(reader,dtype=float)
        reader_bit = find_the_most_bit_present(reader,bit_number=bit_number)
        total_layer_answer_list.append(reader_bit)
    return total_layer_answer_list

def export_parser_design_csv(json_file_location,input_feature_map_location,weight_location,answer_location,bit_number):
    merge_list , merge_list_attribute , merge_file_count= read_json_export_csv(layer_json_file=json_file_location)
    input_feature_shift = read_input_feature_map_export_csv(layer_input_feature_map=input_feature_map_location,bit_number=bit_number)
    total_layer_weight_list = read_every_layer_txt_weight(layer_weight_dir=weight_location,file_length=merge_file_count,merge_list=merge_list,merge_list_attribute=merge_list_attribute,bit_number=bit_number)
    total_layer_answer_list = read_every_layer_csv_answer(layer_answer_dir=answer_location,file_length=merge_file_count,merge_list=merge_list,bit_number=bit_number)
    print(total_layer_answer_list)
    
    for file_number in range(merge_file_count):
        now_position   = file_number
        next_position  = file_number + 1
        pre_position   = file_number - 1
        final_position = merge_file_count - 1

        #----------------check if this node is the first node in there, we need to alignment to input_feature_map------------
        if(now_position==0):
            total_layer_weight_list[now_position]['quant_batch_bias_beta'] = input_feature_shift
        else:
            if(total_layer_weight_list[pre_position]['pooling_quant_finish']!=0):
                total_layer_weight_list[now_position]['quant_batch_bias_beta'] = total_layer_weight_list[pre_position]['quant_batch_bias_beta'] - total_layer_weight_list[pre_position]['pooling_quant_finish']
            elif(total_layer_weight_list[pre_position]['quant_finish']!=0):
                total_layer_weight_list[now_position]['quant_batch_bias_beta'] = total_layer_weight_list[pre_position]['quant_batch_bias_beta'] - total_layer_weight_list[pre_position]['quant_finish']
            else:
                total_layer_weight_list[now_position]['quant_batch_bias_beta'] = total_layer_weight_list[pre_position]['quant_batch_bias_beta']
        
        #----------------check if this node is the last node --------------
        if(now_position != final_position):
            if('MaxPooling2D' in merge_list[now_position]):
                total_layer_weight_list[now_position]['pooling_quant_finish'] = total_layer_weight_list[now_position]['quant_batch_bias_beta'] - total_layer_answer_list[now_position]
            else:
                total_layer_weight_list[now_position]['quant_finish'] = total_layer_weight_list[now_position]['quant_batch_bias_beta'] - total_layer_answer_list[now_position]
            #print(total_layer_answer_list[now_position])
    
    count_number = 0
    fp = open("./"+fix_point_dir+"/"+"SMART_testing.csv",'w')
    for file_number in range(merge_file_count):
        print("layers_{},".format(str(file_number)),end='',file=fp)
        for node in range(len(merge_list[file_number])):
            if(node!=len(merge_list[file_number])-1):
                print("{},".format(count_number),end='',file=fp)
            else:
                print("{}".format(count_number),file=fp)
            count_number = count_number + 1
        print("quant_{},{},{},{},{},{},{},{}".format(str(file_number),
                                                    total_layer_weight_list[file_number]['quant_batch'],
                                                    total_layer_weight_list[file_number]['quant_batch_bias_alpha'],
                                                    total_layer_weight_list[file_number]['quant_batch_bias_beta'],
                                                    total_layer_weight_list[file_number]['quant_finish'],
                                                    total_layer_weight_list[file_number]['quant_obuf'],
                                                    total_layer_weight_list[file_number]['quant_word_size'],
                                                    total_layer_weight_list[file_number]['pooling_quant_finish']),file=fp)
    fp.close()
        
def split_list(l, n):
    # 將list分割 (l:list, n:每個matrix裡面有n個元素)
    for idx in range(0, len(l), n):
        yield literal_eval("0x" + l[idx:idx+n])

def read_hardware_output(Hex_file_location,shift_number,output_data_offset,total_eight_number):
    data_array = []
    return_array = []
    Hex_file = open(Hex_file_location, 'r')
    Hex_Lines = Hex_file.readlines()
    for line in Hex_Lines:
        input_data = line.strip()
        result = list(split_list(input_data,4))
        result.reverse()
        data_array.append(result)
    print(data_array)
    print(len(data_array))
    resize_number = int(total_eight_number / output_data_offset)
    for i in range(output_data_offset):
        for j in range(resize_number):
            return_array.append(data_array[i + j * output_data_offset])
    return_array = np.array(return_array)/2**shift_number
    return return_array

def read_picture_from_jpg(input_feature_map,gray_type=False):
    reader = []
    if(gray_type==True):
        try:
            with open(input_feature_map,'r') as f:
                reader_0 = list(csv.reader(f,delimiter=','))
            reader.append(reader_0)
            
        except:
            print("read gray picture error!!")    
    elif(gray_type==False):
        try:
            img = cv2.imread(input_feature_map)
            (B,G,R) = cv2.split(img)
            reader.append(R)
            reader.append(G)
            reader.append(B)
            reader = np.array(reader,dtype=float)
            print(reader.shape)
            reader = reader/256
            reader = reader.tolist()
        except:
            print("read normal picture error!!")
    return reader
def read_input_feature_map_csv_file(csv_file_location1,csv_file_location2,csv_file_location3):
    try:
        reader = []
        with open(csv_file_location1,'r') as f:
            reader_0 = list(csv.reader(f,delimiter=','))
        with open(csv_file_location2,'r') as f:
            reader_1 = list(csv.reader(f,delimiter=','))
        with open(csv_file_location3,'r') as f:
            reader_2 = list(csv.reader(f,delimiter=','))
        reader.append(reader_0)
        reader.append(reader_1)
        reader.append(reader_2)
    except:
        print("read input feature fail")
    else:
        return reader

def read_parameter(file_serial,file_name,file_number):
    try:
        with open("./"+WEIGHT_DIR+"/"+str(file_serial)+"_"+str(file_number)+"_"+str(file_name)+".txt","r")as f:
            para_vector = list(csv.reader(f,delimiter=','))
    except:
        print("please check your "+str(file_name)+" name or weight location")
    else:
        para_vector = np.array(para_vector,dtype=float)
        #print("{} file is open and read".format(str(file_name)))
    return para_vector

def Saturation(input_value,word_size):
    if(input_value> 2**(word_size-1)-1):
        input_value = 2**(word_size-1)-1
    elif(input_value<(-1 * 2**(word_size-1))):
        input_value = -1*2**(word_size-1)
    else:
        input_value = input_value
    return input_value

def CONV2D(input_feature_map,padding_size,kernel_size,kernel_stride,output_channel,weight_array,fixed_point,bias_array):
    input_channel = len(input_feature_map)
    if(fixed_point == 0):
        #input_feature_map
        padding_feature_map = np.zeros((input_channel,(len(input_feature_map[0])+padding_size*2),(len(input_feature_map[0][0])+padding_size*2),1))
        result_array = np.zeros((output_channel,len(padding_feature_map[0])-kernel_size+1,len(padding_feature_map[0][0])-kernel_size+1))
        bias_array = np.array(bias_array)
        for i in range(input_channel):
            padding_feature_map[i,padding_size:len(padding_feature_map[0])-padding_size,padding_size:len(padding_feature_map[0][0])-padding_size,0] = input_feature_map[i]
    else:
        #input_feature_map
        padding_feature_map = np.zeros((input_channel,(len(input_feature_map[0])+padding_size*2),(len(input_feature_map[0][0])+padding_size*2),1))
        for i in range(input_channel):
            padding_feature_map[i,padding_size:len(padding_feature_map[0])-padding_size,padding_size:len(padding_feature_map[0][0])-padding_size,0] = input_feature_map[i]
        padding_feature_map = (padding_feature_map*2**fixed_point).astype("int64")

        #weight,bias,mean,var~~~
        weight_array = (weight_array*2**fixed_point).astype("int64")
        result_array = np.zeros((output_channel,len(padding_feature_map[0])-kernel_size+1,len(padding_feature_map[0][0])-kernel_size+1)).astype("int64")
        
        #if you want check your bias , open the print underline thx~
        #print("you got the bias please check  these is your org bias , {}".format(bias_array))
        bias_array = np.array(bias_array*2**fixed_point).astype("int64")
        #print("you got the bias please check {}".format(bias_array))
    print(result_array.shape)
    #----------------------calc-begin------------------------
    for och in range(output_channel):
        for ich in range(input_channel):
            #print(weight_array[och][ich][0])
            #print(weight_array[och][ich][1])
            #print(weight_array[och][ich][2])
            #print(weight_array[och][ich][3])
            #print(weight_array[och][ich][4])
            #print(weight_array[och][ich][5])
            #print(weight_array[och][ich][6])
            #print(weight_array[och][ich][7])
            #print(weight_array[och][ich][8])
            #print("---------------------")
            if(len(padding_feature_map[0])-kernel_size==0 or padding_size==0):
                check_bias_row = 1
            else:
                check_bias_row = 0
            if(len(padding_feature_map[0][0])-kernel_size==0 or padding_size==0):
                check_bias_col = 1
            else:
                check_bias_col = 0
            for row in range(0,len(padding_feature_map[ich])-kernel_size+padding_size+check_bias_row,kernel_stride):
                for col in range(0,len(padding_feature_map[ich][row])-kernel_size+padding_size+check_bias_col,kernel_stride):
                    if(kernel_size==3):
                        '''
                                  32bit = 16 bit * 16bit
                                  32bit = 16 bit * 16bit
                                  32bit = 16 bit * 16bit
                                  32bit = 16 bit * 16bit
                                  32bit = 16 bit * 16bit
                                  32bit = 16 bit * 16bit
                                  32bit = 16 bit * 16bit
                                  32bit = 16 bit * 16bit
                                + 32bit = 16 bit * 16bit
                                ------------------------
                                  40 bit >> quant_value -> 16bit
                                  this is the reason why i need set value type to int64,we cannot change the value when we calc
                        '''
                        result = padding_feature_map[ich][row  ][col  ] * weight_array[och][ich][0] + \
                                 padding_feature_map[ich][row  ][col+1] * weight_array[och][ich][1] + \
                                 padding_feature_map[ich][row  ][col+2] * weight_array[och][ich][2] + \
                                 padding_feature_map[ich][row+1][col  ] * weight_array[och][ich][3] + \
                                 padding_feature_map[ich][row+1][col+1] * weight_array[och][ich][4] + \
                                 padding_feature_map[ich][row+1][col+2] * weight_array[och][ich][5] + \
                                 padding_feature_map[ich][row+2][col  ] * weight_array[och][ich][6] + \
                                 padding_feature_map[ich][row+2][col+1] * weight_array[och][ich][7] + \
                                 padding_feature_map[ich][row+2][col+2] * weight_array[och][ich][8]
                        #if(col==0):
                        #print("{} ,{} ,{} ,{} ,{}".format(och,ich,row,col,result))
                        #print("---------------------")
                        #print("{},{},{}".format(padding_feature_map[ich][row  ][col  ],padding_feature_map[ich][row  ][col+1],padding_feature_map[ich][row  ][col+2]))
                        #print("{},{},{}".format(padding_feature_map[ich][row+1][col  ],padding_feature_map[ich][row+1][col+1],padding_feature_map[ich][row+1][col+2]))
                        #print("{},{},{}".format(padding_feature_map[ich][row+2][col  ],padding_feature_map[ich][row+2][col+1],padding_feature_map[ich][row+2][col+2]))
                        #print("---------------------")
                        #print(result)
                    elif(kernel_size==2):
                        result = padding_feature_map[ich][row  ][col  ] * weight_array[och][ich][0] + \
                                 padding_feature_map[ich][row  ][col+1] * weight_array[och][ich][1] + \
                                 padding_feature_map[ich][row+1][col  ] * weight_array[och][ich][2] + \
                                 padding_feature_map[ich][row+1][col+1] * weight_array[och][ich][3]
                        #print("{} ,{} ,{} ,{} ,{}".format(och,ich,row,col,result))
                        #print("---------------------")
                        #print("{},{}".format(padding_feature_map[ich][row  ][col  ],padding_feature_map[ich][row  ][col+1]))
                        #print("{},{}".format(padding_feature_map[ich][row+1][col  ],padding_feature_map[ich][row+1][col+1]))
                        #print("---------------------")
                    else:
                        result = padding_feature_map[ich][row  ][col  ] * weight_array[och][ich][0]
                    #if(fixed_point!=0):
                    #    result = Saturation(result / 2**fixed_point , 16)
                    result_array[och][row][col] = result_array[och][row][col] + result
                    #print("{} ,{} ,{} ,{} ,{}".format(och,ich,row,col,result))            
    
    #----------------------fix_data_process-begin----------------32->16bit
    if(fixed_point!=0):
        result_array = (result_array / 2**fixed_point).astype("int64")
    #print(result_array[0])
    #----------------------add_bias------------------------ 16bit weight + 16bit bias
    if(bias_array.any()!=0):
        print("add bias now!!!!!!!!!!!!!!!!!!")
        for i in range(len(result_array)):
            result_array[i] = (result_array[i] + bias_array[i])

    return result_array

def MAX_POOLING(input_feature_map,stride_size,stride_kernel):
    pool_feature_map = np.zeros((
        len(input_feature_map),
        int(len(input_feature_map[0])/stride_kernel[0]),
        int(len(input_feature_map[0][0])/stride_kernel[1])
                                         ))
    for ich in range(len(input_feature_map)):
        for row in range(0,len(input_feature_map[0]),stride_size[0]):
            for col in range(0,len(input_feature_map[0][0]),stride_size[1]):
                if(stride_size[0]==2 and stride_size[1]==2):
                    pool_result = max(input_feature_map[ich][row  ][col  ],
                                      input_feature_map[ich][row  ][col+1],
                                      input_feature_map[ich][row+1][col  ],
                                      input_feature_map[ich][row+1][col+1]
                                    )
                elif(stride_size[0]==2 and stride_size[1]==1):
                    pool_result = max(input_feature_map[ich][row  ][col  ],
                                      input_feature_map[ich][row+1][col  ],
                                    )
                elif(stride_size[0]==1 and stride_size[1]==2):
                    pool_result = max(input_feature_map[ich][row][col  ],
                                      input_feature_map[ich][row][col+1],
                                    )
                elif(stride_size[0]==1 and stride_size[1]==1):
                    pass #do_nothing
                else:
                    print("please check your maxpooling parameter stride_size[0] is {} and stride_size[1] is {}".format(stride_size[0],stride_size[1]))
                #print("{},{},{},{}".format(ich,row,col,pool_result))
                pool_feature_map[ich][int(row/stride_kernel[0])][int(col/stride_kernel[1])] = pool_result
    return pool_feature_map

def Batch_Normalization(input_feature_map,alpha_array,beta_array,fixed_point):
    if(fixed_point!=0):
        alpha_array = (alpha_array*2**fixed_point).astype("int64")
        beta_array  = (beta_array*2**fixed_point).astype("int64")
        batch_input_feature_map = np.zeros_like(input_feature_map)
        #((alpha x input_feature_map) + beta) / 2**fixed_point 這個想法這邊不會通 因為我輸入的值是16bit
        #相乘之後是最大會到32bit先除下去在+上16bit的beta，如果不這樣做我加上去的只是一個沒用的廢值
        for i in range(len(input_feature_map)):
            batch_input_feature_map[i] = ((input_feature_map[i] * alpha_array[i] / 2**fixed_point) + beta_array[i]).astype("int64")
    else:
        batch_input_feature_map = np.zeros_like(input_feature_map)
        for i in range(len(input_feature_map)):
            batch_input_feature_map[i] = (input_feature_map[i] * alpha_array[i] + beta_array[i])
    print(input_feature_map[0][0][0] , alpha_array[0] , beta_array[0] , ((input_feature_map[0][0][0]*alpha_array[0]/2**fixed_point)+beta_array[0])/2**fixed_point)
    return batch_input_feature_map

def Leakyrelu(input_feature_map,leakyrelu,fixed_point):
    if(fixed_point!=0):
        leakyrelu = np.array(leakyrelu * 2**fixed_point).astype("int64")
        relu_input_feature_map = np.zeros_like(input_feature_map).astype("int64")
        for i in range(len(input_feature_map)):
            relu_input_feature_map[i] = (input_feature_map[i] * leakyrelu / 2**fixed_point).astype("int64")
    else:
        relu_input_feature_map = np.zeros_like(input_feature_map)
        for i in range(len(input_feature_map)):
            relu_input_feature_map[i] = (input_feature_map[i] * leakyrelu)
            
    return relu_input_feature_map

def nrelu(input_feature_map,fixed_point):
    relu_input_feature_map = np.zeros_like(input_feature_map)
    for i in range(len(input_feature_map)):
        relu_input_feature_map[i] = np.maximum(input_feature_map[i],0) 
    return relu_input_feature_map
    
def layer_information(TODO_List,layer_number,input_feature_map_array,conv,max_pool,relu,batch,output_channel,branch_node):
    # parser the information and init
    conv_en     ,conv_kernel        ,conv_stride   ,padding_size         = func_parameter_parser_conv(conv)
    max_pool_en ,max_pool_kernel    ,max_pool_stride                     = func_parameter_parser_max_pool(max_pool)
    relu_en     ,relu_value                                              = func_parameter_parser_relu(relu)
    batch_en    ,weight_array    ,alpha   ,beta  ,bias_array             = func_parameter_parser_batch(batch)
    mse_array = []
    next_layer_array = []
    fixed_branch_node_array = []
    float_branch_node_array = []

    # find the best point location in each location
    for number in range(2):
        print("##########  layer_" + str(number)+"  ##########")
        # choose the pre_layer_input
        if(number==0):
            input_feature_map = input_feature_map_array["floating_answer"]
        elif(number!=0 and input_feature_map_array["fixed_answer"].any()!=0): 
            input_feature_map = input_feature_map_array["fixed_answer"]
        else:
            input_feature_map = input_feature_map_array["floating_answer"]

        # start do this layer
        for mission in TODO_List:
            if(conv_en and mission=="conv"):
                result_array = CONV2D(input_feature_map=input_feature_map,padding_size=padding_size,kernel_size=conv_kernel,kernel_stride=conv_stride,output_channel=output_channel,weight_array=weight_array,fixed_point=number,bias_array=bias_array)
                if(number==0):
                    try:
                        os.stat("./"+fix_point_dir+"/"+str(layer_number)+"_conv_floating_answer")
                    except:
                        os.mkdir("./"+fix_point_dir+"/"+str(layer_number)+"_conv_floating_answer")
                    for i in range(len(result_array)):
                        np.savetxt("./"+fix_point_dir+"/"+str(layer_number)+"_conv_floating_answer/"+str(number)+"_"+str(i)+"_"+str(mission)+"_number.csv",result_array[i]/2**number,delimiter=",")
                else:
                    np.savetxt("./"+fix_point_dir+"/"+str(layer_number)+'_'+str(number)+"_"+str(i)+"_"+str(mission)+"_number.csv",result_array[0]/2**number,delimiter=",")
            elif(batch_en and mission=="batch"):
                result_array = Batch_Normalization(input_feature_map=result_array,alpha_array=alpha,beta_array=beta,fixed_point=number)
                if(number==0):
                    try:
                        os.stat("./"+fix_point_dir+"/"+str(layer_number)+"_batch_floating_answer")
                    except:
                        os.mkdir("./"+fix_point_dir+"/"+str(layer_number)+"_batch_floating_answer")
                    for i in range(len(result_array)):
                        np.savetxt("./"+fix_point_dir+"/"+str(layer_number)+"_batch_floating_answer/"+str(number)+"_"+str(i)+"_"+str(mission)+"_number.csv",result_array[i]/2**number,delimiter=",")
                else:
                    np.savetxt("./"+fix_point_dir+"/"+str(layer_number)+'_'+str(number)+"_"+str(mission)+"_number.csv",result_array[0]/2**number,delimiter=",")
            elif(relu_en and (mission=="relu" or mission=="Leakyrelu")):
                if(mission=="Leakyrelu"):
                    result_array = Leakyrelu(input_feature_map=result_array,leakyrelu=relu_value,fixed_point=number)
                elif(mission=="relu"):
                    result_array = nrelu(input_feature_map=result_array,fixed_point=number)
                if(number==0):
                    try:
                        os.stat("./"+fix_point_dir+"/"+str(layer_number)+"_relu_floating_answer")
                    except:
                        os.mkdir("./"+fix_point_dir+"/"+str(layer_number)+"_relu_floating_answer")
                    for i in range(len(result_array)):
                        np.savetxt("./"+fix_point_dir+"/"+str(layer_number)+"_relu_floating_answer/"+str(number)+"_"+str(i)+"_"+str(mission)+"_number.csv",result_array[i]/2**number,delimiter=",")
                else:
                    np.savetxt("./"+fix_point_dir+"/"+str(layer_number)+'_'+str(number)+"_"+str(mission)+"_number.csv",result_array[0]/2**number,delimiter=",")
            elif(max_pool_en and mission=="max_pool"):
                result_array = MAX_POOLING(input_feature_map=result_array,stride_size=max_pool_stride,stride_kernel=max_pool_kernel)
                if(number==0):
                    try:
                        os.stat("./"+fix_point_dir+"/"+str(layer_number)+"_maxpool_floating_answer")
                    except:
                        os.mkdir("./"+fix_point_dir+"/"+str(layer_number)+"_maxpool_floating_answer")
                    for i in range(len(result_array)):
                        np.savetxt("./"+fix_point_dir+"/"+str(layer_number)+"_maxpool_floating_answer/"+str(number)+"_"+str(i)+"_"+str(mission)+"_number.csv",result_array[i]/2**number,delimiter=",")
                else:
                    np.savetxt("./"+fix_point_dir+"/"+str(layer_number)+'_'+str(number)+"_"+str(mission)+"_number.csv",result_array[0]/2**number,delimiter=",")
            else:
                print("you got some error at TODO_List and your mission is {}".format(mission))

            if(mission == branch_node and number!=0):
                fixed_branch_node_array.append(result_array)
            elif(mission == branch_node and number==0):
                float_branch_node_array.append(result_array)
            else:
                pass

        if(number==0): #floating answer
            floating_answer = result_array
        else:
            mse_array.append(np.square(floating_answer - (result_array/2**number)).mean())
            next_layer_array.append(result_array)
        
        # save the answer from floating_point and fixed_point 
        #np.savetxt("./fix_point_dir/"+str(layer_number)+'_'+str(number)+"_relu_number.csv",result_array[0]/2**number,delimiter=",")
    
    #show the array when find the min value
    print("the list of mse_array is {}".format(mse_array))
    print("the most value in mse_array is {}".format(min(mse_array)))
    print("the most value in mse_array index is << {}".format(mse_array.index(min(mse_array))+1))
    if(not isinstance(branch_node,type(None))):
        return floating_answer , next_layer_array[mse_array.index(min(mse_array))]/2**(mse_array.index(min(mse_array))+1) , float_branch_node_array[-1] , fixed_branch_node_array[mse_array.index(min(mse_array))]/2**(mse_array.index(min(mse_array))+1)
    else:
        return floating_answer , next_layer_array[mse_array.index(min(mse_array))]/2**(mse_array.index(min(mse_array))+1) , None , None
    
    
def func_parameter_parser_conv(para):
    if(para[0]=="conv"):
        conv_en = 1
        conv_kernel = para[1]
        conv_stride = para[2]
        padding_size= para[3]
        print("conv_kernel = {} , conv_stride = {}, padding_size = {}".format(conv_kernel,conv_stride,padding_size))
        return conv_en,conv_kernel,conv_stride,padding_size
    else:
        print("you dont use func_parameter_parser_conv func!!!")
        return None,None,None,None

def func_parameter_parser_max_pool(para):
    if(para[0]=="max_pool"):
        max_pool_en = 1
        max_pool_kernel = para[1]
        max_pool_stride = para[2]
        print("max_pool_kernel = {} , max_pool_stride = {}".format(max_pool_kernel,max_pool_stride))
        return max_pool_en,max_pool_kernel,max_pool_stride
    else:
        print("you dont use func_parameter_parser_max_pool func!!!")
        return None,None,None
def func_parameter_parser_relu(para):        
    if(para[0]=="relu"):
        relu_en = 1
        relu_value = None
        print("relu_value = {}".format(relu_value))
        return relu_en,relu_value
    elif(para[0]=="Leakyrelu"):
        relu_en = 1
        relu_value = para[1]
        print("relu_value = {}".format(relu_value))
        return relu_en,relu_value
    else:
        print("you dont use func_parameter_parser_relu func!!!")
        return None,None
def func_parameter_parser_batch(para):#en,batchnumber,file_number,weight,bias,gamma,variance,mean,beta
    #para init
    weight      = []    # per parameter array
    bias        = []    # per parameter array
    gamma       = []    # per parameter array
    variance    = []    # per parameter array
    mean        = []    # per parameter array
    beta        = []    # per parameter array

    if(para[0]=="batch"):
        batch_en = 1
        batch_number = para[1]
        print("batch_number is {}".format(batch_number))
        for i in range(2,len(para)):
            if(para[i][0]=="weight"):
                weight   = read_parameter(file_serial="conv2d",file_name="weight",file_number=batch_number).reshape((para[i][1],para[i][2],para[i][3]))
                print("open weight ready!!!")
            elif(para[i]=="bias"):
                bias     = read_parameter(file_serial="conv2d",file_name="bias",file_number=batch_number)
                print("open bias ready!!!")
            elif(para[i]=="gamma"):
                gamma    = read_parameter(file_serial="conv2d",file_name="gamma",file_number=batch_number)
                print("open gamma ready!!!")
            elif(para[i]=="variance"):
                variance = read_parameter(file_serial="conv2d",file_name="variance",file_number=batch_number)
                print("open variance ready!!!")
            elif(para[i]=="mean"):
                mean     = read_parameter(file_serial="conv2d",file_name="mean",file_number=batch_number)
                print("open mean ready!!!")
            elif(para[i]=="beta"):
                beta     = read_parameter(file_serial="conv2d",file_name="beta",file_number=batch_number)
                print("open beta ready!!!")
            else:
                print(para[i])
                print("sorry you got some error in func_parameter_parser_batch")
        if(len(gamma)!=0 and len(variance)!=0 and len(mean)!=0 and len(beta)!=0):
            alpha    = gamma/(variance**0.5)
            beta     = beta-(gamma*mean/(variance**0.5))
            #print("alpha!!!!!!!!!!!!!!\n {}".format(alpha))
            #print("beta!!!!!!!!!!!!!!\n {}".format(beta))
            return batch_en , weight , alpha , beta , 0
        elif(len(weight)!=0 and len(bias)!=0):
            return batch_en , weight , None , None , bias
        else:
            return batch_en , weight , None , None , 0
    else:
        print("you dont use func_parameter_parser_batch func!!!")
        return None,None,None,None,0
        
def concatenate(pre_layer,late_layer):

    #conbine two layer
    concate_layer = np.append(pre_layer,late_layer,axis=0)
    
    return concate_layer

def upsampling2d(layer,upsampling_size):
    
    #upsampling size 
    upsampling2d_layer = layer.repeat(upsampling_size, axis=1).repeat(upsampling_size, axis=2)

    return upsampling2d_layer

def Flatten(float_layer):
    flatten_data = []
    for y_size in range(len(float_layer[0])):
        for x_size in range(len(float_layer[0][0])):
            for deep in range(len(float_layer)):
                flatten_data.append(float_layer[deep][y_size][x_size])
    return np.array(flatten_data)

def Dense(float_layer,fixed_layer,file_number,units,weight_shape,bias_shape,relu_type):
    
    try:
        os.stat("./"+fix_point_dir+"/dense_answer_"+str(file_number))
    except:
        os.mkdir("./"+fix_point_dir+"/dense_answer_"+str(file_number))
        
    #fully connect layer
    dense_weight = read_parameter(file_serial="dense",file_name="weight",file_number=file_number)
    dense_bias   = read_parameter(file_serial="dense",file_name="bias",file_number=file_number)
    mse_array    = []
    fixed_array  = []
    float_result = np.zeros([float_layer.shape[0],units])
    fixed_result = np.zeros([fixed_layer.shape[0],units])
    #reshape your weight from txt
    dense_weight = dense_weight.reshape(weight_shape[0],weight_shape[1])
    dense_bias = dense_bias.reshape(bias_shape[0],bias_shape[1])
    for number in range(2):
        if(number==0):
            float_result = np.dot(float_layer,dense_weight) + dense_bias
            if(relu_type=="relu"):
                float_result = nrelu(input_feature_map = float_result, fixed_point = number)
            else:
                pass
            np.savetxt("./"+fix_point_dir+"/dense_answer_"+str(file_number)+"/float_answer.csv",float_result,delimiter=",")
        else:
            shift_layer        = (fixed_layer*2**number).astype("int64")
            shift_dense_weight = (dense_weight*2**number).astype("int64")
            shift_bias         = (dense_bias*2**number).astype("int64")
            fixed_result       = np.dot(shift_layer,shift_dense_weight)/2**number + shift_bias
            if(relu_type=="relu"):
                fixed_result = nrelu(input_feature_map = fixed_result, fixed_point = number)
            else:
                pass
            np.savetxt("./"+fix_point_dir+"/dense_answer_"+str(file_number)+"/fixed_answer.csv",fixed_result,delimiter=",")
            fixed_array.append(fixed_result)
            mse_array.append(np.square(float_result - (fixed_result/2**number).astype("int64")).mean())
    print("the list of mse_array is {}".format(mse_array))
    print("the most value in mse_array is {}".format(min(mse_array)))
    print("the most value in mse_array index is << {}".format(mse_array.index(min(mse_array))+1))
    '''    
    for i in range(len(dense_result)):
        fp = open("./fix_point_dir/dense_floating_answer/"+str(i)+"_number.csv",'w')
        for j in range(len(dense_result[i])):
            if(j!=len(dense_result[i])-1):
                print(dense_result[i][j],file=fp,end=',')
            else:
                print(dense_result[i][j],file=fp)    
    '''
    return np.array(float_result) , np.array(fixed_array[mse_array.index(min(mse_array))])/2**(mse_array.index(min(mse_array))+1)
    
def squeeze(layer,dim):
    
    #squeeze the first dim
    squeeze_layer = np.squeeze(layer,axis=dim)
    result_array = np.zeros_like(squeeze_layer).T
    try:
        os.stat("./"+fix_point_dir+"/squeeze_floating_answer")
    except:
        os.mkdir("./"+fix_point_dir+"/squeeze_floating_answer")
    for data_one in range(len(squeeze_layer[0])):#31
        fp = open("./"+fix_point_dir+"/squeeze_floating_answer/"+str(data_one)+".csv",'w')
        for data_two in range(len(squeeze_layer)):#512
            if(data_two!=len(squeeze_layer)-1):
                print(squeeze_layer[data_two][data_one],file=fp,end=',')
            else:
                print(squeeze_layer[data_two][data_one],file=fp)
            result_array[data_one][data_two] = squeeze_layer[data_two][data_one]
            print("data_one is {} , data_two is  {} , ans is {}".format(data_one,data_two,result_array[data_one][data_two]))
    return result_array

def sigmoid(x):
    y = 1/(1+np.exp(-1*x))
    return y
def tanh(x):
    y = 2*sigmoid(2*x)-1
    # y = (np.exp(x)-np.exp(-1*x))/(np.exp(x)+np.exp(-1*x))
    return y

def hard_sigmoid(x,shift_number):
    #hard_sigmoid_x = x/2**shift_number
    y = x*0.2+0.5
    y = np.clip(y,0,1)
    return y

def hard_tanh(x,shift_number):
    y = 2*hard_sigmoid(2*x,shift_number)-1
    y = np.clip(y,-1,1)
    return y

def shift_result(value,shift_number):
    return value*2**shift_number

def fixed_sigmoid_lookuptable(input_value,shift_number):
    #data range
    first_value = shift_result(5,shift_number)
    second_value = shift_result(2.375,shift_number) #2**1+2**-2+2**-3
    third_value = shift_result(1,shift_number)
    abs_value = shift_result(abs(input_value),shift_number)
    sigmoid_answer = np.zeros_like(abs_value)
    for i in range(len(sigmoid_answer)):
        for j in range(len(sigmoid_answer[i])):
            if(abs_value[i][j] >= first_value):
                sigmoid_answer[i][j] = shift_result(1,shift_number)
            elif(abs_value[i][j] >= second_value and abs_value[i][j] < first_value):
                sigmoid_answer[i][j] = shift_result(abs_value[i][j],-5) + shift_result(0.84375,shift_number) #2**-1+2**-2+2**-4+2**-5
            elif(abs_value[i][j] >= third_value and abs_value[i][j] < second_value):
                sigmoid_answer[i][j] = shift_result(abs_value[i][j],-3) + shift_result(0.625,shift_number)#2**-1+2**-3
            elif(abs_value[i][j] >= 0 and abs_value[i][j] < third_value):
                sigmoid_answer[i][j] = shift_result(abs_value[i][j],-2) + shift_result(0.5,shift_number)#2**-1
            else:
                print(abs_value[i][j])
                print("please check your input number !!!!")
    #sigmoid_answer = shift_result(1,shift_number)-sigmoid_answer
    for i in range(len(sigmoid_answer)):
        for j in range(len(sigmoid_answer[i])):
            if(input_value[i][j]<0):
                sigmoid_answer[i][j] = shift_result(1,shift_number)-sigmoid_answer[i][j]
            else:
                sigmoid_answer[i][j] = sigmoid_answer[i][j]
    return sigmoid_answer/2**shift_number

def fixed_tanh_lookuptable(input_value,shift_number):
    data = 2 * fixed_sigmoid_lookuptable(2*input_value,shift_number)-1#shift_result(1,shift_number)
    return data

def softmax(x):
    y = np.exp(x)/np.sum(np.exp(x))
    return y

def LSTM(fixed_layer,float_layer,file_number,units,xt_shape,ht_shape,dir_choose):
    
    #LSTM CELL ULP TEST
    lstm_total_bias = read_parameter(file_serial="lstm",file_name="f_bias",file_number=file_number)
    bias_i = lstm_total_bias[:128].T
    bias_f = lstm_total_bias[128:256].T
    bias_C = lstm_total_bias[256:384].T
    bias_O = lstm_total_bias[384:].T
    
    lstm_xh_weight = read_parameter(file_serial="lstm",file_name="f_x",file_number=file_number).reshape(xt_shape[0],xt_shape[1])#512,512
    lstm_ht_weight = read_parameter(file_serial="lstm",file_name="f_h",file_number=file_number).reshape(ht_shape[0],ht_shape[1])#128,512
    mse_array = []
    next_layer_array = []
    try:
        os.stat("./"+fix_point_dir+"/lstm_"+str(file_number)+"_floating_answer")
    except:
        os.mkdir("./"+fix_point_dir+"/lstm_"+str(file_number)+"_floating_answer")
    try:
        os.stat("./"+fix_point_dir+"/lstm_"+str(file_number)+"_fixed_answer")
    except:
        os.mkdir("./"+fix_point_dir+"/lstm_"+str(file_number)+"_fixed_answer")
        
    float_array = []
    for layer_number in range(2):
        fixed_array = []
        ct_1 = np.zeros([1,units])
        ht_1 = np.zeros([1,units])
        #print("float_layer_shape is {}".format(float_layer.shape))
        #print("fixed_layer_shape is {}".format(fixed_layer.shape))
        for cell in range(float_layer.shape[0]):
            if(layer_number==0):
                i_round = int((xt_shape[0]+ht_shape[0]) / 8);
                for i in range(i_round):
                    try:
                        os.stat("./sw_answer_dir_"+str(dir_choose)+"/cell_"+str(cell))
                    except:
                        os.mkdir("./sw_answer_dir_"+str(dir_choose)+"/cell_"+str(cell))
                    file_name ="./sw_answer_dir_"+str(dir_choose)+"/cell_"+str(cell)+"/" + str(i) + "_float_8_input_weight_output.txt"
                    with open(file_name,'w') as f:
                        x_begin = i * 8;
                        x_end = i * 8 + 8; 
                        print("-------input--------",file=f)
                        if(i<int(xt_shape[0]/8)):
                            print(float_layer[cell][i*8:i*8+8],file=f)
                        else:
                            i_index = i - int(xt_shape[0]/8)
                            print(ht_1[0][i_index*8:i_index*8+8],file=f)
                        for y in range(xt_shape[0]):
                            print("------lstm_xt_weight--------",file=f)
                            print(lstm_xh_weight[y][x_begin:x_end],file=f)
                        for z in range(ht_shape[0]):
                            print("------lstm_ht_weight--------",file=f)
                            print(lstm_ht_weight[z][x_begin:x_end],file=f)
                            print("--------------",file=f)
                try:
                    os.stat("./sw_answer_dir_dot1x1_"+str(dir_choose)+"/cell_"+str(cell))
                except:
                    os.mkdir("./sw_answer_dir_dot1x1_"+str(dir_choose)+"/cell_"+str(cell))
                for i in range(int(xt_shape[1])):
                    tmp_total = 0;
                    file_name_dot = "./sw_answer_dir_dot1x1_"+str(dir_choose)+"/cell_"+str(cell)+"/"+str(i) + "_dot.txt"
                    with open(file_name_dot,'w') as d:
                        for j in range(int(xt_shape[0])):
                            tmp = float_layer[cell][j] * lstm_xh_weight[j][i]
                            print(tmp,file=d);
                            tmp_total = tmp_total + tmp
                        for j in range(int(ht_shape[0])):
                            tmp = ht_1[0][j] * lstm_ht_weight[j][i]
                            print(tmp,file=d);
                            tmp_total = tmp_total + tmp
                        print("--------total_answer----------",file=d)
                        print(tmp_total,file=d)


                        
                input_xt = np.dot(float_layer[cell],lstm_xh_weight)
                input_ht_1 = np.dot(ht_1,lstm_ht_weight)
                input_value = input_xt + input_ht_1
                #np.savetxt("./lstm_connect_answer_without_activatefunc/"+str(cell)+".csv",input_value,delimiter=",")
                ft = sigmoid(input_value[:,128:256]+bias_f)
                #print("ft shape is {}".format(ft.shape))
                #ft = hard_sigmoid(input_value[:,128:256]+bias_f,0)
                np.savetxt("./"+fix_point_dir+"/lstm_"+str(file_number)+"_floating_answer/"+str(cell)+"_ft_number.csv",ft,delimiter=",")
                it = sigmoid(input_value[:,:128]+bias_i)
                #print("it shape is {}".format(it.shape))
                #it = hard_sigmoid(input_value[:,:128]+bias_i,0)
                np.savetxt("./"+fix_point_dir+"/lstm_"+str(file_number)+"_floating_answer/"+str(cell)+"_it_number.csv",it,delimiter=",")
                ct = tanh(input_value[:,256:384]+bias_C)
                #print("ct shape is {}".format(ct.shape))
                #ct = hard_tanh(input_value[:,256:384]+bias_C,0)
                np.savetxt("./"+fix_point_dir+"/lstm_"+str(file_number)+"_floating_answer/"+str(cell)+"_ct_pre_number.csv",ct,delimiter=",")
                ct = ft*ct_1+it*ct
                np.savetxt("./"+fix_point_dir+"/lstm_"+str(file_number)+"_floating_answer/"+str(cell)+"_ct_aft_number.csv",ct,delimiter=",")
                ot = sigmoid(input_value[:,384:]+bias_O)
                #print("ot shape is {}".format(ot.shape))
                #ot = hard_sigmoid(input_value[:,384:]+bias_O,0)
                np.savetxt("./"+fix_point_dir+"/lstm_"+str(file_number)+"_floating_answer/"+str(cell)+"_ot_number.csv",ot,delimiter=",")
                ht = ot * tanh(ct)
                #print("ht shape is {}".format(ht.shape))
                #ht = ot * hard_tanh(ct,0)
                np.savetxt("./"+fix_point_dir+"/lstm_"+str(file_number)+"_floating_answer/"+str(cell)+"_ht_number.csv",ht,delimiter=",")
                ht_1 = ht
                ct_1 = ct
                np.savetxt("./"+fix_point_dir+"/lstm_"+str(file_number)+"_floating_answer/"+str(cell)+"_number.csv",ht,delimiter=",")
                float_array.append(ht.reshape(units))
                #if(cell==30):
                #    print(ct)
                #    print("---------------------")
                #    print(ht)
            elif(layer_number!=0):
                shift_lstm_xh_weight = (lstm_xh_weight * 2 **layer_number).astype("int64")
                shift_lstm_ht_weight = (lstm_ht_weight * 2 **layer_number).astype("int64")
                shift_layer          = (fixed_layer  * 2 **layer_number).astype("int64")
                shift_bias_i         = (bias_i * 2 **layer_number).astype("int64")
                shift_bias_f         = (bias_f * 2 **layer_number).astype("int64")
                shift_bias_C         = (bias_C * 2 **layer_number).astype("int64")
                shift_bias_O         = (bias_O * 2 **layer_number).astype("int64")
                
                input_xt = (np.dot(shift_layer[cell],shift_lstm_xh_weight)/2**layer_number)#.astype("int64")
                input_ht_1 = (np.dot(ht_1,shift_lstm_ht_weight)/2**layer_number)#.astype("int64")
                input_value = (input_xt + input_ht_1) #2
                ft = ((fixed_sigmoid_lookuptable(input_value[:,128:256]+shift_bias_f,layer_number)))
                np.savetxt("./"+fix_point_dir+"/lstm_"+str(file_number)+"_fixed_answer/"+str(layer_number)+"_"+str(cell)+"_ft_number.csv",ft,delimiter=",")
                it = ((fixed_sigmoid_lookuptable(input_value[:,:128]+shift_bias_i,layer_number)))
                np.savetxt("./"+fix_point_dir+"/lstm_"+str(file_number)+"_fixed_answer/"+str(layer_number)+"_"+str(cell)+"_it_number.csv",it,delimiter=",")
                ct = ((fixed_tanh_lookuptable(input_value[:,256:384]+shift_bias_C,layer_number)))
                np.savetxt("./"+fix_point_dir+"/lstm_"+str(file_number)+"_fixed_answer/"+str(layer_number)+"_"+str(cell)+"_ct_pre_number.csv",ct,delimiter=",")
                ct = (((ft*ct_1)+(it*ct)))#.astype("int64")#2
                np.savetxt("./"+fix_point_dir+"/lstm_"+str(file_number)+"_fixed_answer/"+str(layer_number)+"_"+str(cell)+"_ct_aft_number.csv",ct,delimiter=",")
                ot = ((fixed_sigmoid_lookuptable(input_value[:,384:]+shift_bias_O,layer_number)))
                np.savetxt("./"+fix_point_dir+"/lstm_"+str(file_number)+"_fixed_answer/"+str(layer_number)+"_"+str(cell)+"_ot_number.csv",ot,delimiter=",")
                ht = ((ot * fixed_tanh_lookuptable(ct,layer_number)))
                np.savetxt("./"+fix_point_dir+"/lstm_"+str(file_number)+"_fixed_answer/"+str(layer_number)+"_"+str(cell)+"_ht_number.csv",ot,delimiter=",")
                ht_1 = ht
                '''
                print("----------------xt---------------- {}".format(layer_number))
                if(input_xt.any()==0):
                    print("what_the_fuck")
                else:
                    print(input_xt)
                print("----------------ht---------------- {}".format(layer_number))
                if(input_ht_1.any()==0):
                    print("what_the_fuck")
                else:
                    print(input_ht_1)
                '''
                #print("----------------ft---------------- {}".format(layer_number))
                #print(ht)
                ct_1 = ct
                np.savetxt("./"+fix_point_dir+"/lstm_"+str(file_number)+"_fixed_answer/"+str(layer_number)+"_"+str(cell)+"_number.csv",ht,delimiter=",")
                fixed_array.append(ht.reshape(units))
            else:
                print("please check your code why your going here !!!!!!")
        #mse_array to find best fixed point location
        if(layer_number!=0):
            mse_array.append(np.square(np.array(float_array) - np.array(fixed_array)).mean())
            next_layer_array.append(fixed_array)
    print("the list of mse_array is {}".format(mse_array))
    print("the most value in mse_array is {}".format(min(mse_array)))
    print("the most value in mse_array index is << {}".format(mse_array.index(min(mse_array))+1))
    print(np.array(float_array).shape)
    return np.array(float_array) , np.array(next_layer_array[mse_array.index(min(mse_array))])
            
