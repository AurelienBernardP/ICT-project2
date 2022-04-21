# Write here your codes for questions 16 to 21 (you may delete this comment)
# From here, you may import either opencv (cv2) or the Python Imaging Library (PIL), but no other extra libraries.

from PIL import Image,ImageOps
import sys
import numpy as np

def read_greyscale_image(path):

    img = Image.open(path) 
    if img is None:
        sys.exit("Could not read the image.")

    grey_scale = ImageOps.grayscale(img)
    grey_scale.show() 

    return grey_scale

def save_image(img,path):
    if img is None or path is None:
        sys.exit("Could not save Image.")

    img.save(path, "PNG")

    return

def encode_image(img):
    return np.matrix.flatten(np.unpackbits(np.asarray(img),axis = 1))

def decode_image(image_as_bits,shape):
    return np.reshape(np.packbits(image_as_bits,axis = 0),shape)

def hamming_sequence_encoding(sequence):
    #pad the sequence to be a multiple of 4
    padding_len = len(sequence) % 4
    np.append(sequence,np.zeros(padding_len, dtype=np.int8))

    #apply hamming code to every 4 char
    hamming_sequence = np.zeros(len(sequence)//4*7,dtype=np.int8)
    for i in range(0,len(sequence),4):
        code = hamming_code(sequence[i:i+4])

        hamming_sequence[((i//4)*7):((i//4)*7)+7] = code

    return hamming_sequence

def hamming_sequence_decoding(sequence,original_nb_bits):
    decoded_sequence = np.zeros(len(sequence)//7 * 4,dtype=np.int8)
    nb_corrections = 0 

    for i in range(0,len(sequence),7):
        decoded_bits = decode_hamming(sequence[i:i+7])
        if np.array_equal(decoded_bits, sequence[i:i+4]):
            None
        else :
            nb_corrections += 1

        decoded_sequence[(i//7) *4 : ((i//7) * 4) + 4] = decoded_bits
        
    #remove padding done at source
    return decoded_sequence[0:original_nb_bits]

def sequence_through_channel(original_sequence):
    noisy_sequence = original_sequence

    for i, bit in enumerate(noisy_sequence):
        noisy_sequence[i] = noisy_channel(bit)

    return noisy_sequence

def noisy_channel(bit):
    if np.random.rand() > 0.01 :
        return bit
    else:
        return (bit + 1) % 2

def hamming_code(bits):
    if len(bits) != 4:
        print('error: wrong number of bits given')
        return None

    parity = np.zeros(3,dtype=np.int8)
    for i in range(len(parity)):
        parity[i] = (bits[i%4] + bits[(i+1)%4] + bits[(i+2)%4]) % 2

    code = np.append(bits,parity)
    return code

def decode_hamming(code):
    if len(code) != 7:
        print('error: wrong number of bits given')
        return None

    bits = np.copy(code[0:4])
    received_parity = code[4:7]
    computed_parity = (hamming_code(code[0:4]))[4:7]

    syndrome = np.zeros(3,dtype=np.int8)
    nb_errors = 0 

    for i in range(len(received_parity)):

        if(computed_parity[i] != received_parity[i]):
            syndrome[i] = 1
            nb_errors += 1

    if nb_errors == 3:
        bits[2] = (code[2] + 1) % 2

    if nb_errors == 2:
        if syndrome[0] and syndrome[1]:
            bits[1] = (code[1] + 1) % 2

        if syndrome[1] and syndrome[2]:
            bits[3] = (code[3] + 1) % 2

        if syndrome[2] and syndrome[0]:
            bits[0] = (code[0] + 1) % 2

    return bits


def number_of_differences(seq1,seq2):
    nb_diff = 0
    for i in range(len(seq1)):
        if seq1[i] != seq2[i]:
            nb_diff += 1
    
    return nb_diff

def naive_hamming_seq_decoding(sequence,original_nb_bits):

    decoded_sequence = np.zeros(len(sequence)//7 * 4,dtype=np.int8)

    for i in range(0,len(sequence),7):
        decoded_sequence[(i//7) *4 : ((i//7) * 4) + 4] = sequence[i:i+4]
        
    #remove padding done at source
    return decoded_sequence[0:original_nb_bits]

######################################## Code to answer the questions ##############

#### 16 :Load image and show ####################

original_image = read_greyscale_image("image.png")

#### 17 :Encode image using 1byte/pixel #########

image_as_sequence = encode_image(original_image)

#### 18 :Simulate channel and decode sequence ###
im_width, im_height = original_image.size

after_channel = decode_image(sequence_through_channel(image_as_sequence),(im_height,im_width))
Image.fromarray(after_channel).show()

save_image(Image.fromarray(after_channel),'noisy.png')

#### 19 :Encode image using Hamming code #########
hamming_sequence = hamming_sequence_encoding(image_as_sequence)

#### 20 :Simulate channel on Hamming sequence ####
hamming_after_channel = sequence_through_channel(hamming_sequence)

decoded_hamming = hamming_sequence_decoding(hamming_after_channel,len(image_as_sequence))
naive_decoded_hamming = naive_hamming_seq_decoding(hamming_after_channel,len(image_as_sequence))

print("errors without hamming decoding compared to original image = " + str(number_of_differences(image_as_sequence, naive_decoded_hamming)))
print("errors with  hamming decoding compared to original image = " + str(number_of_differences(image_as_sequence, decoded_hamming)))

Image.fromarray(decode_image(decoded_hamming,(im_height,im_width))).show()
save_image(Image.fromarray(decode_image(decoded_hamming,(im_height,im_width))),'postHamming_decoded.png')
save_image(Image.fromarray(decode_image(naive_decoded_hamming,(im_height,im_width))),'naive_Hamming_decoding.png')
