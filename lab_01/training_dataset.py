NUMBER_COUNT = 29


class Data:
    def __init__(self, inputs, result=None):
        self.inputs = inputs
        self.results = [-1] * NUMBER_COUNT
        if result is not None:
            self.results[result] = 1

values = {
    '00110000011000010010010000101111110100001010000101000010': 0, # А
    '11111001000000100000011111001000010100001010000101111100': 1, # Б
    '11111001000010100001011111001000010100001010000101111100': 2,
    '11111001000010100000010000001000000100000010000001000000': 3,
    '00110000100100010010001001000100100010010011111101000010': 4,
    '11111101000000100000011111001000000100000010000001111110': 5,
    '11010110101010010101000111000011100010101001010101101011': 6,
    '01111001000010000001000111000000010000001010000100111100': 7,
    '10000101000010100011010010101010010110001010000101000010': 8,
    '10000101000100100100010100001110000100100010001001000010': 9,
    '00001100001010001001000100100010010001001001000101000010': 10,
    '10000011100011101010110010011000001100000110000011000001': 11,
    '10000101000010100001011111101000010100001010000101000010': 12,
    '01111001000010100001010000101000010100001010000100111100': 13,
    '11111101000010100001010000101000010100001010000101000010': 14,
    '11111001000010100001011111001000000100000010000001000000': 15,
    '01111001000010100000010000001000000100000010000100111100': 16,
    '11111000010000001000000100000010000001000000100000010000': 17,
    '10000101000010100001001111100000010000001010000100111100': 18,
    '00010000111110100100110010010111110000100000010000001000': 19,
    '10000010100010001010000010000001000001010001000101000001': 20,
    '10000101000010100001001111100000010000001000000100000010': 21,
    '10010011001001100100110010011001001100100110010011111111': 22,
    '11000000100000010000001111000100001010000101000010111110': 23,
    '10000011000001100000111110011000101100010110001011111001': 24,
    '01000000100000010000001111000100001010000101000010111110': 25,
    '01111001000010000001000111100000010000001010000100111100': 26,
    '10011101010001101000111100011010001101000110100011001110': 27,
    '01111101000010100001001111100001010001001001000101000010': 28
}

dataset = [
    Data([int(ch) for ch in string], result)
    for string, result in values.items()
]

# Датасет для тестирования
test_values = {
    '00110000010000010010010000101111110100001010000101000001': 0,
    '11111101000000100000011111001000010100001010000101111100': 1,
    '11111001000010100001011111001000010100001010000100111100': 2,
    '11111001000010100000010000000000000100000010000001000000': 3,
    '00110000100100010010001001000100100010010011101101000010': 4,
    '11111101000000100000011011001000000100000010000001111110': 5,
    '11010110101010010101000111000010100010101001010100100011': 6,
    '01111001000010000001000111000000010000001010000100110100': 7,
    '10000101000010100011010000101010010110001010000101000010': 8,
    '10000011000100100100010100001110000100100010001001000010': 9,
    '00001100001010001001000100100010010001001001000101000000': 10,
    '10000011100011101010110010011000001100000110000011000000': 11,
    '10000101000010100001011011101000010100001010000101000010': 12,
    '01101001000010100001010000101000010100001010000100111100': 13,
    '11111101000010100001010000101000010100001010000101000000': 14,
    '11011001000010100001011111001000000100000010000001000000': 15,
    '01101001000010100000010000001000000100000010000100111100': 16,
    '11111110010000001000000100000010000001000000100000010000': 17,
    '10000101000010100001001110100000010000001010000100111100': 18,
    '00010000111110100100110010010110110000100000010000001000': 19,
    '10000010100010001010000010000001000001010001000101001001': 20,
    '10000101000010100001001111100000010000001000000100000000': 21,
    '10010011001001100100110010011001001100100110010011110111': 22,
    '11000000000000010000001111000100001010000101000010111110': 23,
    '10000011000001100000111110011000101100010110001001111001': 24,
    '01000000100000010000001111000100001010000101000010110110': 25,
    '01111101000010000001000111100000010000001010000100111100': 26,
    '10011101010001101000111100011010001101000110100011001011': 27,
    '01111101000010100001001111100001010001001001000101000000': 28,
}

test_dataset = [
    Data([int(ch) for ch in string], result)
    for string, result in test_values.items()
]
